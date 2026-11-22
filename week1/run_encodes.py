#!/usr/bin/env python3
"""Week 1 - run HM encodes at several QPs, in parallel, with a decoder check.

For each (sequence, QP) job this script:
  1. encodes with HM (TAppEncoderStatic) using a standard HM .cfg,
  2. decodes the bitstream with the reference decoder (TAppDecoderStatic),
  3. checks that the encoder reconstruction and the decoder output are
     identical (and that the decoder's MD5 SEI checks pass),
  4. parses rate and PSNR from HM's SUMMARY and stores result.json.

Finished jobs are skipped on re-run, so an interrupted batch can simply be
restarted. All results of a run are collected in results/<variant>_<cfg>.csv.

Examples
--------
  # time a few frames first and get a wall-clock projection for the full plan
  python run_encodes.py --estimate

  # the anchor: unmodified HM, low-delay P, QP 22/27/32/37, 6 parallel jobs
  python run_encodes.py --variant anchor

  # only the small CIF set, random access
  python run_encodes.py --variant anchor --cfg randomaccess --groups small

  # later weeks: patched encoder + extra options (e.g. your MV-file switch)
  python run_encodes.py --variant qnet --encoder ./HM-mv/bin/TAppEncoderStatic \
      --extra="--MVFile={mvdir}/{name}.mv --MVMode=replace"

  (write --extra='...' with "=", because the value itself starts with "--")
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HM_DIR = ROOT / "HM"
CFGS = {
    "lowdelay_P": "encoder_lowdelay_P_main.cfg",
    "lowdelay_B": "encoder_lowdelay_main.cfg",
    "randomaccess": "encoder_randomaccess_main.cfg",
    "intra": "encoder_intra_main.cfg",
}
SUMMARY_RE = re.compile(
    r"^SUMMARY -+\s*\n.*\n\s*(\d+)\s+a\s+([-\d.naif]+)\s+([-\d.naif]+)\s+([-\d.naif]+)"
    r"\s+([-\d.naif]+)\s+([-\d.naif]+)", re.M)
BYTES_RE = re.compile(r"Bytes written to file:\s*(\d+)")
TIME_RE = re.compile(r"Total Time:\s*([\d.]+)\s*sec")


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def run_job(job: dict, args: argparse.Namespace) -> dict:
    seq, qp, jdir = job["seq"], job["qp"], Path(job["dir"])
    jdir.mkdir(parents=True, exist_ok=True)
    res_file = jdir / "result.json"
    if res_file.exists() and not args.force and not args.estimate:
        r = json.loads(res_file.read_text())
        if r.get("ok"):
            r["skipped"] = True
            return r

    bin_, rec, dec = jdir / "str.bin", jdir / "rec.yuv", jdir / "dec.yuv"
    frames = job["frames"]
    extra = args.extra.format(name=seq["name"], qp=qp, w=seq["width"], h=seq["height"],
                              frames=frames, mvdir=args.mvdir, dir=str(jdir))
    cmd = [str(args.encoder), "-c", str(args.cfg_path),
           "-i", seq["path"], "-wdt", str(seq["width"]), "-hgt", str(seq["height"]),
           "-fr", str(round(seq["fps"])), "-f", str(frames), "-q", str(qp),
           f"--InputBitDepth={seq['bit_depth']}", "--SEIDecodedPictureHash=1",
           "-b", str(bin_), "-o", str(rec)] + shlex.split(extra)

    t0 = time.time()
    with (jdir / "enc.log").open("w") as log:
        log.write("# " + shlex.join(cmd) + "\n")
        log.flush()
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    enc_wall = time.time() - t0
    enc_log = (jdir / "enc.log").read_text(errors="replace")

    result = {"name": seq["name"], "group": seq["group"], "width": seq["width"],
              "height": seq["height"], "fps": seq["fps"], "frames": frames, "qp": qp,
              "variant": args.variant, "cfg": args.cfg, "enc_wall_s": round(enc_wall, 2),
              "cmd": shlex.join(cmd), "ok": False}
    m = SUMMARY_RE.search(enc_log)
    if p.returncode != 0 or not m:
        result["error"] = f"encoder failed (exit {p.returncode}); see {jdir/'enc.log'}"
        res_file.write_text(json.dumps(result, indent=2))
        return result

    nbytes = int(BYTES_RE.search(enc_log).group(1)) if BYTES_RE.search(enc_log) else bin_.stat().st_size
    result.update({
        "bytes": nbytes,
        "kbps": nbytes * 8 * seq["fps"] / frames / 1000.0,   # exact, from file size
        "hm_kbps": float(m.group(2)),
        "psnr_y": float(m.group(3)), "psnr_u": float(m.group(4)),
        "psnr_v": float(m.group(5)), "psnr_yuv_hm": float(m.group(6)),
    })
    result["psnr_yuv611"] = (6 * result["psnr_y"] + result["psnr_u"] + result["psnr_v"]) / 8
    tm = TIME_RE.search(enc_log)
    result["hm_time_s"] = float(tm.group(1)) if tm else None

    # Decoder check: the bitstream must decode to exactly the encoder's reconstruction.
    with (jdir / "dec.log").open("w") as log:
        pd = subprocess.run([str(args.decoder), "-b", str(bin_), "-o", str(dec)],
                            stdout=log, stderr=subprocess.STDOUT)
    dec_log = (jdir / "dec.log").read_text(errors="replace")
    md5_ok = len(re.findall(r"\[MD5:[^\]]*\(OK\)\]", dec_log))
    md5_bad = dec_log.count("***ERROR***")
    match = pd.returncode == 0 and dec.exists() and file_md5(rec) == file_md5(dec)
    result.update({"decode_match": match, "sei_md5_ok_frames": md5_ok,
                   "sei_md5_errors": md5_bad})
    result["ok"] = bool(match and md5_bad == 0 and md5_ok == frames)
    if not result["ok"]:
        result["error"] = "DECODER MISMATCH: bitstream does not decode to the reconstruction"

    if not args.keep_recon:
        for f in (rec, dec):
            f.unlink(missing_ok=True)
    if args.estimate:
        result["sec_per_frame"] = enc_wall / frames
    else:
        res_file.write_text(json.dumps(result, indent=2))
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(ROOT / "sequences.json"))
    ap.add_argument("--variant", default="anchor", help="label for this set of results")
    ap.add_argument("--cfg", default="lowdelay_P", help=f"one of {list(CFGS)} or a path to a .cfg")
    ap.add_argument("--qps", type=int, nargs="+", default=[22, 27, 32, 37])
    ap.add_argument("--jobs", type=int, default=6, help="parallel encodes (default 6 = physical cores)")
    ap.add_argument("--groups", nargs="*", help="only these manifest groups (small, uvg, xiphhd)")
    ap.add_argument("--names", nargs="*", help="only these sequence names")
    ap.add_argument("--frames", type=int, help="encode fewer frames than in the manifest")
    ap.add_argument("--encoder", default=str(HM_DIR / "bin" / "TAppEncoderStatic"))
    ap.add_argument("--decoder", default=str(HM_DIR / "bin" / "TAppDecoderStatic"))
    ap.add_argument("--extra", default="", help="extra HM options; may use {name} {qp} {w} {h} {frames} {mvdir} {dir}")
    ap.add_argument("--mvdir", default=str(ROOT / "mv"), help="value for {mvdir} in --extra")
    ap.add_argument("--keep-recon", action="store_true", help="keep rec.yuv/dec.yuv (large!)")
    ap.add_argument("--force", action="store_true", help="re-run jobs that already have results")
    ap.add_argument("--estimate", action="store_true",
                    help="encode 9 frames at QP 32 of one sequence per resolution and project total time")
    args = ap.parse_args()

    args.cfg_path = Path(args.cfg) if args.cfg.endswith(".cfg") else HM_DIR / "cfg" / CFGS[args.cfg]
    for exe in (args.encoder, args.decoder, args.cfg_path):
        if not Path(exe).exists():
            print(f"missing: {exe}  (run ./build_hm.sh first)")
            return 1
    seqs = json.loads(Path(args.manifest).read_text())
    if args.groups:
        seqs = [s for s in seqs if s["group"] in args.groups]
    if args.names:
        seqs = [s for s in seqs if s["name"] in args.names]
    if not seqs:
        print("no sequences selected")
        return 1
    cfg_label = Path(args.cfg).stem if args.cfg.endswith(".cfg") else args.cfg
    outdir = ROOT / "results" / args.variant / cfg_label

    if args.estimate:
        return estimate(seqs, args, outdir)

    jobs = []
    for s in seqs:
        n = min(args.frames or s["frames"], s["frames"])
        for qp in args.qps:
            jobs.append({"seq": s, "qp": qp, "frames": n,
                         "dir": str(outdir / f"{s['name']}_{s['width']}x{s['height']}" / f"QP{qp}")})
    # largest jobs first so the pool does not end on one long straggler
    jobs.sort(key=lambda j: -j["seq"]["width"] * j["seq"]["height"] * j["frames"] * (60 - j["qp"]))

    print(f"{len(jobs)} jobs | variant={args.variant} cfg={cfg_label} | {args.jobs} parallel | "
          f"encoder={args.encoder}")
    t0, results, failed = time.time(), [], 0
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_job, j, args): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            results.append(r)
            tag = "skip" if r.get("skipped") else ("ok  " if r["ok"] else "FAIL")
            failed += not r["ok"]
            info = (f"{r['kbps']:9.2f} kbps  Y {r['psnr_y']:.3f} dB  {r['enc_wall_s']:7.1f} s"
                    if "kbps" in r else r.get("error", ""))
            print(f"[{i:3d}/{len(jobs)}] {tag} {r['name']:14s} QP{r['qp']}  {info}", flush=True)

    results.sort(key=lambda r: (r["group"], r["name"], r["qp"]))
    csv_path = ROOT / "results" / f"{args.variant}_{cfg_label}.csv"
    cols = ["variant", "cfg", "group", "name", "width", "height", "fps", "frames", "qp", "bytes",
            "kbps", "psnr_y", "psnr_u", "psnr_v", "psnr_yuv611", "psnr_yuv_hm",
            "enc_wall_s", "hm_time_s", "decode_match", "ok"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(r for r in results if "kbps" in r)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. {failed} failed. Results: {csv_path}")
    return 1 if failed else 0


def estimate(seqs: list, args: argparse.Namespace, outdir: Path) -> int:
    reps = {}
    for s in seqs:
        reps.setdefault((s["width"], s["height"]), s)
    print("Timing 9 frames at QP 32 for one sequence per resolution (single job each)...")
    spf = {}
    for (w, h), s in reps.items():
        r = run_job({"seq": s, "qp": 32, "frames": min(9, s["frames"]),
                     "dir": str(outdir / "_estimate" / f"{w}x{h}")}, args)
        if not r["ok"]:
            print(f"  {w}x{h}: failed - {r.get('error')}")
            return 1
        spf[(w, h)] = r["sec_per_frame"]
        print(f"  {w}x{h}: {r['sec_per_frame']:.2f} s/frame ({s['name']})")
    # Low QPs are slower than QP 32 in HM; 1.3x is a rough allowance.
    total = sum(spf[(s["width"], s["height"])] * min(args.frames or s["frames"], s["frames"])
                for s in seqs) * len(args.qps) * 1.3
    wall = total / max(1, min(args.jobs, os.cpu_count() or 1))
    fmt = lambda s: f"{s / 3600:.1f} h" if s >= 3600 else (f"{s / 60:.0f} min" if s >= 60 else "<1 min")
    print(f"\nProjected for {len(seqs)} sequences x {len(args.qps)} QPs: "
          f"~{fmt(total)} CPU time, ~{fmt(wall)} wall-clock with {args.jobs} parallel jobs.")
    print("(Rough: parallel jobs share memory bandwidth, so expect somewhat more.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
