#!/usr/bin/env python3
"""Week 1 - turn downloaded sequences into trimmed raw 8-bit 4:2:0 .yuv files
plus a manifest that the encoding scripts read.

HM reads raw planar YUV only (no .y4m), so .y4m sources are converted with
ffmpeg. UVG files are already raw .yuv and are trimmed directly.

Examples
--------
  # 33 frames (1 intra + 32 inter in low-delay P), all downloaded sets
  python prepare_sequences.py

  # longer clips of the UVG set only
  python prepare_sequences.py --sets uvg --frames 97

  # 120 fps UVG has very small per-frame motion; keep every 2nd frame (=60 fps)
  python prepare_sequences.py --sets uvg --step 2 --tag 60fps
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "seq"
UVG_RE = re.compile(r"(?P<name>[A-Za-z]+)_(?P<w>\d+)x(?P<h>\d+)_(?P<fps>\d+)fps_420_8bit")


def frame_bytes(w: int, h: int) -> int:
    return w * h * 3 // 2  # 8-bit 4:2:0


def probe_y4m(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=width,height,r_frame_rate,nb_read_packets,pix_fmt",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    return {"width": int(s["width"]), "height": int(s["height"]),
            "fps": float(Fraction(s["r_frame_rate"])),
            "frames": int(s["nb_read_packets"]), "pix_fmt": s.get("pix_fmt", "?")}


def convert_y4m(src: Path, dst: Path, frames: int, step: int) -> None:
    vf = f"select='not(mod(n\\,{step}))',setpts=N/TB," if step > 1 else ""
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(src),
         "-vf", f"{vf}format=yuv420p", "-frames:v", str(frames),
         "-f", "rawvideo", str(dst)],
        check=True)


def trim_yuv(src: Path, dst: Path, w: int, h: int, frames: int, step: int) -> int:
    fb = frame_bytes(w, h)
    total = src.stat().st_size // fb
    written = 0
    with src.open("rb") as fi, dst.open("wb") as fo:
        for idx in range(0, total, step):
            if written == frames:
                break
            fi.seek(idx * fb)
            fo.write(fi.read(fb))
            written += 1
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", default=["small", "uvg", "xiphhd"],
                    help="sub-folders of data/raw to process (missing ones are skipped)")
    ap.add_argument("--frames", type=int, default=33, help="frames per output clip (default 33)")
    ap.add_argument("--step", type=int, default=1, help="keep every N-th frame (default 1)")
    ap.add_argument("--tag", default="", help="suffix for the manifest name, e.g. '60fps'")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []

    for group in args.sets:
        gdir = RAW / group
        if not gdir.is_dir():
            print(f"skip {group}: {gdir} not found")
            continue
        for src in sorted(list(gdir.glob("*.y4m")) + list(gdir.glob("**/*.yuv"))):
            if src.suffix == ".y4m":
                info = probe_y4m(src)
                name = src.stem.split("_")[0] if group == "small" else src.stem
                w, h = info["width"], info["height"]
                fps, avail = info["fps"], info["frames"]
                if info["pix_fmt"] not in ("yuv420p", "?"):
                    print(f"  note: {src.name} is {info['pix_fmt']}, converting to 4:2:0 8-bit")
            else:
                m = UVG_RE.search(src.name)
                if not m:
                    print(f"  skip {src.name}: cannot parse WxH/fps from file name")
                    continue
                name, w, h, fps = m["name"], int(m["w"]), int(m["h"]), float(m["fps"])
                avail = src.stat().st_size // frame_bytes(w, h)

            n = min(args.frames, (avail + args.step - 1) // args.step)
            out_fps = fps / args.step
            fps_str = f"{out_fps:g}"
            dst = OUT / f"{name}_{w}x{h}_{fps_str}fps_{n}f.yuv"

            if dst.exists() and dst.stat().st_size == n * frame_bytes(w, h):
                print(f"  ok   {dst.name} (exists)")
            else:
                print(f"  make {dst.name}")
                if src.suffix == ".y4m":
                    convert_y4m(src, dst, n, args.step)
                else:
                    trim_yuv(src, dst, w, h, n, args.step)
                got = dst.stat().st_size // frame_bytes(w, h)
                if got != n:
                    print(f"  ERROR: expected {n} frames, got {got}", file=sys.stderr)
                    return 1

            manifest.append({"name": name, "group": group, "path": str(dst),
                             "width": w, "height": h, "fps": out_fps, "frames": n,
                             "bit_depth": 8, "source": str(src)})

    if not manifest:
        print("No sequences found. Run ./download_data.sh first.")
        return 1

    mname = "sequences" + (f"_{args.tag}" if args.tag else "") + ".json"
    (ROOT / mname).write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {mname} with {len(manifest)} sequences:")
    for s in manifest:
        print(f"  {s['group']:7s} {s['name']:16s} {s['width']}x{s['height']} "
              f"{s['fps']:g} fps, {s['frames']} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
