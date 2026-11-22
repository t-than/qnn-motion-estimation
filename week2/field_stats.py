#!/usr/bin/env python3
"""Week 2 - compare two motion fields (.mvf), e.g. RAFT's against HM's final coded vectors.

For every sequence of a manifest (and every QP, if a path uses {qp}), over the entries
(POC, reference POC) present in both fields and the 4x4 cells valid in both:

  epe_mean, epe_median   endpoint error |mv_a - mv_b| in pixels
  within1, within_half   % of cells with an endpoint error <= 1 pel / <= 0.5 pel
  same_int               % of cells whose vectors round to the same integer-pel vector
                         (what --MVMode=replace hands to HM's fractional refinement)
  mag_a, mag_b           mean |mv| in pixels
  mc_a, mc_b, mc_zero    motion-compensation error: mean absolute luma error per pixel when
                         predicting frame POC from the original frame of the reference POC with
                         field A, field B or zero motion (bilinear at quarter-pel, edge-clamped)
  mc_a_all               the same for A over all of A's valid cells, not only the common ones
  valid_b                % of all cells valid in B (for HM's field: inter blocks coded from
                         that reference; intra, merge from other references etc. are invalid)

MC errors use the original frames, not the encoder's reconstruction, so they say how well a
field describes the motion, not what it costs to code.

Examples (from week2/, venv active):
  python field_stats.py --manifest ../week1/sequences.json --groups small --a mv/raft/{name}.mvf \\
      --b "results/dump_d2/lowdelay_P/{name}_{w}x{h}/QP{qp}/field.mvf" --out results/raft_field_stats.csv
  python field_stats.py --manifest ../week1/sequences_60fps.json --a mv/raft_60fps/{name}.mvf \\
      --b "results/hmdump60fps/lowdelay_P/{name}_{w}x{h}/QP{qp}/field.mvf" --out results/raft_60fps_field_stats.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mvfield import GRID, read_mvf  # noqa: E402


def read_luma(path, width: int, height: int, n: int) -> np.ndarray:
    """Luma of frame n of an 8-bit 4:2:0 .yuv file, uint8 (H, W)."""
    fs = width * height + 2 * ((width + 1) // 2) * ((height + 1) // 2)
    y = np.fromfile(path, np.uint8, width * height, offset=n * fs)
    if y.size != width * height:
        raise ValueError(f"{path}: frame {n} is missing (file too short)")
    return y.reshape(height, width)


def mc_mae(cur: np.ndarray, ref: np.ndarray, cells: np.ndarray, mask: np.ndarray | None = None) -> tuple:
    """Mean absolute error per pixel of predicting cur from ref with the cells' vectors
    (quarter-pel, bilinear interpolation, coordinates clamped to the picture like HM's padding).
    Only cells with confidence > 0 (and mask, if given, per cell) count. Returns (mae, pixels)."""
    h, w = cur.shape
    valid = cells[..., 2] > 0
    if mask is not None:
        valid &= mask
    rep = lambda a: np.repeat(np.repeat(a, GRID, 0), GRID, 1)[:h, :w]  # noqa: E731
    ys, xs = np.nonzero(rep(valid))
    if len(ys) == 0:
        return 0.0, 0
    gy, gx = ys // GRID, xs // GRID
    px = np.clip(xs + cells[gy, gx, 0] / 4.0, 0, w - 1)
    py = np.clip(ys + cells[gy, gx, 1] / 4.0, 0, h - 1)
    x0, y0 = np.floor(px).astype(np.int64), np.floor(py).astype(np.int64)
    fx, fy = px - x0, py - y0
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    r = ref.astype(np.float64)
    pred = ((1 - fy) * ((1 - fx) * r[y0, x0] + fx * r[y0, x1]) + fy * ((1 - fx) * r[y1, x0] + fx * r[y1, x1]))
    return float(np.abs(cur[ys, xs].astype(np.float64) - pred).mean()), int(len(ys))


def compare(seq: dict, path_a: Path, path_b: Path, frames: int | None) -> dict:
    fa, fb = read_mvf(path_a), read_mvf(path_b)
    w, h = seq["width"], seq["height"]
    for f, p in ((fa, path_a), (fb, path_b)):
        if (f.width, f.height) != (w, h):
            raise SystemExit(f"{p}: field is {f.width}x{f.height}, sequence is {w}x{h}")
    keys = sorted(k for k in fa.entries.keys() & fb.entries.keys() if frames is None or k[0] < frames)
    if not keys:
        raise SystemExit(f"{path_a} and {path_b} have no (POC, reference POC) entry in common")
    epe, same_int, mag_a, mag_b = [], [], [], []
    mc = {"a": [0.0, 0], "b": [0.0, 0], "zero": [0.0, 0], "a_all": [0.0, 0]}
    nvalid_b = ncells = 0
    for poc, rp in keys:
        a, b = fa.entries[(poc, rp)], fb.entries[(poc, rp)]
        both = (a[..., 2] > 0) & (b[..., 2] > 0)
        nvalid_b += int((b[..., 2] > 0).sum())
        ncells += b[..., 2].size
        va, vb = a[both][:, :2].astype(np.float64) / 4, b[both][:, :2].astype(np.float64) / 4
        epe.append(np.hypot(*(va - vb).T))
        same_int.append((np.rint(va) == np.rint(vb)).all(axis=1))
        mag_a.append(np.hypot(*va.T))
        mag_b.append(np.hypot(*vb.T))
        cur, ref = read_luma(seq["path"], w, h, poc), read_luma(seq["path"], w, h, rp)
        zero = a.copy()
        zero[..., :2] = 0
        zero[..., 2] = 1
        for k, cells, m in (("a", a, both), ("b", b, both), ("zero", zero, both), ("a_all", a, None)):
            e, n = mc_mae(cur, ref, cells, m)
            mc[k][0] += e * n
            mc[k][1] += n
    epe, same_int = np.concatenate(epe), np.concatenate(same_int)
    pct = lambda x: round(100.0 * float(x.mean()), 2) if x.size else float("nan")  # noqa: E731
    mean = lambda x: round(float(np.concatenate(x).mean()), 3) if sum(map(len, x)) else float("nan")  # noqa: E731
    return {
        "entries": len(keys), "cells": int(epe.size),
        "valid_b": round(100.0 * nvalid_b / max(ncells, 1), 2),
        "epe_mean": round(float(epe.mean()), 3) if epe.size else float("nan"),
        "epe_median": round(float(np.median(epe)), 3) if epe.size else float("nan"),
        "within1": pct(epe <= 1.0), "within_half": pct(epe <= 0.5), "same_int": pct(same_int),
        "mag_a": mean(mag_a), "mag_b": mean(mag_b),
        **{f"mc_{k}": round(v[0] / max(v[1], 1), 3) for k, v in mc.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--groups", nargs="*")
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--frames", type=int, help="only entries with POC < FRAMES")
    ap.add_argument("--a", required=True, help="field A path; may use {name} {w} {h} {qp}")
    ap.add_argument("--b", required=True, help="field B (reference) path; may use {name} {w} {h} {qp}")
    ap.add_argument("--qps", type=int, nargs="+", default=[22, 27, 32, 37], help="used if a path contains {qp}")
    ap.add_argument("--out", help="write the table as CSV")
    a = ap.parse_args()

    seqs = json.loads(Path(a.manifest).read_text())
    if a.groups:
        seqs = [s for s in seqs if s.get("group") in a.groups]
    if a.names:
        seqs = [s for s in seqs if s["name"] in a.names]
    if not seqs:
        raise SystemExit("no sequences selected")
    qps = a.qps if "{qp}" in a.a + a.b else [None]
    rows = []
    cols = ["sequence", "qp", "entries", "valid_b", "epe_mean", "epe_median", "within_half", "within1",
            "same_int", "mag_a", "mag_b", "mc_a", "mc_b", "mc_zero", "mc_a_all"]
    print(f"A = {a.a}\nB = {a.b}")
    print(f"{'sequence':14s} {'QP':>3s} {'B valid%':>8s} {'EPE px':>7s} {'med':>6s} {'<=0.5%':>7s} {'<=1%':>6s} "
          f"{'sameInt%':>8s} {'|A| px':>7s} {'|B| px':>7s} {'MC A':>6s} {'MC B':>6s} {'MC 0':>6s} {'MC A all':>8s}")
    for s in seqs:
        for q in qps:
            fmt = dict(name=s["name"], w=s["width"], h=s["height"], qp=q)
            pa, pb = Path(a.a.format(**fmt)), Path(a.b.format(**fmt))
            missing = [str(p) for p in (pa, pb) if not p.exists()]
            if missing:
                raise SystemExit(f"missing: {', '.join(missing)}")
            r = {"sequence": s["name"], "qp": q if q is not None else "", **compare(s, pa, pb, a.frames)}
            rows.append(r)
            print(f"{s['name']:14s} {str(r['qp']):>3s} {r['valid_b']:8.1f} {r['epe_mean']:7.2f} {r['epe_median']:6.2f} "
                  f"{r['within_half']:7.1f} {r['within1']:6.1f} {r['same_int']:8.1f} {r['mag_a']:7.2f} {r['mag_b']:7.2f} "
                  f"{r['mc_a']:6.2f} {r['mc_b']:6.2f} {r['mc_zero']:6.2f} {r['mc_a_all']:8.2f}")
    if a.out:
        with open(a.out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            wr.writeheader()
            wr.writerows(rows)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
