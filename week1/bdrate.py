#!/usr/bin/env python3
"""Week 1 - Bjontegaard delta rate (BD-rate) and BD-PSNR between two runs.

BD-rate = average bitrate difference at equal quality (negative = test saves
bits). Two interpolation methods are provided:
  pchip  piecewise cubic Hermite interpolation (what the JVET CTC Excel sheet
         and most recent codec papers use)        [default]
  cubic  the original 3rd-order polynomial fit of VCEG-M33 (Bjontegaard 2001)

Also reports the encoding-time ratio (geometric mean of per-sequence
test/anchor encoder CPU time, as in JVET "EncT").

Examples
--------
  python bdrate.py results/anchor_lowdelay_P.csv results/qnet_lowdelay_P.csv
  python bdrate.py A.csv B.csv --metric yuv611 --method cubic
  python bdrate.py --selftest
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict

import numpy as np
from scipy.interpolate import PchipInterpolator


def _integral_pchip(x: np.ndarray, y: np.ndarray, lo: float, hi: float) -> float:
    order = np.argsort(x)
    f = PchipInterpolator(x[order], y[order])
    return float(f.integrate(lo, hi))


def _integral_cubic(x: np.ndarray, y: np.ndarray, lo: float, hi: float) -> float:
    p = np.polyint(np.polyfit(x, y, 3))
    return float(np.polyval(p, hi) - np.polyval(p, lo))


INTEGRALS = {"pchip": _integral_pchip, "cubic": _integral_cubic}


def bd_rate(r_a, d_a, r_t, d_t, method="pchip") -> float:
    """Percent rate change of test vs anchor at equal distortion (PSNR)."""
    la, lt = np.log10(np.asarray(r_a, float)), np.log10(np.asarray(r_t, float))
    d_a, d_t = np.asarray(d_a, float), np.asarray(d_t, float)
    lo, hi = max(d_a.min(), d_t.min()), min(d_a.max(), d_t.max())
    if hi <= lo:
        return float("nan")  # RD curves do not overlap in quality
    integ = INTEGRALS[method]
    avg = (integ(d_t, lt, lo, hi) - integ(d_a, la, lo, hi)) / (hi - lo)
    return (10 ** avg - 1) * 100


def bd_psnr(r_a, d_a, r_t, d_t, method="pchip") -> float:
    """Average PSNR change (dB) of test vs anchor at equal rate."""
    la, lt = np.log10(np.asarray(r_a, float)), np.log10(np.asarray(r_t, float))
    d_a, d_t = np.asarray(d_a, float), np.asarray(d_t, float)
    lo, hi = max(la.min(), lt.min()), min(la.max(), lt.max())
    if hi <= lo:
        return float("nan")
    integ = INTEGRALS[method]
    return (integ(lt, d_t, lo, hi) - integ(la, d_a, lo, hi)) / (hi - lo)


def enc_time(row: dict) -> float:
    """Encoder CPU time: HM's "Total Time", measured with clock(), i.e. CPU time of the
    encoder process. Wall-clock time (enc_wall_s) depends on what else runs on the
    machine, so it is only a fallback for rows that lack hm_time_s."""
    v = row.get("hm_time_s")
    return float(v) if v not in (None, "", "None") else float(row["enc_wall_s"])


def load(path: str) -> dict:
    data = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("ok", "True") not in ("True", "true", "1"):
                continue
            key = (row["group"], row["name"], row["width"], row["height"])
            data[key].append(row)
    return data


def compare(anchor_csv: str, test_csv: str, metric: str, method: str) -> int:
    col = {"y": "psnr_y", "yuv611": "psnr_yuv611", "u": "psnr_u", "v": "psnr_v"}[metric]
    A, T = load(anchor_csv), load(test_csv)
    common = sorted(set(A) & set(T))
    if not common:
        print("No sequences in common between the two files.")
        return 1

    print(f"Anchor: {anchor_csv}\nTest:   {test_csv}\nMetric: PSNR-{metric}, method: {method}\n")
    print(f"{'group':7s} {'sequence':16s} {'res':>10s} {'BD-rate %':>10s} {'BD-PSNR dB':>11s} {'EncT %':>8s}")
    per_group = defaultdict(list)
    for key in common:
        a = {int(r["qp"]): r for r in A[key]}
        t = {int(r["qp"]): r for r in T[key]}
        qps = sorted(set(a) & set(t))
        if len(qps) < 4:
            print(f"{key[0]:7s} {key[1]:16s}  skipped: only {len(qps)} matching QPs (need 4)")
            continue
        ra = [float(a[q]["kbps"]) for q in qps]
        rt = [float(t[q]["kbps"]) for q in qps]
        da = [float(a[q][col]) for q in qps]
        dt = [float(t[q][col]) for q in qps]
        bdr, bdp = bd_rate(ra, da, rt, dt, method), bd_psnr(ra, da, rt, dt, method)
        ta = sum(enc_time(a[q]) for q in qps)
        tt = sum(enc_time(t[q]) for q in qps)
        per_group[key[0]].append((bdr, bdp, tt / ta))
        print(f"{key[0]:7s} {key[1]:16s} {key[2] + 'x' + key[3]:>10s} {bdr:10.2f} {bdp:11.3f} {100 * tt / ta:8.1f}")

    print("-" * 66)
    allv = []
    for g, vals in sorted(per_group.items()):
        allv += vals
        print(f"{'avg':7s} {g:16s} {'':>10s} {np.nanmean([v[0] for v in vals]):10.2f} "
              f"{np.nanmean([v[1] for v in vals]):11.3f} "
              f"{100 * math.exp(np.mean([math.log(v[2]) for v in vals])):8.1f}")
    if len(per_group) > 1:
        print(f"{'avg':7s} {'ALL':16s} {'':>10s} {np.nanmean([v[0] for v in allv]):10.2f} "
              f"{np.nanmean([v[1] for v in allv]):11.3f} "
              f"{100 * math.exp(np.mean([math.log(v[2]) for v in allv])):8.1f}")
    print("\nBD-rate < 0: test needs fewer bits for the same PSNR (better).")
    print("EncT: test encoder CPU time as % of anchor (geometric mean over sequences).")
    return 0


def selftest() -> int:
    ok = True
    r = [1000, 1800, 3300, 6000]
    d = [32.0, 34.5, 37.0, 39.4]
    for m in INTEGRALS:
        # identical curves -> 0
        ok &= abs(bd_rate(r, d, r, d, m)) < 1e-9
        # test uses exactly 10% fewer bits at every point -> -10%
        ok &= abs(bd_rate(r, d, [x * 0.9 for x in r], d, m) + 10) < 1e-6
        # test is +0.5 dB everywhere -> BD-PSNR +0.5
        ok &= abs(bd_psnr(r, d, r, [x + 0.5 for x in d], m) - 0.5) < 1e-6
    try:  # cross-check with the 'bjontegaard' package if installed
        import bjontegaard as bdpkg
        rt, dt = [950, 1750, 3150, 5900], [32.1, 34.7, 37.1, 39.6]
        for m in INTEGRALS:
            mine, ref = bd_rate(r, d, rt, dt, m), bdpkg.bd_rate(r, d, rt, dt, method=m)
            print(f"  {m}: ours {mine:.4f}%  bjontegaard pkg {ref:.4f}%")
            ok &= abs(mine - ref) < 1e-3
    except ImportError:
        print("  (bjontegaard package not installed; skipped cross-check)")
    print("selftest", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("anchor", nargs="?")
    ap.add_argument("test", nargs="?")
    ap.add_argument("--metric", default="y", choices=["y", "yuv611", "u", "v"])
    ap.add_argument("--method", default="pchip", choices=list(INTEGRALS))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.anchor and a.test):
        ap.error("give an anchor CSV and a test CSV (or --selftest)")
    return compare(a.anchor, a.test, a.metric, a.method)


if __name__ == "__main__":
    sys.exit(main())
