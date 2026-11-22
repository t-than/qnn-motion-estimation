#!/usr/bin/env python3
"""Week 2 - compare the bitstreams of two encode runs, job by job.

Compares <dir>/<sequence>_<WxH>/QP<n>/str.bin between two result folders (as written
by run_encodes.py) by MD5 and reports, per job, whether the two bitstreams are
identical. Used for the no-op check (unpatched HM-mv vs. the week-1 anchor) and the
replay test (test 1a): both must be identical on every job.

  python compare_bitstreams.py ../week1/results/anchor/lowdelay_P results/noop/lowdelay_P

With --common, only jobs present in both folders are compared (e.g. a CIF-only run
against the full anchor).

Exit status: 0 if every compared job is identical (and, without --common, every job
present in either folder is in both), 1 otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def jobs(root: Path) -> dict[str, Path]:
    """sequence_WxH/QPn -> str.bin, skipping run_encodes.py's _estimate folder."""
    return {str(p.parent.relative_to(root)): p
            for p in sorted(root.glob("*/QP*/str.bin")) if not p.parts[-3].startswith("_")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path, help="first result folder, e.g. ../week1/results/anchor/lowdelay_P")
    ap.add_argument("b", type=Path, help="second result folder")
    ap.add_argument("-v", "--verbose", action="store_true", help="list identical jobs too")
    ap.add_argument("--common", action="store_true", help="compare only jobs present in both folders")
    args = ap.parse_args()

    A, B = jobs(args.a), jobs(args.b)
    if not A and not B:
        print(f"No */QP*/str.bin found in {args.a} or {args.b}")
        return 1
    same = diff = 0
    only_a, only_b = sorted(set(A) - set(B)), sorted(set(B) - set(A))
    if args.common:
        if not set(A) & set(B):
            print("No job in common between the two folders")
            return 1
        only_a, only_b = [], []
    for key in sorted(set(A) & set(B)):
        a, b = A[key], B[key]
        ok = a.stat().st_size == b.stat().st_size and md5(a) == md5(b)
        same += ok
        diff += not ok
        if not ok:
            print(f"DIFFERENT  {key}  ({a.stat().st_size} vs {b.stat().st_size} bytes)")
        elif args.verbose:
            print(f"identical  {key}")
    for key in only_a:
        print(f"MISSING    {key}  (only in {args.a})")
    for key in only_b:
        print(f"MISSING    {key}  (only in {args.b})")

    total = same + diff + len(only_a) + len(only_b)
    print(f"\n{same}/{total} identical, {diff} different, {len(only_a) + len(only_b)} missing")
    return 0 if same == total else 1


if __name__ == "__main__":
    sys.exit(main())
