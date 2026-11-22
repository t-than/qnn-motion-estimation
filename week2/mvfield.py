#!/usr/bin/env python3
"""Week 2 - motion-field (.mvf) and motion-search dump (.dump) files, in Python.

The same formats are written and read by HM-mv (source/Lib/TLibEncoder/TEncMvIO.cpp).
All little-endian.

.mvf (motion field), version 1
    header (24 bytes): b"QMVF", uint32 version=1, width, height, grid=4, number of entries
    entry: int32 POC, int32 reference POC, then (ceil(height/4), ceil(width/4)) cells in
           raster order, each int16 mv_x, int16 mv_y (quarter-pel), int16 confidence
           (0 = no vector, 1..1000)
    A vector points from the current block to its match in the reference picture:
    ref(x + mv/4) ~ cur(x), the same sign as optical flow from the current to the reference frame.

.dump (one record per call to HM's integer motion search), version 1
    header (16 bytes): b"QMED", uint32 version=1, uint32 record size=32, uint32 reserved
    record: int32 poc, ref_poc; int16 flags (bit 0: list L1, bit 1: bi-pred refinement),
            ref_idx, x, y, w, h, pred_x, pred_y (quarter-pel), int_x, int_y (integer pel),
            mv_x, mv_y (final, quarter-pel)

Library use:
    f = read_mvf("field.mvf")            # MotionField with .entries[(poc, ref_poc)] -> (H/4, W/4, 3) int16
    write_mvf("out.mvf", f)
    d = read_dump("me.dump")             # numpy structured array, one row per search

Command line:
    python mvfield.py info  FILE.mvf|FILE.dump   summary statistics
    python mvfield.py flip  IN.mvf OUT.mvf       negate every vector (negative control for test 1b)
    python mvfield.py selftest
"""
from __future__ import annotations

import argparse
import struct
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MVF_MAGIC, DUMP_MAGIC = b"QMVF", b"QMED"
GRID = 4
CONF_MAX = 1000

DUMP_DTYPE = np.dtype([("poc", "<i4"), ("ref_poc", "<i4"), ("flags", "<i2"), ("ref_idx", "<i2"),
                       ("x", "<i2"), ("y", "<i2"), ("w", "<i2"), ("h", "<i2"),
                       ("pred_x", "<i2"), ("pred_y", "<i2"), ("int_x", "<i2"), ("int_y", "<i2"),
                       ("mv_x", "<i2"), ("mv_y", "<i2")])
assert DUMP_DTYPE.itemsize == 32


@dataclass
class MotionField:
    width: int
    height: int
    # (poc, ref_poc) -> int16 array (grid_h, grid_w, 3): mv_x, mv_y (quarter-pel), confidence
    entries: dict = field(default_factory=dict)

    @property
    def grid_shape(self) -> tuple:
        return (-(-self.height // GRID), -(-self.width // GRID))

    def add(self, poc: int, ref_poc: int, cells: np.ndarray) -> None:
        cells = np.asarray(cells)
        if cells.shape != self.grid_shape + (3,):
            raise ValueError(f"cells must have shape {self.grid_shape + (3,)}, not {cells.shape}")
        if cells.min() < -32768 or cells.max() > 32767:
            raise ValueError("values out of int16 range")
        if ((cells[..., 2] < 0) | (cells[..., 2] > CONF_MAX)).any():
            raise ValueError(f"confidence must be 0..{CONF_MAX}")
        self.entries[(int(poc), int(ref_poc))] = cells.astype("<i2")


def read_mvf(path) -> MotionField:
    data = Path(path).read_bytes()
    if len(data) < 24 or data[:4] != MVF_MAGIC:
        raise ValueError(f"{path}: not a .mvf file")
    version, width, height, grid, count = struct.unpack_from("<5I", data, 4)
    if version != 1 or grid != GRID:
        raise ValueError(f"{path}: unsupported version {version} / grid {grid}")
    mf = MotionField(width, height)
    gh, gw = mf.grid_shape
    entry_bytes = 8 + gh * gw * 6
    if width == 0 and count == 0:
        return mf
    if len(data) != 24 + count * entry_bytes:
        raise ValueError(f"{path}: size {len(data)} does not match {count} entries of {entry_bytes} bytes "
                         "(file truncated, or the encoder did not finish?)")
    for i in range(count):
        off = 24 + i * entry_bytes
        poc, ref_poc = struct.unpack_from("<2i", data, off)
        cells = np.frombuffer(data, "<i2", gh * gw * 3, off + 8).reshape(gh, gw, 3).copy()
        if (poc, ref_poc) in mf.entries:
            raise ValueError(f"{path}: duplicate entry POC {poc} -> {ref_poc}")
        mf.entries[(poc, ref_poc)] = cells
    return mf


def write_mvf(path, mf: MotionField) -> None:
    with open(path, "wb") as f:
        f.write(MVF_MAGIC + struct.pack("<5I", 1, mf.width, mf.height, GRID, len(mf.entries)))
        for (poc, ref_poc), cells in mf.entries.items():
            f.write(struct.pack("<2i", poc, ref_poc))
            f.write(np.ascontiguousarray(cells, "<i2").tobytes())


def read_dump(path) -> np.ndarray:
    data = Path(path).read_bytes()
    if len(data) < 16 or data[:4] != DUMP_MAGIC:
        raise ValueError(f"{path}: not a .dump file")
    version, rec_size, _ = struct.unpack_from("<3I", data, 4)
    if version != 1 or rec_size != DUMP_DTYPE.itemsize:
        raise ValueError(f"{path}: unsupported version {version} / record size {rec_size}")
    if (len(data) - 16) % rec_size:
        raise ValueError(f"{path}: truncated record at the end")
    return np.frombuffer(data, DUMP_DTYPE, offset=16).copy()


def flipped(mf: MotionField) -> MotionField:
    """The same field with every vector negated (confidence unchanged)."""
    out = MotionField(mf.width, mf.height)
    for key, cells in mf.entries.items():
        c = cells.astype(np.int32)
        c[..., :2] = -c[..., :2]
        out.add(*key, np.clip(c, -32768, 32767))
    return out


def info(path) -> None:
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic == MVF_MAGIC:
        mf = read_mvf(path)
        gh, gw = mf.grid_shape
        print(f"{path}: motion field {mf.width}x{mf.height}, grid {gw}x{gh} cells, {len(mf.entries)} entries")
        print(f"{'POC':>4} {'ref':>4} {'valid %':>8} {'mean |mv| px':>13} {'max |mv| px':>12} {'zero %':>7}")
        for (poc, ref), c in sorted(mf.entries.items()):
            v = c[..., 2] > 0
            mv = c[v][:, :2].astype(float) / 4
            mag = np.hypot(mv[:, 0], mv[:, 1]) if len(mv) else np.zeros(1)
            zero = (mag == 0).mean() * 100 if len(mv) else 0
            print(f"{poc:4d} {ref:4d} {v.mean() * 100:8.1f} {mag.mean():13.2f} {mag.max():12.2f} {zero:7.1f}")
    elif magic == DUMP_MAGIC:
        d = read_dump(path)
        print(f"{path}: {len(d)} motion-search records")
        for poc in np.unique(d["poc"]):
            s = d[d["poc"] == poc]
            refs = ", ".join(f"{r}:{(s['ref_poc'] == r).sum()}" for r in np.unique(s["ref_poc"]))
            print(f"  POC {poc:3d}: {len(s):7d} searches   by reference POC {refs}")
        sub = (d["mv_x"] - 4 * d["int_x"], d["mv_y"] - 4 * d["int_y"])
        print(f"  final - 4*integer, x in [{sub[0].min()}, {sub[0].max()}], y in [{sub[1].min()}, {sub[1].max()}] "
              "(quarter-pel refinement, expected within [-3, 3])")
        print(f"  block sizes: {sorted(set(zip(d['w'].tolist(), d['h'].tolist())))}")
    else:
        raise SystemExit(f"{path}: unknown file type")


def selftest() -> int:
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as td:
        for w, h in [(352, 288), (1920, 1080), (18, 10)]:
            mf = MotionField(w, h)
            gh, gw = mf.grid_shape
            for poc in range(1, 4):
                cells = np.zeros((gh, gw, 3), np.int32)
                cells[..., :2] = rng.integers(-32768, 32768, (gh, gw, 2))
                cells[..., 2] = rng.integers(0, CONF_MAX + 1, (gh, gw))
                mf.add(poc, poc - 1, cells)
            p = Path(td) / "t.mvf"
            write_mvf(p, mf)
            back = read_mvf(p)
            assert (back.width, back.height) == (w, h)
            assert back.entries.keys() == mf.entries.keys()
            for k in mf.entries:
                assert np.array_equal(back.entries[k], mf.entries[k]), k
            ff = flipped(flipped(back))
            for k in mf.entries:   # flipping twice is the identity, except -32768 which clips
                a, b = mf.entries[k].astype(int), ff.entries[k].astype(int)
                ok = (a == b) | (a == -32768)
                assert ok.all(), k
        rec = np.zeros(3, DUMP_DTYPE)
        rec["poc"], rec["mv_x"], rec["int_x"] = [1, 2, 3], [5, -7, 0], [1, -2, 0]
        p = Path(td) / "t.dump"
        p.write_bytes(DUMP_MAGIC + struct.pack("<3I", 1, 32, 0) + rec.tobytes())
        assert np.array_equal(read_dump(p), rec)
    print("mvfield selftest PASSED")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info").add_argument("file")
    p = sub.add_parser("flip")
    p.add_argument("src")
    p.add_argument("dst")
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "info":
        info(a.file)
    elif a.cmd == "flip":
        write_mvf(a.dst, flipped(read_mvf(a.src)))
        print(f"wrote {a.dst}")
    else:
        return selftest()
    return 0


if __name__ == "__main__":
    sys.exit(main())
