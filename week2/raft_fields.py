#!/usr/bin/env python3
"""Week 2 - optical flow from torchvision's RAFT, written as HM-mv motion fields (.mvf).

For every sequence of a manifest (week1/sequences*.json) and every frame t >= 1, RAFT computes
the flow from frame t to frame t - 1 on the original frames (decision D1: only the previous
frame). The flow is averaged over each 4x4 block, converted to quarter-pel and rounded, and
written as the entry (POC t, reference POC t - 1) of <out-dir>/<name>.mvf, with confidence
1000 in every cell. A sidecar <name>.json records how the field was made.

Sign: RAFT(image1, image2) gives the flow with image1(x) ~ image2(x + flow(x)). With
image1 = frame t and image2 = frame t - 1 this is exactly HM's convention, ref(x + mv) ~ cur(x),
so mv = 4 * flow (quarter-pel), with no change of sign.

After each sequence the script prints a motion-compensation check on the original frames
(mean absolute luma error per pixel when predicting frame t from frame t - 1 with the field,
with zero motion and with the sign-flipped field). With pretrained weights, "field" must be
clearly below "zero" and far below "flipped"; if not, stop and check before encoding.

RAFT needs width and height divisible by 8: frames are padded (edge) and the flow is cropped.
1080p at full resolution needs about 10-12 GB of GPU memory; if it runs out, use --scale 0.5
(flow computed at 960x540 and scaled back up).

Examples (from week2/, venv active):
  python raft_fields.py --manifest ../week1/sequences.json --groups small --out-dir mv/raft
  python raft_fields.py --manifest ../week1/sequences_60fps.json --out-dir mv/raft_60fps
  python raft_fields.py --selftest
  # Claude's workspace (no pretrained weights there): untrained network, CPU, 3 frames
  python raft_fields.py --manifest m.json --weights none --device cpu --frames 3 --out-dir /tmp/raft
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mvfield import CONF_MAX, GRID, MotionField, write_mvf  # noqa: E402
from field_stats import mc_mae  # noqa: E402

# Limited-range YCbCr -> RGB (luma 16..235, chroma 16..240)
MATRICES = {"bt601": (0.299, 0.114), "bt709": (0.2126, 0.0722)}


# ---------------------------------------------------------------- frames

def read_frame(path, width: int, height: int, n: int) -> tuple:
    """Frame n of an 8-bit 4:2:0 .yuv file: (Y (H, W), Cb (H/2, W/2), Cr (H/2, W/2)), uint8."""
    cw, ch = (width + 1) // 2, (height + 1) // 2
    fs = width * height + 2 * cw * ch
    raw = np.fromfile(path, np.uint8, fs, offset=n * fs)
    if raw.size != fs:
        raise ValueError(f"{path}: frame {n} is missing (file too short)")
    y = raw[:width * height].reshape(height, width)
    cb = raw[width * height:width * height + cw * ch].reshape(ch, cw)
    cr = raw[width * height + cw * ch:].reshape(ch, cw)
    return y, cb, cr


def yuv_to_rgb(y, cb, cr, matrix: str) -> np.ndarray:
    """8-bit limited-range 4:2:0 -> float32 RGB in [0, 1], shape (3, H, W). Chroma upsampled by repetition."""
    kr, kb = MATRICES[matrix]
    kg = 1.0 - kr - kb
    h, w = y.shape
    up = lambda c: np.repeat(np.repeat(c, 2, 0), 2, 1)[:h, :w].astype(np.float32)  # noqa: E731
    yf = (y.astype(np.float32) - 16.0) / 219.0
    pb = (up(cb) - 128.0) / 224.0
    pr = (up(cr) - 128.0) / 224.0
    r = yf + 2 * (1 - kr) * pr
    b = yf + 2 * (1 - kb) * pb
    g = (yf - kr * r - kb * b) / kg
    return np.clip(np.stack([r, g, b]), 0.0, 1.0)


def default_matrix(width: int, height: int) -> str:
    """BT.709 for HD (UVG), BT.601 for SD (Xiph CIF)."""
    return "bt709" if width >= 1280 or height >= 720 else "bt601"


# ---------------------------------------------------------------- flow -> field

def flow_to_cells(flow: np.ndarray) -> np.ndarray:
    """Flow (2, H, W) in pixels -> mvfield cells (ceil(H/4), ceil(W/4), 3) int16:
    per 4x4 block the mean flow, times 4, rounded to the nearest quarter-pel; confidence CONF_MAX.
    Blocks cut by the picture border average only the pixels inside the picture."""
    _, h, w = flow.shape
    gh, gw = -(-h // GRID), -(-w // GRID)
    pad = np.zeros((2, gh * GRID, gw * GRID), np.float64)
    cnt = np.zeros((gh * GRID, gw * GRID), np.float64)
    pad[:, :h, :w] = flow
    cnt[:h, :w] = 1.0
    s = pad.reshape(2, gh, GRID, gw, GRID).sum(axis=(2, 4))
    n = cnt.reshape(gh, GRID, gw, GRID).sum(axis=(1, 3))
    mean = s / n
    cells = np.empty((gh, gw, 3), np.int32)
    cells[..., 0] = np.clip(np.rint(4.0 * mean[0]), -32768, 32767)
    cells[..., 1] = np.clip(np.rint(4.0 * mean[1]), -32768, 32767)
    cells[..., 2] = CONF_MAX
    return cells


class RaftFlow:
    """torchvision RAFT large: flow(img1, img2) with img* float32 RGB (3, H, W) in [0, 1] ->
    flow (2, H, W) in pixels (channel 0 = x, 1 = y), image1(x) ~ image2(x + flow)."""

    def __init__(self, weights: str, device: str, updates: int, scale: float):
        import torch
        import torchvision
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
        self.torch = torch
        self.F = torch.nn.functional
        if weights == "none":
            self.weights_name = "none (untrained, for tests only)"
            w = None
        else:
            w = Raft_Large_Weights.DEFAULT if weights == "default" else Raft_Large_Weights[weights]
            self.weights_name = f"Raft_Large_Weights.{w.name}"
        self.device = torch.device(device)
        self.model = raft_large(weights=w, progress=True).to(self.device).eval()
        self.updates = updates
        self.scale = scale
        self.versions = {"torch": torch.__version__, "torchvision": torchvision.__version__}

    def __call__(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        torch, F = self.torch, self.F
        _, h, w = img1.shape
        x = torch.from_numpy(np.stack([img1, img2])).to(self.device)        # (2, 3, H, W)
        if self.scale != 1.0:
            sh, sw = max(8, round(h * self.scale)), max(8, round(w * self.scale))
            x = F.interpolate(x, size=(sh, sw), mode="bilinear", align_corners=False, antialias=True)
        else:
            sh, sw = h, w
        ph, pw = (-sh) % 8, (-sw) % 8
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        x = x * 2.0 - 1.0                                                     # RAFT expects [-1, 1]
        try:
            with torch.inference_mode():
                flow = self.model(x[0:1], x[1:2], num_flow_updates=self.updates)[-1]   # (1, 2, H', W')
        finally:
            # torchvision keeps the correlation pyramid as an attribute of the model after the call
            # (5.6 GB at 1080p). Left there, it is still allocated while the next frame pair builds
            # its own volume (another 8.4 GB at the peak), and 1080p runs out of memory on 16 GB.
            self.model.corr_block.corr_pyramid = []
        del x
        flow = flow[:, :, :sh, :sw]
        if (sh, sw) != (h, w):
            flow = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False)
            flow[:, 0] *= w / sw
            flow[:, 1] *= h / sh
        return flow[0].float().cpu().numpy()


# ---------------------------------------------------------------- per sequence

def file_md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def process(seq: dict, frames: int, flow_fn, out_dir: Path, matrix: str | None, meta_common: dict) -> dict:
    w, h, path = seq["width"], seq["height"], seq["path"]
    if seq.get("bit_depth", 8) != 8:
        raise SystemExit(f"{seq['name']}: only 8-bit input is supported")
    mat = matrix or default_matrix(w, h)
    mf = MotionField(w, h)
    mae = {"field": 0.0, "zero": 0.0, "flipped": 0.0}
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass
    t0 = time.time()
    prev = read_frame(path, w, h, 0)
    prev_rgb = yuv_to_rgb(*prev, mat)
    for t in range(1, frames):
        cur = read_frame(path, w, h, t)
        cur_rgb = yuv_to_rgb(*cur, mat)
        flow = flow_fn(cur_rgb, prev_rgb)                 # frame t -> frame t - 1
        cells = flow_to_cells(flow)
        mf.add(t, t - 1, cells)
        flip = cells.copy()
        flip[..., :2] = -flip[..., :2]
        zero = cells.copy()
        zero[..., :2] = 0
        for k, c in (("field", cells), ("zero", zero), ("flipped", flip)):
            mae[k] += mc_mae(cur[0], prev[0], c)[0]
        prev, prev_rgb = cur, cur_rgb
        print(f"\r  {seq['name']}: {t}/{frames - 1} frame pairs", end="", flush=True)
    secs = time.time() - t0
    out = out_dir / f"{seq['name']}.mvf"
    write_mvf(out, mf)
    try:
        import torch
        peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
    except ImportError:
        peak = None
    mags = np.concatenate([np.hypot(*(c[..., :2].reshape(-1, 2).T / 4.0)) for c in mf.entries.values()])
    n = max(frames - 1, 1)
    meta = dict(meta_common, sequence=seq["name"], input=path, width=w, height=h, frames=frames,
                entries=len(mf.entries), colour_matrix=mat, seconds=round(secs, 1),
                seconds_per_pair=round(secs / n, 3), mean_abs_mv_px=round(float(mags.mean()), 3),
                max_abs_mv_px=round(float(mags.max()), 2),
                mc_mae_orig={k: round(v / n, 3) for k, v in mae.items()},
                gpu_peak_gib=round(peak, 2) if peak is not None else None,
                mvf=out.name, mvf_md5=file_md5(out))
    (out_dir / f"{seq['name']}.json").write_text(json.dumps(meta, indent=2) + "\n")
    m = meta["mc_mae_orig"]
    ok = m["field"] < m["zero"] and m["field"] < m["flipped"]
    gpu = f", GPU peak {peak:.1f} GiB" if peak is not None else ""
    print(f"\r  {seq['name']}: {len(mf.entries)} entries, {secs / n:.2f} s per pair{gpu}, mean |mv| "
          f"{meta['mean_abs_mv_px']:.2f} px; MC error field {m['field']:.2f} / zero {m['zero']:.2f} / "
          f"flipped {m['flipped']:.2f}{'' if ok else '   <-- field not better than zero/flipped!'}")
    return meta


# ---------------------------------------------------------------- self-test

def selftest() -> int:
    from mvfield import read_mvf
    import tempfile
    rng = np.random.default_rng(3)
    # 1. units and rounding: constant flow -> constant quarter-pel cells
    f = np.zeros((2, 10, 18), np.float32)
    f[0], f[1] = 1.25, -0.6
    c = flow_to_cells(f)
    assert c.shape == (3, 5, 3), c.shape
    assert (c[..., 0] == 5).all() and (c[..., 1] == -2).all() and (c[..., 2] == CONF_MAX).all()
    # 2. per-block mean, also for border blocks cut by the picture edge
    f = rng.normal(0, 3, (2, 10, 18)).astype(np.float32)
    c = flow_to_cells(f)
    for gy in range(3):
        for gx in range(5):
            blk = f[:, gy * 4:gy * 4 + 4, gx * 4:gx * 4 + 4]
            want = np.rint(4 * blk.reshape(2, -1).mean(1))
            assert (c[gy, gx, :2] == want).all(), (gy, gx)
    # 3. colour conversion: limited-range black, white, grey
    for v, rgb in ((16, 0.0), (235, 1.0), (126, (126 - 16) / 219)):
        y = np.full((4, 4), v, np.uint8)
        cc = np.full((2, 2), 128, np.uint8)
        for m in MATRICES:
            assert np.allclose(yuv_to_rgb(y, cc, cc, m), rgb, atol=1e-6), (v, m)
    # 4. sign convention end to end: frame t is frame t-1 moved by (+3, -2) pixels, i.e.
    #    cur(x) = ref(x - d). A flow that follows RAFT's convention, cur(x) ~ ref(x + flow),
    #    is flow = -d; the field built from it must predict cur from ref exactly (inner blocks).
    ref = rng.integers(0, 256, (64, 96)).astype(np.uint8)
    d = (3, -2)
    cur = np.roll(ref, shift=(d[1], d[0]), axis=(0, 1))
    flow = np.empty((2, 64, 96), np.float32)
    flow[0], flow[1] = -d[0], -d[1]
    cells = flow_to_cells(flow)
    assert (cells[..., 0] == -12).all() and (cells[..., 1] == 8).all()
    inner = np.zeros(cells.shape[:2], bool)
    inner[2:-2, 2:-2] = True
    cin = cells.copy()
    cin[~inner, 2] = 0
    assert mc_mae(cur, ref, cin)[0] == 0.0
    flip = cin.copy()
    flip[..., :2] *= -1
    assert mc_mae(cur, ref, flip)[0] > 50
    # 5. padding / scaling path of RaftFlow with a stand-in model (no torch weights needed)
    try:
        import torch
    except ImportError:
        print("  (torch not installed: RaftFlow padding test skipped)")
    else:
        class Stub:          # returns a constant flow (2.0, -1.0) at whatever size it gets
            def __init__(self):
                self.corr_block = type("CorrBlock", (), {})()

            def __call__(self, a, b, num_flow_updates):
                self.corr_block.corr_pyramid = [torch.zeros(3)]   # like torchvision: left on the model
                assert a.shape[-1] % 8 == 0 and a.shape[-2] % 8 == 0, a.shape
                fl = torch.zeros(1, 2, *a.shape[-2:])
                fl[:, 0], fl[:, 1] = 2.0, -1.0
                return [fl]
        rf = RaftFlow.__new__(RaftFlow)
        rf.torch, rf.F, rf.device, rf.model, rf.updates = torch, torch.nn.functional, torch.device("cpu"), Stub(), 1
        for (hh, ww), s, want in (((36, 44), 1.0, (2.0, -1.0)), ((1080 // 4, 1920 // 4), 0.5, (4.0, -2.0))):
            rf.scale = s
            img = rng.random((3, hh, ww)).astype(np.float32)
            out = rf(img, img)
            assert out.shape == (2, hh, ww), out.shape
            assert np.allclose(out[0], want[0], atol=1e-4) and np.allclose(out[1], want[1], atol=1e-4), (s, out.mean((1, 2)))
            assert rf.model.corr_block.corr_pyramid == [], "correlation pyramid not released"
    # 6. .mvf round trip of a field built here
    with tempfile.TemporaryDirectory() as td:
        mf = MotionField(96, 64)
        mf.add(1, 0, cells)
        write_mvf(Path(td) / "t.mvf", mf)
        assert np.array_equal(read_mvf(Path(td) / "t.mvf").entries[(1, 0)], cells.astype("<i2"))
    print("raft_fields selftest PASSED")
    return 0


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", help="week1/sequences*.json")
    ap.add_argument("--groups", nargs="*", help="only these manifest groups (small, uvg)")
    ap.add_argument("--names", nargs="*", help="only these sequence names")
    ap.add_argument("--frames", type=int, help="fewer frames than in the manifest")
    ap.add_argument("--out-dir", default="mv/raft")
    ap.add_argument("--weights", default="default",
                    help="default (C_T_SKHT_V2), another Raft_Large_Weights name (e.g. C_T_V2), or none (untrained)")
    ap.add_argument("--updates", type=int, default=12, help="RAFT refinement iterations (torchvision default 12)")
    ap.add_argument("--scale", type=float, default=1.0, help="compute the flow at this fraction of the size (0.5 if 1080p runs out of memory)")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--matrix", choices=list(MATRICES), help="YCbCr matrix (default: BT.709 for HD, BT.601 otherwise)")
    ap.add_argument("--force", action="store_true", help="overwrite existing fields")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.manifest:
        ap.error("--manifest is required")

    seqs = json.loads(Path(a.manifest).read_text())
    if a.groups:
        seqs = [s for s in seqs if s.get("group") in a.groups]
    if a.names:
        seqs = [s for s in seqs if s["name"] in a.names]
    if not seqs:
        raise SystemExit("no sequences selected")
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu" and a.weights != "none":
        print("note: running RAFT on the CPU (slow)")
    flow_fn = RaftFlow(a.weights, device, a.updates, a.scale)
    meta_common = {"weights": flow_fn.weights_name, "num_flow_updates": a.updates, "scale": a.scale,
                   "device": torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu",
                   **flow_fn.versions, "manifest": str(Path(a.manifest).resolve()),
                   "made": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(f"RAFT large, weights {flow_fn.weights_name}, {a.updates} updates, scale {a.scale}, on {meta_common['device']}")
    for s in seqs:
        frames = min(s["frames"], a.frames or s["frames"])
        out = out_dir / f"{s['name']}.mvf"
        side = out_dir / f"{s['name']}.json"
        if out.exists() and side.exists() and not a.force:
            old = json.loads(side.read_text())
            same = all(old.get(k) == meta_common[k] for k in ("weights", "num_flow_updates", "scale")) \
                and old.get("frames") == frames and old.get("input") == s["path"]
            if not same:
                raise SystemExit(f"{out} exists but was made with other settings ({side}); use --force or another --out-dir")
            print(f"  {s['name']}: exists, skipped")
            continue
        try:
            process(s, frames, flow_fn, out_dir, a.matrix, meta_common)
        except torch.cuda.OutOfMemoryError:
            free, total = torch.cuda.mem_get_info()
            raise SystemExit(f"\n{s['name']}: out of GPU memory ({free / 2**30:.1f} of {total / 2**30:.1f} GiB free now; "
                             "1080p needs about 8 GiB). Check nvidia-smi for other processes using the GPU; "
                             "if it still fails, re-run with --scale 0.5 and a separate --out-dir")
    return 0


if __name__ == "__main__":
    sys.exit(main())
