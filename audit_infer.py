#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================
AUDIT  DENSE  METRIC-DEPTH  INFERENCE  (standalone)
============================================================

Runs the AUDIT depth pipeline (DPT-Large -> relative depth -> PVCNN/SPVCNN
global scale+shift alignment) and writes a DENSE metric depth map (H, W) in
metres for every image listed in a test_all.txt file.

This is the depth-producing half of PJ-cs/DistanceEstimationTracking
(DemoDistanceEstimationTracking.ipynb). The MegaDetector + DINO + SORT
tracking stages are intentionally omitted: the benchmark only needs the
per-pixel aligned depth map `dpt_aligned = relative_depth * scale + shift`
that the demo computes in step 2 before tracking.

WHY A SEPARATE SCRIPT / ENV
---------------------------
AUDIT depends on torchsparse (old commit), the intel-isl/DPT repo, timm,
DINO, etc. These conflict with the Depth-Anything-3 / UniDepth stack, so
AUDIT is run in its own conda env and invoked as a subprocess by the main
benchmark notebook. Output .npy files are then read back by the notebook.

EXPECTED WORKING DIRECTORY LAYOUT
---------------------------------
Run this script from a directory prepared exactly like the demo's setup
cell (cell 0), i.e. these must be importable / present in CWD:
    dataset.py                (provides get_transforms_dpt)
    models.py             (provides SPVCNN_CLASSIFICATION)
    DPT/                       (intel-isl DPT repo, with run_monodepth.py)
    DPT/weights/dpt_large-midas-2f21e586.pt
    DistanceEstimationTracking/align_weights.pth   (PVCNN weights)

USAGE
-----
    conda run -n audit python audit_infer.py \
        --test_txt   /path/to/test_all.txt \
        --out_dir    /path/to/benchmark_out/preds/audit \
        --dpt_weights DPT/weights/dpt_large-midas-2f21e586.pt \
        --pvcnn_weights DistanceEstimationTracking/align_weights.pth

Each output is written as  <out_dir>/<rgb_stem>.npy  (float32, shape (H, W),
metres, 0 where invalid). A manifest.json records the stem -> path mapping.
============================================================
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shlex
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# ---- AUDIT / DPT imports (must be importable from CWD, see docstring) --------
sys.path.append("DPT")
import DPT.run_monodepth as run_dpt_depth              # noqa: E402
from models import SPVCNN_CLASSIFICATION           # noqa: E402
from dataset import get_transforms_dpt                 # noqa: E402
from torchsparse.utils.helpers import sparse_collate_tensors  # noqa: E402


# ============================================================
# FOCAL LENGTH  —  identical convention to the DA3 training script
# ============================================================
# The benchmark assumes focal length is "known and provided the same way as
# in the training script": a per-source constant selected on the rgb path.
FX_LONGLEAT_V1, FY_LONGLEAT_V1 = 617.16, 617.23
FX_REMAINDER,   FY_REMAINDER   = 2140.58, 2138.63
FOCAL_PX_LONGLEAT_V1 = 0.5 * (FX_LONGLEAT_V1 + FY_LONGLEAT_V1)
FOCAL_PX_REMAINDER   = 0.5 * (FX_REMAINDER   + FY_REMAINDER)


def get_focal_px(rgb_path: str) -> float:
    """Return per-source focal length in pixels (matches train_DA3.py)."""
    return FOCAL_PX_LONGLEAT_V1 if "Longleat_V1" in rgb_path else FOCAL_PX_REMAINDER


# ============================================================
# SAMPLE LIST  (same format/parser as the training script)
# ============================================================
def load_sample_list(txt_path: str):
    """Each line: '/path/to/rgb.png  /path/to/depth.png' -> [(rgb, depth), ...]."""
    samples = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rgb, depth = shlex.split(line)
                samples.append((rgb, depth))
    return samples


# ============================================================
# DPT DISPARITY  ->  RELATIVE DEPTH   (verbatim from the demo)
# ============================================================
def dpt_disparity_to_relative_depth(dpt_img: np.ndarray) -> np.ndarray:
    """
    Convert a raw DPT disparity map to the AUDIT 'relative depth' the PVCNN
    was trained to align. This is the exact transform used in the demo:

        d = (d - d.min()) / d.max()          # normalise disparity to ~[0, 1]
        d = 1.0 / (d * 0.5 + 0.02)           # invert -> relative depth
    """
    d = dpt_img.copy().astype(np.float32)
    d -= d.min()
    d /= d.max()
    d = 1.0 / (d * 0.5 + 0.02)
    return d


def main():
    ap = argparse.ArgumentParser(description="AUDIT dense metric-depth inference")
    ap.add_argument("--test_txt",      required=True)
    ap.add_argument("--out_dir",       required=True)
    ap.add_argument("--dpt_weights",   default="DPT/weights/dpt_large-midas-2f21e586.pt")
    ap.add_argument("--pvcnn_weights", default="DistanceEstimationTracking/align_weights.pth")
    ap.add_argument("--dpt_model_type", default="dpt_large")
    ap.add_argument("--voxel_size",    type=float, default=0.01)
    ap.add_argument("--num_points",    type=int,   default=50000)
    ap.add_argument("--work_dir",      default="temp_audit",
                    help="scratch dir for symlinked frames + DPT .pfm outputs")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[AUDIT] device = {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_sample_list(args.test_txt)
    print(f"[AUDIT] {len(samples)} images from {args.test_txt}")

    # ── Stage a flat frame directory with deterministic names ─────────────────
    # DPT.run() processes a whole folder, and rgb paths come from many source
    # folders with possibly-colliding basenames, so we index them uniquely and
    # keep a manifest mapping index -> (rgb_path, stem, focal, H, W).
    work        = Path(args.work_dir)
    frames_dir  = work / "frames"
    dpt_dir     = work / "dpt"
    for d in (frames_dir, dpt_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for idx, (rgb_path, _depth_path) in enumerate(samples):
        img = cv2.imread(rgb_path)
        if img is None:
            print(f"[AUDIT][WARN] cannot read {rgb_path} — skipping")
            continue
        H, W = img.shape[:2]
        key  = f"{idx:06d}"
        # copy (not symlink) so DPT's imread is robust across filesystems
        cv2.imwrite(str(frames_dir / f"{key}.png"), img)
        manifest[key] = {
            "rgb_path": rgb_path,
            "stem":     Path(rgb_path).stem,
            "focal_px": float(get_focal_px(rgb_path)),
            "H":        int(H),
            "W":        int(W),
        }
    print(f"[AUDIT] staged {len(manifest)} frames -> {frames_dir}")

    # ── Stage 1: DPT-Large relative disparity for the whole folder ────────────
    print(f"[AUDIT] 1/2 running DPT ({args.dpt_model_type}) ...")
    run_dpt_depth.run(str(frames_dir), str(dpt_dir), args.dpt_weights, args.dpt_model_type)

    # ── Stage 2: PVCNN (SPVCNN) global scale+shift alignment -> metric depth ──
    print("[AUDIT] 2/2 aligning to metric depth via SPVCNN ...")
    spvcnn = SPVCNN_CLASSIFICATION(
        input_channel=3, num_classes=2, cr=1.0,
        pres=args.voxel_size, vres=args.voxel_size,
    )
    ckpt = torch.load(args.pvcnn_weights, map_location=device)
    spvcnn.load_state_dict(ckpt["spvcnn_model_state_dict"])
    spvcnn.to(device).eval()

    dpt_transforms = get_transforms_dpt(args.voxel_size, args.num_points)

    written = 0
    with torch.no_grad():
        for pfm_path in tqdm(sorted(glob.glob(str(dpt_dir / "*.pfm"))), desc="AUDIT align"):
            key = Path(pfm_path).stem
            if key not in manifest:
                continue
            meta = manifest[key]
            H, W, focal_px = meta["H"], meta["W"], meta["focal_px"]

            dpt_img = cv2.imread(pfm_path, cv2.IMREAD_UNCHANGED)
            rel = dpt_disparity_to_relative_depth(dpt_img)          # relative depth
            rel_t = torch.from_numpy(rel).unsqueeze(0)              # (1, h, w)

            # resize relative-depth map to the native RGB resolution
            if tuple(rel_t.shape[-2:]) != (H, W):
                rel_t = torch.nn.functional.interpolate(
                    rel_t.unsqueeze(0), size=(H, W),
                    mode="bicubic", align_corners=False,
                ).squeeze(0)

            # relative depth + KNOWN focal -> sparse point cloud for the PVCNN
            dpt_sparse, _dpt_norm = dpt_transforms((rel_t, focal_px))
            sparse_in = sparse_collate_tensors([dpt_sparse]).to(device)

            model_out = spvcnn(sparse_in)          # (1, 2): [scale, shift]
            scale = float(model_out[0, 0].cpu())
            shift = float(model_out[0, 1].cpu())

            metric = rel_t.squeeze(0).squeeze(0).cpu().numpy() * scale + shift
            metric = metric.astype(np.float32)

            np.save(str(out_dir / f"{meta['stem']}.npy"), metric)
            written += 1

    # ── Persist manifest for the benchmark notebook to cross-reference ────────
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[AUDIT] done — wrote {written} dense depth maps to {out_dir}")


if __name__ == "__main__":
    main()
