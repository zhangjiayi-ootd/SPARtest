#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare SPAR-style layout from an official Structured3D installation (RGB only).

Source (official-like):
Structured3D/
  scene_02199/
    2D_rendering/
      3972/
        perspective/
          full/
            0/
              rgb_rawlight.png
            1/
              rgb_rawlight.png
            ...

Target (created locally; nothing uploaded):
structured3d/
  images/<scene_id>/
    image_color/
      <view_id>_<cam_id>.jpg     # e.g., 3972_0.jpg

Notes
-----
- Only image_color is generated (no video dirs).
- By default, we CONVERT source PNG → JPG with a configurable quality.
- If you want to keep source suffix (e.g., .png), pass --force-ext ".png".

Usage (examples)
----------------
# Convert all scenes, default: rgb_rawlight.png → JPG quality=95
python tools/prepare_structured3d_layout.py \
  --structured3d-root /path/to/Structured3D \
  --out-root structured3d/images

# Only selected scenes, keep PNG suffix, dry-run first:
python tools/prepare_structured3d_layout.py \
  --structured3d-root /path/to/Structured3D \
  --out-root structured3d/images \
  --scenes-list scenes_structured3d.txt \
  --force-ext ".png" \
  --dry-run

# If your file name is different (e.g., rgb_nolight.png), or try fallbacks:
python tools/prepare_structured3d_layout.py \
  --structured3d-root /path/to/Structured3D \
  --out-root structured3d/images \
  --rgb-name rgb_nolight.png \
  --fallback-rgb rgb_rawlight.png rgb_shading.png
"""

import argparse
import os
import re
import sys
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from PIL import Image
except Exception:
    Image = None  # 若未安装 Pillow，且要求转 JPG，会提示

def parse_args():
    p = argparse.ArgumentParser(description="Prepare SPAR-style layout from Structured3D (RGB only)")
    p.add_argument("--structured3d-root", type=Path, required=True,
                   help="Path to Structured3D root (contains scene_XXXX/2D_rendering/...).")
    p.add_argument("--out-root", type=Path, required=True,
                   help="Output root, e.g., structured3d/images")
    p.add_argument("--scenes-list", type=Path, default=None,
                   help="Optional file listing scene IDs to include (one per line, e.g., scene_02199).")

    # File picking
    p.add_argument("--rgb-name", type=str, default="rgb_rawlight.png",
                   help="Preferred RGB filename under .../perspective/full/<cam>/ (default: rgb_rawlight.png)")
    p.add_argument("--fallback-rgb", type=str, nargs="*", default=["rgb_nolight.png", "rgb_shading.png"],
                   help="Fallback filenames to try if --rgb-name not found.")

    # Output naming & format
    p.add_argument("--force-ext", type=str, default=".jpg",
                   help="Output extension (default: .jpg). Use '.png' to keep PNG, or '' to preserve source suffix.")
    p.add_argument("--jpg-quality", type=int, default=95,
                   help="JPEG quality if converting to JPEG (default: 95).")

    # Safety & speed
    p.add_argument("--dry-run", action="store_true",
                   help="Print actions without writing anything.")
    p.add_argument("--copy", action="store_true",
                   help="Copy bytes instead of converting when suffix matches or --force-ext='' and src suffix equals dst suffix.")

    return p.parse_args()

def read_scenes_list(path: Optional[Path]) -> List[str]:
    if not path or not path.exists():
        return []
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]

def ensure_dir(d: Path, dry: bool):
    if dry:
        print(f"[DRY] mkdir -p {d}")
    else:
        d.mkdir(parents=True, exist_ok=True)

def save_image_convert(src: Path, dst: Path, force_ext: str, jpg_quality: int, dry: bool, copy_bytes: bool):
    """
    Save/convert image:
      - If force_ext is '', preserve source suffix.
      - If force_ext is '.jpg' or '.png', force the target suffix/format.
      - If copy_bytes is True and suffixes match, do a byte copy.
      - Otherwise, convert with Pillow (if available).
    """
    # Decide dst suffix
    if force_ext:
        dst = dst.with_suffix(force_ext)
    else:
        dst = dst.with_suffix(src.suffix.lower())

    if dry:
        print(f"[DRY] write {dst}  (from {src})")
        return

    if copy_bytes and dst.suffix.lower() == src.suffix.lower():
        # Direct byte copy
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return

    # Convert via Pillow if needed
    if Image is None:
        raise RuntimeError("Pillow is not installed, cannot convert images. Install 'Pillow' or use --copy with matching suffix.")

    img = Image.open(src)
    # Ensure RGB for JPEG
    if dst.suffix.lower() in [".jpg", ".jpeg"] and img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.suffix.lower() in [".jpg", ".jpeg"]:
        img.save(dst, format="JPEG", quality=jpg_quality, optimize=True)
    elif dst.suffix.lower() == ".png":
        img.save(dst, format="PNG")
    else:
        # Other formats if Pillow supports
        img.save(dst)

def find_rgb_file(cam_dir: Path, prefer: str, fallbacks: List[str]) -> Optional[Path]:
    """Pick rgb file under a camera directory by preferred filename then fallbacks."""
    cand = cam_dir / prefer
    if cand.exists():
        return cand
    for fb in fallbacks:
        c = cam_dir / fb
        if c.exists():
            return c
    # As a last resort: any png with 'rgb' in name
    for f in cam_dir.glob("*.png"):
        if "rgb" in f.name:
            return f
    return None

def list_scenes(structured3d_root: Path) -> List[Path]:
    # scene directories typically named 'scene_XXXXX'
    return sorted([p for p in structured3d_root.iterdir() if p.is_dir() and p.name.startswith("scene_")],
                  key=lambda p: p.name)

def process_scene(scene_dir: Path, out_root: Path,
                  prefer_name: str, fallback_names: List[str],
                  force_ext: str, jpg_quality: int, dry: bool, copy_bytes: bool) -> int:
    """
    Returns number of images written for this scene.
    """
    scene_id = scene_dir.name
    twoD = scene_dir / "2D_rendering"
    if not twoD.exists():
        print(f"[WARN] Skip {scene_id}: 2D_rendering not found")
        return 0

    # Layout: 2D_rendering/<view_id>/perspective/full/<cam_id>/<rgb_file>
    count = 0
    for view_dir in sorted([p for p in twoD.iterdir() if p.is_dir() and p.name.isdigit()],
                           key=lambda p: int(p.name)):
        view_id = view_dir.name
        persp = view_dir / "perspective" / "full"
        if not persp.exists():
            # Some scenes may not have 'perspective/full'
            # Try a broader search if needed (not typical)
            print(f"[WARN] {scene_id}: perspective/full not found in view {view_id}")
            continue

        # iterate cameras under 'full'
        cam_dirs = sorted([p for p in persp.iterdir() if p.is_dir()],
                          key=lambda p: p.name)
        for cam_dir in cam_dirs:
            cam_id = cam_dir.name  # e.g., "0", "1"
            src = find_rgb_file(cam_dir, prefer_name, fallback_names)
            if src is None:
                print(f"[WARN] {scene_id}: missing RGB under {cam_dir}")
                continue

            out_scene = out_root / scene_id / "image_color"
            ensure_dir(out_scene, dry)
            dst_name = f"{view_id}_{cam_id}"
            dst = out_scene / dst_name  # suffix assigned in save_image_convert
            try:
                save_image_convert(src, dst, force_ext, jpg_quality, dry, copy_bytes)
                count += 1
            except Exception as e:
                print(f"[WARN] {scene_id}: convert failed for {src} → {dst}: {e}")

    if count == 0:
        print(f"[WARN] {scene_id}: no images written")
    else:
        print(f"[INFO] {scene_id}: images written: {count}")
    return count

def main():
    args = parse_args()

    root = args.structured3d_root
    if not root.exists():
        print(f"[ERR] Structured3D root not found: {root}")
        sys.exit(1)

    allow = set(read_scenes_list(args.scenes_list))
    if allow:
        print(f"[INFO] Filtering {len(allow)} scenes")

    total = 0
    for scene_dir in list_scenes(root):
        if allow and scene_dir.name not in allow:
            continue
        total += process_scene(scene_dir, args.out_root,
                               args.rgb_name, args.fallback_rgb,
                               args.force_ext, args.jpg_quality,
                               args.dry_run, args.copy)

    print(f"[DONE] Scenes processed: {total}")
    print(f"[INFO] Output root: {args.out_root.resolve()}")

if __name__ == "__main__":
    main()
