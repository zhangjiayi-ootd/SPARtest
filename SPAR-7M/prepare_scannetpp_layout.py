#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare SPAR-style layout from an official ScanNet++ installation (RGB only).
- Build image_color/ as 0.jpg, 1.jpg, ...
- READ an existing video_idx.txt and materialize video_color/frame{i}_{idx}.jpg
  by locating source frames in iphone/rgb/frame_{idx:06d}.jpg.

Source (official):
scannetpp/
  scans/<scene_id>/iphone/rgb/frame_000000.jpg

Target (created locally; nothing uploaded):
scannetpp/
  images/<scene_id>/
    image_color/      0.jpg, 1.jpg, 2.jpg, ...
    video_color/      frame0_0.jpg, frame1_35.jpg, ...
    video_idx.txt     (must already exist; read-only)

Examples
--------
# Basic usage (read video_idx.txt from <out-root>/<scene>/video_idx.txt):
python tools/prepare_scannetpp_layout.py \
  --scannetpp-root /path/to/scannetpp \
  --out-root scannetpp/images \
  --use-video-idx

# If your idx files live under a different root (e.g., scannetpp/images/<scene>/video_idx.txt):
python tools/prepare_scannetpp_layout.py \
  --scannetpp-root /path/to/scannetpp \
  --out-root scannetpp/images \
  --use-video-idx \
  --video-idx-root scannetpp/images

# Process only selected scenes (one per line), dry-run first:
python tools/prepare_scannetpp_layout.py \
  --scannetpp-root /path/to/scannetpp \
  --out-root scannetpp/images \
  --scenes-list scenes_scannetpp.txt \
  --use-video-idx \
  --dry-run
"""

import argparse, os, re, shutil, sys
from pathlib import Path
from typing import List, Optional, Tuple

def parse_args():
    p = argparse.ArgumentParser(description="Prepare SPAR-style layout from ScanNet++ (RGB only)")
    p.add_argument("--scannetpp-root", type=Path, required=True,
                   help="Path to official ScanNet++ root (must contain 'scans/<scene_id>/iphone/rgb').")
    p.add_argument("--out-root", type=Path, required=True,
                   help="Output root, e.g., scannetpp/images")
    p.add_argument("--rgb-subdir", type=str, default="iphone/rgb",
                   help="Relative path under each scene to RGB frames (default: iphone/rgb).")
    p.add_argument("--rgb-pattern", type=str, default="frame_*.jpg",
                   help="Glob pattern for RGB frames (default: frame_*.jpg).")
    p.add_argument("--force-ext", type=str, default=".jpg",
                   help="Extension for output files (default: .jpg). Use '' to preserve source suffix.")
    p.add_argument("--copy", action="store_true",
                   help="Copy files instead of symlink (default: symlink).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print actions without writing anything.")
    p.add_argument("--scenes-list", type=Path, default=None,
                   help="Optional text file listing scene IDs to include (one per line).")

    # 读取现有 video_idx.txt 并据此生成 video_color/
    p.add_argument("--use-video-idx", action="store_true",
                   help="Read existing <scene>/video_idx.txt and build video_color/frame{i}_{idx}.jpg")
    p.add_argument("--video-idx-root", type=Path, default=None,
                   help="Root containing <scene>/video_idx.txt (default: use --out-root/<scene>/video_idx.txt)")

    return p.parse_args()

def read_scenes_list(path: Optional[Path]) -> List[str]:
    if not path or not path.exists(): return []
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]

def ensure_dir(d: Path, dry: bool):
    if dry: print(f"[DRY] mkdir -p {d}")
    else: d.mkdir(parents=True, exist_ok=True)

def remove_if_exists(p: Path, dry: bool):
    if not p.exists() and not p.is_symlink(): return
    if dry: print(f"[DRY] rm -rf {p}"); return
    if p.is_symlink() or p.is_file(): p.unlink()
    else: shutil.rmtree(p)

def link_or_copy(src: Path, dst: Path, do_copy: bool, dry: bool):
    if not src.exists():
        print(f"[WARN] Missing source: {src}"); return
    if dst.exists() or dst.is_symlink():
        remove_if_exists(dst, dry)
    ensure_dir(dst.parent, dry)
    if dry:
        print(f"[DRY] {'copy' if do_copy else 'symlink'} {src} -> {dst}")
        return
    if do_copy: shutil.copy2(src, dst)
    else: dst.symlink_to(src.resolve())

_num_re = re.compile(r"(\d+)")

def extract_frame_index(name: str) -> Optional[int]:
    m = _num_re.search(name)
    if not m: return None
    try: return int(m.group(1))
    except ValueError: return None

def sorted_rgb_frames(rgb_dir: Path, pattern: str) -> List[Tuple[Path, int]]:
    files = list(rgb_dir.glob(pattern))
    items = []
    for f in files:
        idx = extract_frame_index(f.name) or extract_frame_index(f.stem)
        if idx is None: continue
        items.append((f, idx))
    items.sort(key=lambda x: x[1])  # sort by original index
    return items

def find_rgb_by_index(rgb_dir: Path, idx: int) -> Optional[Path]:
    # 尝试常见扩展名
    for ext in (".jpg", ".jpeg", ".png"):
        cand = rgb_dir / f"frame_{idx:06d}{ext}"
        if cand.exists(): return cand
    # 兜底：遍历一遍（慢，但稳）
    for f in rgb_dir.iterdir():
        if not f.is_file(): continue
        i = extract_frame_index(f.name) or extract_frame_index(f.stem)
        if i == idx: return f
    return None

def main():
    args = parse_args()

    scans_dir = args.scannetpp_root / "scans"
    if not scans_dir.exists():
        print(f"[ERR] Expecting {scans_dir}")
        sys.exit(1)

    allow = set(read_scenes_list(args.scenes_list))
    if allow: print(f"[INFO] Filtering {len(allow)} scenes")

    n_scenes = 0
    for scene_dir in sorted((p for p in scans_dir.iterdir() if p.is_dir()),
                            key=lambda p: p.name):
        scene = scene_dir.name  # e.g., "0a5c013435"
        if allow and scene not in allow:
            continue

        rgb_dir = scene_dir / args.rgb_subdir
        if not rgb_dir.exists():
            print(f"[WARN] Skip {scene}: RGB dir not found: {rgb_dir}")
            continue

        frames = sorted_rgb_frames(rgb_dir, args.rgb_pattern)
        if not frames:
            print(f"[WARN] Skip {scene}: no frames matching {args.rgb_pattern} in {rgb_dir}")
            continue

        print(f"[SCENE] {scene}  (frames: {len(frames)})")

        # 1) image_color/: 0.jpg, 1.jpg, ...
        out_scene = args.out_root / scene / "image_color"
        ensure_dir(out_scene, args.dry_run)
        for i, (src, orig_idx) in enumerate(frames):
            dst = out_scene / (f"{i}{args.force_ext}" if args.force_ext else f"{i}{src.suffix.lower()}")
            link_or_copy(src, dst, args.copy, args.dry_run)

        # 2) video_color/: read video_idx.txt → frame{i}_{idx}.jpg
        if args.use_video_idx:
            idx_file = (args.out_root / scene / "video_idx.txt")
            if args.video_idx_root is not None:
                cand = args.video_idx_root / scene / "video_idx.txt"
                if cand.exists(): idx_file = cand

            if not idx_file.exists():
                print(f"[WARN] {scene}: video_idx.txt not found at {idx_file}; skip video_color")
            else:
                vcolor_dir = args.out_root / scene / "video_color"
                ensure_dir(vcolor_dir, args.dry_run)
                lines = [ln.strip() for ln in idx_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
                kept = 0
                for i, ln in enumerate(lines):
                    try:
                        idx = int(ln)
                    except ValueError:
                        print(f"[WARN] {scene}: bad index '{ln}' in {idx_file}"); continue
                    src = find_rgb_by_index(rgb_dir, idx)
                    if src is None:
                        print(f"[WARN] {scene}: source not found for idx={idx}")
                        continue
                    # frame{i}_{idx}.<ext>
                    ext = (args.force_ext if args.force_ext else src.suffix.lower()) or ".jpg"
                    dst = vcolor_dir / f"frame{i}_{idx}{ext}"
                    link_or_copy(src, dst, args.copy, args.dry_run)
                    kept += 1
                print(f"[INFO] {scene}: video_color built: {kept}/{len(lines)}")

        n_scenes += 1

    print(f"[DONE] Scenes processed: {n_scenes}")
    print(f"[INFO] Output root: {args.out_root.resolve()}")

if __name__ == "__main__":
    main()
