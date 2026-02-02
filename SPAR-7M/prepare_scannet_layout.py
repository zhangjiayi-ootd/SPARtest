#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare SPAR-style layout from an official ScanNet installation (no original files uploaded).

Target layout (created locally):
spar/
  scannet/
    images/
      scene0000_00/
        image_color/      -> link/copy from scans/scene0000_00/color
        image_depth/      -> link/copy from scans/scene0000_00/depth          (if --rgbd)
        intrinsic/        -> link/copy from scans/scene0000_00/intrinsic       (if --rgbd)
        pose/             -> link/copy from scans/scene0000_00/pose            (if --rgbd)
        video_color/      -> built from video_idx.txt as frame{i}_{idx}.jpg    (if --use-video-idx)
        video_depth/      -> reserved directory                                 (if --rgbd)
        video_pose/       -> reserved directory                                 (if --rgbd)
        video_idx.txt     -> from --video-idx-root/<scene>/video_idx.txt or empty placeholder

Notes
-----
- When --use-video-idx is enabled, we read each integer index from `video_idx.txt` and
  pick the corresponding source image from `image_color/`, then materialize:
      video_color/frame{i}_{idx}.jpg
  Example: indices [0, 35, 70, 106] → frame0_0.jpg, frame1_35.jpg, ...
- Source matching is robust to common ScanNet filename styles:
  `174.jpg`, `000174.jpg`, `frame-000174.color.jpg`, etc.
- By default we create **symlinks**; add `--copy` to duplicate files instead.

USAGE (examples)
----------------
# 1) Minimal (RGB only), all scenes, symlinks:
python tools/prepare_scannet_layout.py \
  --scannet-root /path/to/scannet \
  --out-root spar/scannet/images

# 2) RGBD (also link/copy depth/intrinsic/pose):
python tools/prepare_scannet_layout.py \
  --scannet-root /path/to/scannet \
  --out-root spar/scannet/images \
  --rgbd

# 3) Use an existing video_idx.txt to populate video_color/
#    (your idx files live in spar-jsonl/scannet/images/<scene>/video_idx.txt)
python tools/prepare_scannet_layout.py \
  --scannet-root /path/to/scannet \
  --out-root spar/scannet/images \
  --use-video-idx \
  --video-idx-root spar-jsonl/scannet/images

# 4) Only selected scenes (one per line in scenes.txt), copy instead of symlink, dry-run first:
python tools/prepare_scannet_layout.py \
  --scannet-root /path/to/scannet \
  --out-root spar/scannet/images \
  --scenes-list scenes.txt \
  --copy \
  --dry-run

# 5) Customize filename patterns if your ScanNet color/depth names differ:
python tools/prepare_scannet_layout.py \
  --scannet-root /path/to/scannet \
  --out-root spar/scannet/images \
  --rgbd \
  --use-video-idx --video-idx-root spar-jsonl/scannet/images \
  --color-pattern "*.jpg" \
  --depth-pattern "*.png"

Arguments
---------
--scannet-root PATH        Path to official ScanNet root (contains scans/sceneXXXX_XX/...).
--out-root PATH            Output root for SPAR-style layout (e.g., spar/scannet/images).
--rgbd                     Also prepare depth/intrinsic/pose (+ create video_depth/video_pose dirs).
--use-video-idx            Build video_color/ as frame{i}_{idx}.jpg using video_idx.txt.
--video-idx-root PATH      Root containing <scene>/video_idx.txt (default: use <out-root>/<scene>/video_idx.txt).
--scenes-list FILE         Optional file listing scene IDs to include (one per line).
--copy                     Copy files instead of creating symlinks (default is symlink).
--dry-run                  Print planned actions without writing anything.
--color-pattern STR        Glob for color images inside color/ (default: *.jpg).
--depth-pattern STR        Glob for depth images inside depth/ (default: *.png).

Caveats
-------
- This script **does not** upload any ScanNet content; it only organizes your **local** installation.
- If you previously committed original images to your repo, remove them and clean history (e.g., git lfs/filter-repo).
- `video_depth/` and `video_pose/` are created as placeholders; extend the script similarly if you later want to
  materialize depth-based frames with the same index list.
"""
import argparse, os, shutil, sys, re, subprocess
from pathlib import Path
from typing import List, Optional

def parse_args():
    p = argparse.ArgumentParser(description="Prepare SPAR-style layout from ScanNet")
    p.add_argument("--scannet-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--rgbd", action="store_true")
    p.add_argument("--copy", action="store_true", help="copy instead of symlink")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--scenes-list", type=Path, default=None)
    p.add_argument("--color-pattern", default="*.jpg")
    p.add_argument("--depth-pattern", default="*.png")
    p.add_argument("--use-video-idx", action="store_true",
                   help="Read video_idx.txt and materialize video_color as frame{i}_{idx}.jpg")
    p.add_argument("--video-idx-root", type=Path, default=None,
                   help="Root that contains <scene>/video_idx.txt (e.g. spar-jsonl/scannet/images). "
                        "Default: use --out-root/<scene>/video_idx.txt")
    return p.parse_args()

def read_scenes_list(path: Optional[Path]) -> List[str]:
    if not path or not path.exists(): return []
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]

def ensure_dir(d: Path, dry=False):
    if dry: print(f"[DRY] mkdir -p {d}"); return
    d.mkdir(parents=True, exist_ok=True)

def remove_if_exists(p: Path, dry=False):
    if not p.exists() and not p.is_symlink(): return
    if dry: print(f"[DRY] rm -rf {p}"); return
    if p.is_symlink() or p.is_file(): p.unlink()
    else: shutil.rmtree(p)

def link_or_copy(src: Path, dst: Path, do_copy: bool, dry=False):
    if not src.exists():
        print(f"[WARN] Missing source: {src}"); return
    if dst.exists() or dst.is_symlink(): remove_if_exists(dst, dry)
    ensure_dir(dst.parent, dry)
    if dry: print(f"[DRY] {'copy' if do_copy else 'symlink'} {src} -> {dst}"); return
    if do_copy:
        if src.is_dir(): shutil.copytree(src, dst, dirs_exist_ok=True)
        else: shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())

_num_name_re = re.compile(r"(\d+)(?:\.\w+)?$")

def _candidate_color_files(color_dir: Path, idx: int) -> List[Path]:
    cands = []
    cands.append(color_dir / f"{idx}.jpg")
    cands.append(color_dir / f"{idx:06d}.jpg")
    cands.append(color_dir / f"{idx}.png")
    cands.append(color_dir / f"{idx:06d}.png")
    for p in color_dir.glob("*.jpg"):
        m = _num_name_re.search(p.stem)
        if m and int(m.group(1)) == idx: cands.append(p)
    for p in color_dir.glob("*.png"):
        m = _num_name_re.search(p.stem)
        if m and int(m.group(1)) == idx: cands.append(p)
    seen, uniq = set(), []
    for p in cands:
        if p.exists() and p not in seen:
            uniq.append(p); seen.add(p)
    return uniq

def build_video_color_from_idx(color_dir: Path, video_dir: Path,
                               idx_file: Path, do_copy: bool, dry=False):
    if not idx_file.exists():
        print(f"[WARN] video_idx.txt not found: {idx_file}"); return
    ensure_dir(video_dir, dry)
    lines = [ln.strip() for ln in idx_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    kept = 0
    for i, ln in enumerate(lines):
        try:
            idx = int(ln)
        except ValueError:
            print(f"[WARN] bad line in {idx_file}: {ln}"); continue
        cands = _candidate_color_files(color_dir, idx)
        if not cands:
            print(f"[WARN] no source for idx={idx} in {color_dir}")
            continue
        src = cands[0]
        dst = video_dir / f"frame{i}_{idx}{src.suffix.lower()}"
        link_or_copy(src, dst, do_copy, dry)
        kept += 1
    print(f"[INFO] video_color built: {kept} frames from {len(lines)} indices")

def main():
    args = parse_args()
    scans_dir = args.scannet_root / "scans"
    if not scans_dir.exists():
        print(f"[ERR] Expecting {scans_dir}"); sys.exit(1)

    allow = set(read_scenes_list(args.scenes_list))
    if allow: print(f"[INFO] Filtering {len(allow)} scenes")

    for scene_dir in sorted(scans_dir.glob("scene*_*")):
        scene = scene_dir.name
        if allow and scene not in allow: continue
        print(f"[SCENE] {scene}")

        out_scene = args.out_root / scene
        color_src = scene_dir / "image" / "color"
        depth_src = scene_dir / "image" / "depth"
        intr_src  = scene_dir / "image" / "intrinsic"
        pose_src  = scene_dir / "image" / "pose"

        color_dst = out_scene / "image_color"
        depth_dst = out_scene / "image_depth"
        intr_dst  = out_scene / "intrinsic"
        pose_dst  = out_scene / "pose"
        vcolor_dir = out_scene / "video_color"
        vdepth_dir = out_scene / "video_depth"
        vpose_dir  = out_scene / "video_pose"
        vidx_default = out_scene / "video_idx.txt"

        ensure_dir(out_scene, args.dry_run)

        if color_src.exists(): link_or_copy(color_src, color_dst, args.copy, args.dry_run)
        else: print(f"[WARN] Missing color/: {color_src}")

        if args.rgbd:
            if depth_src.exists(): link_or_copy(depth_src, depth_dst, args.copy, args.dry_run)
            else: print(f"[WARN] Missing depth/: {depth_src}")
            if intr_src.exists(): link_or_copy(intr_src, intr_dst, args.copy, args.dry_run)
            else: print(f"[WARN] Missing intrinsic/: {intr_src}")
            if pose_src.exists(): link_or_copy(pose_src, pose_dst, args.copy, args.dry_run)
            else: print(f"[WARN] Missing pose/: {pose_src}")

        ensure_dir(vcolor_dir, args.dry_run)
        if args.rgbd:
            ensure_dir(vdepth_dir, args.dry_run)
            ensure_dir(vpose_dir, args.dry_run)

        idx_file = vidx_default
        if args.video_idx_root is not None:
            idx_candidate = args.video_idx_root / scene / "video_idx.txt"
            if idx_candidate.exists():
                idx_file = idx_candidate

        if args.use_video_idx:
            build_video_color_from_idx(color_dst, vcolor_dir, idx_file, args.copy, args.dry_run)
        else:
            ensure_dir(vidx_default.parent, args.dry_run)
            if args.dry_run: print(f"[DRY] touch {vidx_default}")
            else:
                if not vidx_default.exists():
                    vidx_default.write_text("", encoding="utf-8")

    print(f"[DONE] Output root: {args.out_root.resolve()}")

if __name__ == "__main__":
    main()
