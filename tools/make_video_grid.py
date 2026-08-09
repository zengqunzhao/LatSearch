#!/usr/bin/env python3
"""Combine candidate MP4 files into comparison grids."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create MP4 grids from candidate videos.")
    parser.add_argument("video_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--candidates", type=int, default=6)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=24)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.video_dir.is_dir():
        raise FileNotFoundError(f"Video directory does not exist: {args.video_dir}")
    if args.candidates < 1 or args.columns < 1:
        raise ValueError("--candidates and --columns must be positive.")

    import moviepy.video.fx.all as vfx
    from moviepy.editor import VideoFileClip, clips_array

    output_dir = args.output_dir or args.video_dir / "grids"
    output_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in args.video_dir.glob("*_candidate_*.mp4"):
        groups[path.name.split("_candidate_", maxsplit=1)[0]].append(path)

    for group_name, paths in groups.items():
        if len(paths) != args.candidates:
            print(f"Skipping {group_name}: expected {args.candidates}, found {len(paths)}")
            continue
        clips = [
            VideoFileClip(str(path)).resize((args.width, args.height)) for path in sorted(paths)
        ]
        try:
            duration = min(clip.duration for clip in clips)
            clips = [
                vfx.margin(clip.subclip(0, duration), mar=5, color=(255, 255, 255))
                for clip in clips
            ]
            rows = [
                clips[index : index + args.columns] for index in range(0, len(clips), args.columns)
            ]
            output_path = output_dir / f"{group_name}_grid.mp4"
            clips_array(rows).write_videofile(str(output_path), fps=args.fps)
        finally:
            for clip in clips:
                clip.close()


if __name__ == "__main__":
    main()
