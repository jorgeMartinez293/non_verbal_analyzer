"""
Non-Verbal Language Analyzer
=============================
Analyzes a pre-recorded MP4 video for non-verbal gestures using
MediaPipe (pose + face + hands, heavy model).

Default behaviour
-----------------
    Detects gestures and saves a ~20-frame clip (configurable) for each
    confirmed detection to:
        <output_dir>/<video_stem>/<gesture>_frame<N>.mp4

Debug mode  (--debug)
---------------------
    Draws pose landmarks, live threshold values, and gesture state on
    every frame and writes a single annotated video to:
        <output_dir>/<video_stem>_annotated.mp4
    Useful for tuning thresholds and verifying detection logic.

Usage
-----
    python main.py <video_path> [--output-dir OUTPUT] [--config CONFIG] [--debug]
"""

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze non-verbal language in a pre-recorded MP4 video."
    )
    parser.add_argument("video_path",
                        help="Path to the input .mp4 file.")
    parser.add_argument("--output-dir", default="output",
                        help="Output directory (default: output/).")
    parser.add_argument("--config", default="config/thresholds.json",
                        help="Thresholds config file (default: config/thresholds.json).")
    parser.add_argument("--debug", action="store_true",
                        help="Output a full annotated video instead of gesture clips.")
    return parser.parse_args()


def load_config(path: str) -> dict:
    cfg = Path(path)
    if not cfg.exists():
        print(f"[ERROR] Config not found: {path}", file=sys.stderr)
        sys.exit(1)
    with cfg.open() as f:
        return json.load(f)


def main() -> None:
    args   = parse_args()
    config = load_config(args.config)

    from core.gesture_manager import GestureManager
    from core.video_processor import VideoProcessor

    gesture_manager = GestureManager(config)

    if args.debug:
        processor = VideoProcessor(
            gesture_manager = gesture_manager,
            config          = config,
            output_dir      = args.output_dir,
            debug           = True,
        )
    else:
        from core.clip_saver import ClipSaver
        clip_cfg   = config.get("clip", {})
        clip_saver = ClipSaver(
            frames_before = clip_cfg.get("frames_before", 10),
            frames_after  = clip_cfg.get("frames_after",  10),
            output_dir    = args.output_dir,
        )
        processor = VideoProcessor(
            gesture_manager = gesture_manager,
            config          = config,
            clip_saver      = clip_saver,
            output_dir      = args.output_dir,
            debug           = False,
        )

    processor.process(args.video_path)


if __name__ == "__main__":
    main()
