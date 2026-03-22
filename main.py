"""
Non-Verbal Language Analyzer
=============================
Analyzes a pre-recorded MP4 video for non-verbal gestures using
MediaPipe Holistic (pose + face + hands).

When a gesture is confirmed, a ~20-frame clip (configurable via
config/thresholds.json) is saved to the output directory.

Usage
-----
    python main.py <video_path> [--output-dir OUTPUT] [--config CONFIG]

Examples
--------
    python main.py interview.mp4
    python main.py interview.mp4 --output-dir results/
    python main.py interview.mp4 --config config/thresholds.json
"""

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze non-verbal language in a pre-recorded MP4 video."
    )
    parser.add_argument(
        "video_path",
        help="Path to the input .mp4 video file.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory where gesture clips will be saved (default: output/).",
    )
    parser.add_argument(
        "--config",
        default="config/thresholds.json",
        help="Path to the thresholds configuration file (default: config/thresholds.json).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Draw pose landmarks and live threshold values on saved clips.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        print(f"[ERROR] Config file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with cfg_path.open() as f:
        return json.load(f)


def main() -> None:
    args   = parse_args()
    config = load_config(args.config)

    # Late imports so startup errors surface cleanly
    from core.clip_saver       import ClipSaver
    from core.gesture_manager  import GestureManager
    from core.video_processor  import VideoProcessor

    clip_cfg = config.get("clip", {})
    clip_saver = ClipSaver(
        frames_before = clip_cfg.get("frames_before", 10),
        frames_after  = clip_cfg.get("frames_after",  10),
        output_dir    = args.output_dir,
    )

    gesture_manager = GestureManager(config)
    processor       = VideoProcessor(gesture_manager, clip_saver, config, debug=args.debug)

    processor.process(args.video_path)


if __name__ == "__main__":
    main()
