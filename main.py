"""
Non-Verbal Language Analyzer — debug-viz branch
================================================
Processes a pre-recorded MP4 and outputs a fully annotated copy with:
  • Pose landmarks (wrists, elbows, shoulders, hips)
  • Wrist-to-wrist crossing line (green = condition met, red = not)
  • Torso height band
  • Live threshold values (top-left HUD)
  • Gesture confirmation progress bar (bottom of frame)
  • DETECTED banner when a gesture fires

Usage
-----
    python main.py <video_path> [--output-dir OUTPUT] [--config CONFIG]
"""

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Output a fully annotated debug video with gesture detection."
    )
    parser.add_argument("video_path", help="Path to the input .mp4 file.")
    parser.add_argument("--output-dir", default="output",
                        help="Directory for the annotated output video (default: output/).")
    parser.add_argument("--config", default="config/thresholds.json",
                        help="Thresholds config file (default: config/thresholds.json).")
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
    processor       = VideoProcessor(gesture_manager, config, output_dir=args.output_dir)
    processor.process(args.video_path)


if __name__ == "__main__":
    main()
