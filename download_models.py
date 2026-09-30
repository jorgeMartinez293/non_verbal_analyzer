"""Download the MediaPipe model bundles required by the analyzer into ``models/``.

Usage:
    uv run download_models.py [--force]
"""

import argparse
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://storage.googleapis.com/mediapipe-models"

MODELS = {
    "pose_landmarker_heavy.task": f"{BASE_URL}/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "face_landmarker.task": f"{BASE_URL}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "hand_landmarker.task": f"{BASE_URL}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
}

MODELS_DIR = Path(__file__).resolve().parent / "models"


def download(name: str, url: str, force: bool) -> None:
    dest = MODELS_DIR / name
    if dest.exists() and not force:
        print(f"[skip] {name} already present")
        return

    print(f"[get ] {name}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"[ ok ] {name} ({dest.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    for name, url in MODELS.items():
        download(name, url, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
