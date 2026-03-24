"""
Gesture Manager

Dynamically discovers and loads every gesture defined in the gestures/
directory (one class per file, inheriting from BaseGesture).  Gesture
thresholds are injected from the config dict keyed by the gesture file's
stem (e.g. crossed_arms.py → config["crossed_arms"]).

To add a new gesture:
  1. Create gestures/<your_gesture>.py
  2. Define a class that inherits from BaseGesture and implements detect()
  3. Add a matching key in config/thresholds.json with its thresholds
  4. Done — the manager will pick it up automatically on the next run.
"""

import importlib.util
import inspect
from pathlib import Path

from gestures.base_gesture import BaseGesture


class GestureManager:

    def __init__(self, config: dict):
        self.config = config
        self.gestures: list[BaseGesture] = []
        self._load_gestures()

    # ------------------------------------------------------------------
    def _load_gestures(self) -> None:
        gestures_dir = Path(__file__).parent.parent / "gestures"
        loaded = []

        for gesture_file in sorted(gestures_dir.glob("*.py")):
            if gesture_file.name.startswith("_") or gesture_file.stem == "base_gesture":
                continue

            spec   = importlib.util.spec_from_file_location(gesture_file.stem, gesture_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for _, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, BaseGesture)
                    and obj is not BaseGesture
                    and obj.__module__ == module.__name__
                ):
                    thresholds = self.config.get(gesture_file.stem, {})
                    instance   = obj(thresholds)
                    self.gestures.append(instance)
                    loaded.append(gesture_file.stem)
                    break  # one gesture class per file

        if loaded:
            print(f"[GestureManager] Loaded gestures: {', '.join(loaded)}")
        else:
            print("[GestureManager] Warning: no gesture files found.")

    # ------------------------------------------------------------------
    def get_states(self) -> dict[str, dict]:
        """Return the current internal state of every loaded gesture."""
        return {g.name: g.state for g in self.gestures}

    def reset_windows(self) -> None:
        """Clear the sliding-window confirmation state of every gesture.

        Called when tracking is deemed unstable so that noisy landmark
        positions do not accumulate toward a false positive.
        """
        for gesture in self.gestures:
            gesture._window.clear()

    def process_frame(self, landmarks: dict) -> list[str]:
        """
        Update every gesture detector with the current frame's landmarks.

        Returns a list of gesture names that fired this frame (usually
        empty; at most one entry per gesture per cooldown window).
        """
        triggered = []
        for gesture in self.gestures:
            if gesture.update(landmarks):
                triggered.append(gesture.name)
        return triggered
