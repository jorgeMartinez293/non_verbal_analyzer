"""Base class for gesture detectors that require per-person calibration.

Extends BaseGesture with a calibration phase: the first N stable frames are
used to collect resting metrics.  Once enough samples are gathered, the
subclass computes a personalised threshold via ``finalize_calibration()``.

Until calibration is complete, ``update()`` always returns False (no gesture
can fire).

Subclasses must implement:
    ``detect(landmarks)``               — single-frame detection (from BaseGesture)
    ``extract_calibration_sample(landmarks)`` — return a metric tuple or None
    ``finalize_calibration(samples)``         — compute personalised thresholds
"""

from __future__ import annotations
from abc import abstractmethod
from gestures.base_gesture import BaseGesture


class CalibratingGesture(BaseGesture):

    def __init__(self, name: str, thresholds: dict):
        super().__init__(name, thresholds)
        self._calib_complete      = False
        self._calib_samples: list = []
        self._calib_stable_streak = 0

    # ------------------------------------------------------------------
    @abstractmethod
    def extract_calibration_sample(self, landmarks: dict):
        """Return a metric value/tuple for this frame, or None if invalid.

        Returning None resets the stability streak.
        """
        ...

    @abstractmethod
    def finalize_calibration(self, samples: list) -> None:
        """Called once when enough samples have been collected.

        The subclass should compute personalised thresholds here.
        """
        ...

    # ------------------------------------------------------------------
    def update(self, landmarks: dict) -> bool:
        if not self._calib_complete:
            self._run_calibration(landmarks)
            return False
        return super().update(landmarks)

    # ------------------------------------------------------------------
    def _run_calibration(self, landmarks: dict) -> None:
        stability = self.thresholds.get("calib_stability_window", 8)
        target    = self.thresholds.get("calibration_min_samples", 200)

        sample = self.extract_calibration_sample(landmarks)
        if sample is None:
            self._calib_stable_streak = 0
            return

        self._calib_stable_streak += 1
        if self._calib_stable_streak < stability:
            return

        self._calib_samples.append(sample)

        if len(self._calib_samples) >= target:
            self.finalize_calibration(self._calib_samples)
            self._calib_complete = True
            self._calib_samples  = []
