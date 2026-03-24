"""
Raised Eyebrows gesture detector.

Detection logic
---------------
For each eyebrow we measure the vertical gap between the top of the eyebrow
arch and the upper eyelid, then normalise by the horizontal inter-iris
distance.  This single normalisation compensates for camera distance and
small head turns.

    raise_ratio_right = (eye_upper_right.y - brow_top_right.y) / inter_iris_dist
    raise_ratio_left  = (eye_upper_left.y  - brow_top_left.y)  / inter_iris_dist

Image y-axis: 0 = top, 1 = bottom.  Higher ratio = more raised eyebrows.
Both sides must exceed the threshold simultaneously.

Per-person calibration
----------------------
Because the resting brow-to-eye distance varies between people, the effective
threshold is personalised during a calibration phase:

    personalized_threshold = baseline + eyebrow_raise_delta

Where `baseline` is the Nth percentile of resting ratios collected during
the first `calibration_min_samples` stable face-detected frames.  Using a
low percentile (default 10th) avoids inflating the baseline if the person
occasionally raises their eyebrows during calibration.

Noise filtering during calibration uses two camera-distance-invariant guards:
  1. Stability streak  — only accept samples after `calib_stability_window`
                         consecutive valid frames (ghost detections are brief).
  2. Plausibility bounds — ratio must be in [_RATIO_MIN, _RATIO_MAX]; values
                           outside indicate a spurious face detection.

Configurable thresholds (config/thresholds.json → "raised_eyebrows")
----------------------------------------------------------------------
eyebrow_raise_ratio       (float, default 0.45)  — fallback threshold before calibration
eyebrow_raise_delta       (float, default 0.10)  — lift above baseline after calibration
calibration_min_samples   (int,   default 500)   — stable face frames needed to calibrate
calibration_percentile    (int,   default 10)    — percentile used for baseline
calib_stability_window    (int,   default 8)     — consecutive frames before trusting data
min_inter_iris_dist       (float, default 0.06)  — frontal-view guard for live detection
confirmation_window / confirmation_ratio / cooldown_frames — BaseGesture
"""

from __future__ import annotations
import statistics
from gestures.base_gesture import BaseGesture

# Top-arch landmarks for each eyebrow (MediaPipe 478-point face mesh)
_R_BROW_TOP = [70, 63, 105, 66, 107]
_L_BROW_TOP = [300, 293, 334, 296, 336]

_R_IRIS = 468
_L_IRIS = 473

_REQUIRED = _R_BROW_TOP + _L_BROW_TOP + [_R_IRIS, _L_IRIS]

# Physically plausible bounds for the brow-to-eye ratio (camera-distance invariant)
_RATIO_MIN = 0.10
_RATIO_MAX = 0.80


class RaisedEyebrows(BaseGesture):

    def __init__(self, thresholds: dict):
        super().__init__("raised_eyebrows", thresholds)
        self._calib_complete      = False
        self._calib_samples: list = []          # (ratio_r, ratio_l) tuples
        self._calib_stable_streak = 0
        self._personalized_thr    = thresholds.get("eyebrow_raise_ratio", 0.45)
        self._baseline            = None
        self._last_metrics: dict  = {}

    # ------------------------------------------------------------------
    def update(self, landmarks: dict) -> bool:
        if not self._calib_complete:
            self._collect_calibration(landmarks)
            target = self.thresholds.get("calibration_min_samples", 500)
            if len(self._calib_samples) >= target:
                self._finalize_calibration()
            return False

        return super().update(landmarks)

    # ------------------------------------------------------------------
    def detect(self, landmarks: dict) -> bool:
        face = landmarks.get("face")
        if face is None or len(face) <= max(_REQUIRED):
            return False

        inter_iris = abs(face[_R_IRIS].x - face[_L_IRIS].x)
        min_dist   = self.thresholds.get("min_inter_iris_dist", 0.06)
        if inter_iris < min_dist:
            return False

        brow_y_right = statistics.mean(face[i].y for i in _R_BROW_TOP)
        brow_y_left  = statistics.mean(face[i].y for i in _L_BROW_TOP)
        ratio_right  = (face[_R_IRIS].y - brow_y_right) / inter_iris
        ratio_left   = (face[_L_IRIS].y - brow_y_left)  / inter_iris
        self._last_metrics = {"inter_iris": inter_iris, "ratio_r": ratio_right, "ratio_l": ratio_left}

        return ratio_right >= self._personalized_thr and ratio_left >= self._personalized_thr

    # ------------------------------------------------------------------
    @property
    def state(self) -> dict:
        s = super().state
        s["calibrating"]      = not self._calib_complete
        s["calib_samples"]    = len(self._calib_samples)
        s["calib_target"]     = self.thresholds.get("calibration_min_samples", 500)
        s["baseline"]         = self._baseline
        s["personalized_thr"] = self._personalized_thr
        s.update(self._last_metrics)
        return s

    # ------------------------------------------------------------------
    def _collect_calibration(self, landmarks: dict) -> None:
        face      = landmarks.get("face")
        stability = self.thresholds.get("calib_stability_window", 8)

        if not face or len(face) <= max(_REQUIRED):
            self._calib_stable_streak = 0
            return

        inter_iris = abs(face[_R_IRIS].x - face[_L_IRIS].x)
        if inter_iris <= 0:
            self._calib_stable_streak = 0
            return

        brow_y_right = statistics.mean(face[i].y for i in _R_BROW_TOP)
        brow_y_left  = statistics.mean(face[i].y for i in _L_BROW_TOP)
        ratio_r = (face[_R_IRIS].y - brow_y_right) / inter_iris
        ratio_l = (face[_L_IRIS].y - brow_y_left)  / inter_iris
        self._last_metrics = {"inter_iris": inter_iris, "ratio_r": ratio_r, "ratio_l": ratio_l}

        # Plausibility gate — camera-distance invariant
        if not (_RATIO_MIN <= ratio_r <= _RATIO_MAX and
                _RATIO_MIN <= ratio_l <= _RATIO_MAX):
            self._calib_stable_streak = 0
            return

        self._calib_stable_streak += 1
        if self._calib_stable_streak < stability:
            return

        self._calib_samples.append((ratio_r, ratio_l))

    # ------------------------------------------------------------------
    def _finalize_calibration(self) -> None:
        pct     = self.thresholds.get("calibration_percentile", 10)
        resting = sorted(min(r, l) for r, l in self._calib_samples)
        idx     = max(0, int(len(resting) * pct / 100) - 1)

        self._baseline         = resting[idx]
        delta                  = self.thresholds.get("eyebrow_raise_delta", 0.10)
        self._personalized_thr = self._baseline + delta

        print(f"[RaisedEyebrows] Calibration complete. "
              f"baseline={self._baseline:.3f}  threshold={self._personalized_thr:.3f}")

        self._calib_complete = True
        self._calib_samples  = []     # free memory
