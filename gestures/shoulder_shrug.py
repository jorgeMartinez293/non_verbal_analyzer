"""
Shoulder Shrug gesture detector.

Detection logic
---------------
Tracks the vertical position of both shoulders relative to a calibrated
resting baseline.  A shrug is confirmed when both shoulders rise above
``baseline + min_shrug_ratio`` (normalised by torso height) and then
return toward baseline.

Uses per-person calibration (CalibratingGesture) to establish the
resting shoulder height during the first N frames.

Configurable thresholds (config/thresholds.json → "shoulder_shrug")
--------------------------------------------------------------------
min_shrug_ratio          (float, default 0.08) — rise above baseline (normalised)
max_asymmetry            (float, default 0.30) — max |left-right| difference
calibration_min_samples  (int,   default 100)
calibration_percentile   (int,   default 50)
calib_stability_window   (int,   default 8)
min_shoulder_torso_ratio (float, default 0.25)  — facing-camera gate
min_landmark_visibility  (float, default 0.3)
confirmation_window / confirmation_ratio / cooldown_frames — BaseGesture
"""

from __future__ import annotations
from gestures.calibrating_gesture import CalibratingGesture
from gestures.utils import visibility_gate, facing_camera

_L_SHOULDER = 11
_R_SHOULDER = 12
_L_HIP      = 23
_R_HIP      = 24

_REQUIRED = [_L_SHOULDER, _R_SHOULDER, _L_HIP, _R_HIP]


class ShoulderShrug(CalibratingGesture):

    def __init__(self, thresholds: dict):
        super().__init__("shoulder_shrug", thresholds)
        self._baseline       = None
        self._last_metrics: dict = {}

    # ------------------------------------------------------------------
    def extract_calibration_sample(self, landmarks: dict):
        pose = landmarks.get("pose")
        if pose is None:
            return None

        min_vis = self.thresholds.get("min_landmark_visibility", 0.3)
        if not visibility_gate(pose, _REQUIRED, min_vis):
            return None

        ls, rs = pose[_L_SHOULDER], pose[_R_SHOULDER]
        lh, rh = pose[_L_HIP], pose[_R_HIP]

        hip_y      = (lh.y + rh.y) / 2.0
        shoulder_y = (ls.y + rs.y) / 2.0
        torso_h    = hip_y - shoulder_y
        if torso_h <= 0:
            return None

        self._last_metrics = {"shoulder_ratio": torso_h}
        return torso_h

    # ------------------------------------------------------------------
    def finalize_calibration(self, samples: list) -> None:
        pct = self.thresholds.get("calibration_percentile", 50)
        sorted_samples = sorted(samples)
        idx = max(0, int(len(sorted_samples) * pct / 100) - 1)
        self._baseline = sorted_samples[idx]
        if self._baseline <= 0:
            self._baseline = None
            return
        print(f"[ShoulderShrug] Calibration complete. baseline={self._baseline:.4f}")

    # ------------------------------------------------------------------
    def detect(self, landmarks: dict) -> bool:
        if self._baseline is None:
            return False

        pose = landmarks.get("pose")
        if pose is None:
            return False

        min_vis = self.thresholds.get("min_landmark_visibility", 0.3)
        if not visibility_gate(pose, _REQUIRED, min_vis):
            return False

        # Facing-camera gate
        min_facing = self.thresholds.get("min_shoulder_torso_ratio", 0.25)
        fc = facing_camera(pose, min_facing)
        if fc is not None and not fc:
            return False

        ls, rs = pose[_L_SHOULDER], pose[_R_SHOULDER]
        lh, rh = pose[_L_HIP], pose[_R_HIP]

        hip_y      = (lh.y + rh.y) / 2.0
        shoulder_y = (ls.y + rs.y) / 2.0
        torso_h_current = hip_y - shoulder_y
        if torso_h_current <= 0 or self._baseline is None:
            return False

        rise = (torso_h_current - self._baseline) / self._baseline

        rise_l_raw = (hip_y - ls.y) / self._baseline
        rise_r_raw = (hip_y - rs.y) / self._baseline
        asymmetry  = abs(rise_l_raw - rise_r_raw)

        self._last_metrics = {"shoulder_ratio": torso_h_current / self._baseline, "shrug_rise": rise}

        max_asym = self.thresholds.get("max_asymmetry", 0.3)
        if asymmetry > max_asym:
            return False

        min_rise = self.thresholds.get("min_shrug_ratio", 0.08)
        return rise >= min_rise

    # ------------------------------------------------------------------
    @property
    def state(self) -> dict:
        s = super().state
        s["calibrating"] = not self._calib_complete
        s["calib_samples"] = len(self._calib_samples)
        s["baseline"] = self._baseline
        s.update(self._last_metrics)
        return s
