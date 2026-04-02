"""
Crossed Arms gesture detector.

Detection logic
---------------
Checks whether the wrists have crossed each other in the horizontal axis.
The required gap is normalised by the horizontal shoulder distance, which
scales with camera distance, making the threshold depth-invariant.

    shoulder_dist = ls.x - rs.x          (positive when subject faces camera)
    cross_gap     = lw.x - rw.x          (positive = right wrist is left of left wrist)
    cross_ratio   = cross_gap / shoulder_dist

Trigger: cross_ratio >= wrist_cross_ratio

A subject far from the camera has a narrow shoulder span, so a smaller
absolute wrist gap satisfies the same ratio — matching the proportionally
smaller geometry seen in the frame.

Configurable thresholds (config/thresholds.json → "crossed_arms")
-------------------------------------------------------------------
wrist_cross_ratio    (float, default 0.10)
    Wrists must overlap by at least this fraction of shoulder width.
    0.0 = wrists exactly aligned; 0.10 = right wrist 10 % of shoulder
    width to the left of left wrist.

min_shoulder_torso_ratio  (float, default 0.40)
    shoulder_dist / torso_height facing-camera gate (depth-invariant).

wrist_height_min_ratio / wrist_height_max_ratio  (float, defaults 0.0 / 1.1)
    Wrists must sit within this vertical band of the torso
    (0 = shoulder level, 1 = hip level).

min_landmark_visibility  (float, default 0.5)

confirmation_window / confirmation_ratio / cooldown_frames  (BaseGesture)
"""

from gestures.base_gesture import BaseGesture

_L_SHOULDER = 11
_R_SHOULDER = 12
_L_WRIST    = 15
_R_WRIST    = 16
_L_HIP      = 23
_R_HIP      = 24

_REQUIRED = [_L_SHOULDER, _R_SHOULDER, _L_WRIST, _R_WRIST, _L_HIP, _R_HIP]


class CrossedArms(BaseGesture):

    def __init__(self, thresholds: dict):
        super().__init__("crossed_arms", thresholds)
        self._last_metrics: dict = {}

    def detect(self, landmarks: dict) -> bool:
        pose = landmarks.get("pose")
        if pose is None:
            return False

        # ---- always compute metric (for overlay even before gates) ----
        try:
            ls = pose[_L_SHOULDER]; rs = pose[_R_SHOULDER]
            lw = pose[_L_WRIST];    rw = pose[_R_WRIST]
            s_dist = ls.x - rs.x
            if s_dist > 0:
                self._last_metrics = {"cross_ratio": (lw.x - rw.x) / s_dist}
        except (IndexError, AttributeError):
            pass

        # ---- visibility gate ------------------------------------------
        min_vis = self.thresholds.get("min_landmark_visibility", 0.5)
        try:
            for idx in _REQUIRED:
                if pose[idx].visibility < min_vis:
                    return False
        except IndexError:
            return False

        ls = pose[_L_SHOULDER]; rs = pose[_R_SHOULDER]
        lw = pose[_L_WRIST];    rw = pose[_R_WRIST]
        lh = pose[_L_HIP];      rh = pose[_R_HIP]

        # ---- torso height (needed for both gates) ---------------------
        shoulder_y = (ls.y + rs.y) / 2.0
        hip_y      = (lh.y + rh.y) / 2.0
        torso_h    = hip_y - shoulder_y
        if torso_h <= 0:
            return False

        # ---- facing-camera gate (depth-invariant ratio) ---------------
        shoulder_dist = ls.x - rs.x
        min_ratio = self.thresholds.get("min_shoulder_torso_ratio", 0.40)
        if shoulder_dist / torso_h < min_ratio:
            return False

        # ---- wrist crossing (shoulder-width normalised) ---------------
        cross_ratio = (lw.x - rw.x) / shoulder_dist
        threshold   = self.thresholds.get("wrist_cross_ratio", 0.10)
        if cross_ratio < threshold:
            return False

        # ---- wrist height gate ----------------------------------------
        h_min = self.thresholds.get("wrist_height_min_ratio", 0.0)
        h_max = self.thresholds.get("wrist_height_max_ratio", 1.1)
        for wrist in (lw, rw):
            if not (h_min <= (wrist.y - shoulder_y) / torso_h <= h_max):
                return False

        return True

    @property
    def state(self) -> dict:
        s = super().state
        s.update(self._last_metrics)
        return s
