"""
Open Arms gesture detector.

Detection logic
---------------
Two conditions must be met simultaneously:

  1. SPREAD — how far the wrists have opened beyond shoulder width, normalised
     by the total arm length (shoulder→elbow→wrist path on each side):

         arm_len_l   = dist(L_SHOULDER, L_ELBOW) + dist(L_ELBOW, L_WRIST)
         arm_len_r   = dist(R_SHOULDER, R_ELBOW) + dist(R_ELBOW, R_WRIST)
         open_ratio  = (dist(L_WRIST, R_WRIST) - dist(L_SHOULDER, R_SHOULDER))
                       / (arm_len_l + arm_len_r)

     Interpretation:
       0.0  → wrists at the same distance as shoulders (arms at sides)
       1.0  → arms fully horizontal (maximum possible spread)
       0.5  → arms open to roughly 45 ° from vertical

  2. FACING GATE — same shoulder/torso ratio used in crossed_arms:
     suppresses detection when the subject is turned sideways or away.

Configurable thresholds (config/thresholds.json → "open_arms")
---------------------------------------------------------------
min_open_ratio            (float, default 0.50)
    Minimum (wrist_dist - shoulder_dist) / total_arm_len ratio.
    0.5 ≈ arms open to ~45 ° from vertical.

min_shoulder_torso_ratio  (float, default 0.40)
    Minimum shoulder_x_span / torso_height — facing-camera gate.

min_landmark_visibility   (float, default 0.5)
    MediaPipe visibility score gate for all required landmarks.

confirmation_window / confirmation_ratio / cooldown_frames
    Handled by BaseGesture (sliding-window confirmation).
"""

from __future__ import annotations
import math
from gestures.base_gesture import BaseGesture

_L_SHOULDER = 11
_R_SHOULDER = 12
_L_ELBOW    = 13
_R_ELBOW    = 14
_L_WRIST    = 15
_R_WRIST    = 16
_L_HIP      = 23
_R_HIP      = 24


def _dist(a, b) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


class OpenArms(BaseGesture):

    def __init__(self, thresholds: dict):
        super().__init__("open_arms", thresholds)
        self._last_metrics: dict = {}

    def detect(self, landmarks: dict) -> bool:
        pose = landmarks.get("pose")
        if pose is None:
            return False

        try:
            lw = pose[_L_WRIST];    rw = pose[_R_WRIST]
            ls = pose[_L_SHOULDER]; rs = pose[_R_SHOULDER]
            le = pose[_L_ELBOW];    re = pose[_R_ELBOW]
            lh = pose[_L_HIP];      rh = pose[_R_HIP]
        except (IndexError, AttributeError):
            return False

        # ---- visibility gate --------------------------------------------
        min_vis = self.thresholds.get("min_landmark_visibility", 0.5)
        for lm in (lw, rw, ls, rs, le, re):
            if lm.visibility < min_vis:
                return False

        # ---- always compute metrics ----------------------------------
        shoulder_dist = _dist(ls, rs)
        wrist_dist    = _dist(lw, rw)
        arm_len_l     = _dist(ls, le) + _dist(le, lw)
        arm_len_r     = _dist(rs, re) + _dist(re, rw)
        total_arm_len = arm_len_l + arm_len_r
        open_ratio    = (wrist_dist - shoulder_dist) / total_arm_len if total_arm_len > 0 else 0.0
        self._last_metrics = {"shoulder_dist": shoulder_dist, "open_ratio": open_ratio}

        # ---- facing-camera gate (skipped when hips not visible) -----
        if pose[_L_HIP].visibility >= 0.3 and pose[_R_HIP].visibility >= 0.3:
            shoulder_y = (ls.y + rs.y) / 2.0
            torso_h    = (lh.y + rh.y) / 2.0 - shoulder_y
            if torso_h > 0:
                min_facing = self.thresholds.get("min_shoulder_torso_ratio", 0.25)
                if (ls.x - rs.x) / torso_h < min_facing:
                    return False

        # ---- spread gate --------------------------------------------
        if total_arm_len <= 0:
            return False

        threshold = self.thresholds.get("min_open_ratio", 0.50)
        return open_ratio >= threshold

    @property
    def state(self) -> dict:
        s = super().state
        s.update(self._last_metrics)
        return s
