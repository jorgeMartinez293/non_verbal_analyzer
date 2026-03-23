"""
Open Arms gesture detector.

Detection logic
---------------
Two conditions must be met simultaneously:

  1. SPREAD — the Euclidean distance between the two wrists must be large
     relative to the Euclidean distance between the shoulders:

         open_ratio = euclidean(L_WRIST, R_WRIST) / euclidean(L_SHOULDER, R_SHOULDER)

     Using Euclidean (not just X) for both distances means the ratio is
     stable regardless of the arm angle (horizontal, diagonal, etc.) and
     compensates for slight camera rotations.

     At rest (arms hanging): wrists are roughly hip-width apart, ratio ≈ 1.0–1.3
     Arms open:              wrists spread wide,                  ratio ≈ 2.0+

  2. HEIGHT — both wrists must be at or above their respective elbows
     (wrist.y ≤ elbow.y in image coords where y increases downward).
     This prevents the spread condition from triggering when arms simply
     hang at the sides (where wrists are below elbows).

  3. FACING GATE — same shoulder/torso ratio used in crossed_arms:
     suppresses detection when the subject is turned sideways or away.

Configurable thresholds (config/thresholds.json → "open_arms")
---------------------------------------------------------------
min_open_ratio            (float, default 1.8)
    Minimum wrist-to-wrist / shoulder-to-shoulder Euclidean ratio.

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
_L_WRIST    = 15
_R_WRIST    = 16
_L_HIP      = 23
_R_HIP      = 24

_REQUIRED = [_L_SHOULDER, _R_SHOULDER, _L_WRIST, _R_WRIST]


def _dist(a, b) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


class OpenArms(BaseGesture):

    def __init__(self, thresholds: dict):
        super().__init__("open_arms", thresholds)

    def detect(self, landmarks: dict) -> bool:
        pose = landmarks.get("pose")
        if pose is None:
            return False

        try:
            lw = pose[_L_WRIST];    rw = pose[_R_WRIST]
            ls = pose[_L_SHOULDER]; rs = pose[_R_SHOULDER]
            lh = pose[_L_HIP];      rh = pose[_R_HIP]
        except (IndexError, AttributeError):
            return False

        # ---- facing-camera gate (skipped when hips not visible) -----
        shoulder_x_dist = ls.x - rs.x
        if pose[_L_HIP].visibility >= 0.3 and pose[_R_HIP].visibility >= 0.3:
            shoulder_y = (ls.y + rs.y) / 2.0
            torso_h    = (lh.y + rh.y) / 2.0 - shoulder_y
            if torso_h > 0:
                min_facing = self.thresholds.get("min_shoulder_torso_ratio", 0.40)
                if shoulder_x_dist / torso_h < min_facing:
                    return False

        # ---- spread gate: wrist distance / shoulder distance --------
        shoulder_dist = _dist(ls, rs)
        if shoulder_dist <= 0:
            return False

        open_ratio = _dist(lw, rw) / shoulder_dist
        threshold  = self.thresholds.get("min_open_ratio", 1.8)
        return open_ratio >= threshold
