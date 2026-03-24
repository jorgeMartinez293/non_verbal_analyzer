"""
Crossed Arms gesture detector.

Detection logic
---------------
We measure how far each wrist has travelled toward the OPPOSITE shoulder,
normalised by the real arm length (shoulder→elbow→wrist path).  This makes
the threshold geometrically meaningful and independent of shoulder width and
body proportions.

For each wrist we compute a "cross ratio":

    arm_len_r   = dist(R_SHOULDER, R_ELBOW) + dist(R_ELBOW, R_WRIST)
    arm_len_l   = dist(L_SHOULDER, L_ELBOW) + dist(L_ELBOW, L_WRIST)

    right_ratio = (rw.x - rs.x) / arm_len_r
    left_ratio  = (ls.x - lw.x) / arm_len_l

Interpretation (image x-axis: 0 = left edge, 1 = right edge):
    ratio = 0.0  →  wrist in line with its own shoulder (neutral)
    ratio = 0.4  →  wrist has crossed 40 % of arm length past own shoulder
    ratio = 1.0  →  wrist displaced a full arm length horizontally

If shoulder_dist ≤ min_shoulder_dist the subject is facing away or turned
too far sideways, so the gesture is suppressed without any separate
facing-camera check.

Configurable thresholds (config/thresholds.json → "crossed_arms")
-------------------------------------------------------------------
min_shoulder_torso_ratio  (float, default 0.40)
    Minimum ratio of shoulder_dist / torso_height.  Both distances
    shrink equally with subject-to-camera distance, so the ratio is
    invariant to depth.  A low ratio means the subject is turned
    sideways or facing away.  Raise to require a more frontal view.

wrist_cross_ratio    (float, default 0.40)
    Both wrists must reach at least this fraction of arm length past
    their own shoulder.  0.4 ≈ clearly crossing the midline.

wrist_height_min_ratio / wrist_height_max_ratio  (float, defaults 0.0 / 1.1)
    Wrists must sit within this vertical band of the torso
    (0 = shoulder level, 1 = hip level).

min_landmark_visibility  (float, default 0.5)
    MediaPipe visibility score gate for all required landmarks.

confirmation_window / confirmation_ratio / cooldown_frames
    Handled by BaseGesture (sliding-window confirmation).
"""

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

_REQUIRED = [_L_SHOULDER, _R_SHOULDER, _L_ELBOW, _R_ELBOW, _L_WRIST, _R_WRIST, _L_HIP, _R_HIP]


def _dist(a, b) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


class CrossedArms(BaseGesture):

    def __init__(self, thresholds: dict):
        super().__init__("crossed_arms", thresholds)
        self._last_metrics: dict = {}

    def detect(self, landmarks: dict) -> bool:
        pose = landmarks.get("pose")
        if pose is None:
            return False

        # ---- always compute metrics (shoulders + elbows + wrists) --
        try:
            _ls = pose[_L_SHOULDER]; _rs = pose[_R_SHOULDER]
            _le = pose[_L_ELBOW];    _re = pose[_R_ELBOW]
            _lw = pose[_L_WRIST];    _rw = pose[_R_WRIST]
            _arm_r = _dist(_rs, _re) + _dist(_re, _rw)
            _arm_l = _dist(_ls, _le) + _dist(_le, _lw)
            if _arm_r > 0 and _arm_l > 0:
                self._last_metrics = {
                    "right_ratio": (_rw.x - _rs.x) / _arm_r,
                    "left_ratio":  (_ls.x - _lw.x) / _arm_l,
                }
        except (IndexError, AttributeError):
            pass

        # ---- visibility gate ----------------------------------------
        min_vis = self.thresholds.get("min_landmark_visibility", 0.5)
        try:
            for idx in _REQUIRED:
                if pose[idx].visibility < min_vis:
                    return False
        except IndexError:
            return False

        lw = pose[_L_WRIST];    rw = pose[_R_WRIST]
        ls = pose[_L_SHOULDER]; rs = pose[_R_SHOULDER]
        le = pose[_L_ELBOW];    re = pose[_R_ELBOW]
        lh = pose[_L_HIP];      rh = pose[_R_HIP]

        # ---- torso height (needed for both gates below) -------------
        shoulder_y = (ls.y + rs.y) / 2.0
        hip_y      = (lh.y + rh.y) / 2.0
        torso_h    = hip_y - shoulder_y
        if torso_h <= 0:
            return False

        # ---- shoulder span gate (facing-camera + distance invariant)
        # shoulder_dist / torso_h is invariant to distance: both shrink
        # equally as the subject moves away.  A low ratio means the person
        # is turned sideways or away, regardless of how close they are.
        # shoulder_dist must also be positive (ls.x > rs.x) — negative
        # means the subject is facing away (mirrored axes).
        shoulder_dist = ls.x - rs.x
        min_ratio = self.thresholds.get("min_shoulder_torso_ratio", 0.40)
        if shoulder_dist / torso_h < min_ratio:
            return False

        # ---- crossing ratios (arm-length normalised) ----------------
        arm_len_r = _dist(rs, re) + _dist(re, rw)
        arm_len_l = _dist(ls, le) + _dist(le, lw)
        if arm_len_r <= 0 or arm_len_l <= 0:
            return False
        right_ratio = (rw.x - rs.x) / arm_len_r
        left_ratio  = (ls.x - lw.x) / arm_len_l

        threshold = self.thresholds.get("wrist_cross_ratio", 0.50)
        if right_ratio < threshold or left_ratio < threshold:
            return False

        h_min = self.thresholds.get("wrist_height_min_ratio", 0.0)
        h_max = self.thresholds.get("wrist_height_max_ratio", 1.1)
        for wrist in (lw, rw):
            ratio = (wrist.y - shoulder_y) / torso_h
            if not (h_min <= ratio <= h_max):
                return False

        return True

    @property
    def state(self) -> dict:
        s = super().state
        s.update(self._last_metrics)
        return s
