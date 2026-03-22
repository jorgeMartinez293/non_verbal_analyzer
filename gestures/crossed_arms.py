"""
Crossed Arms gesture detector.

Detection logic
---------------
A person is considered to have their arms crossed when:
  1. The right wrist is to the LEFT of the left wrist in image space
     (i.e. right_wrist.x < left_wrist.x minus a configurable margin).
  2. Both wrists sit within a configurable vertical band of the torso
     (measured relative to the shoulder-to-hip distance).
  3. All required landmarks have sufficient visibility.

All numeric frontiers are read from thresholds.json under the key
"crossed_arms", making them easy to tune without touching code.

Thresholds (all configurable via config/thresholds.json)
---------------------------------------------------------
wrist_cross_margin (float, default 0.05)
    The right wrist must be at least this much (in normalised x units)
    to the LEFT of the left wrist.  Raise to require a more pronounced
    crossing; lower to catch subtle ones.

wrist_height_min_ratio (float, default 0.0)
    Minimum relative vertical position of both wrists inside the torso
    band (0.0 = shoulder level, 1.0 = hip level).  Values below 0 allow
    the wrists to be above the shoulders.

wrist_height_max_ratio (float, default 1.1)
    Maximum relative vertical position.  Values above 1.0 allow the
    wrists to drop slightly below the hips.

min_landmark_visibility (float, default 0.5)
    MediaPipe visibility score threshold.  Landmarks below this value
    are treated as occluded and the gesture is not triggered.

confirmation_frames (int, default 12)
    How many consecutive frames the condition must hold before the
    gesture is considered confirmed (handled by BaseGesture).

cooldown_frames (int, default 90)
    Frames to wait after a confirmed detection before the gesture can
    trigger again (handled by BaseGesture).
"""

from gestures.base_gesture import BaseGesture

# MediaPipe Pose landmark indices (same for Holistic)
_L_SHOULDER = 11
_R_SHOULDER = 12
_L_ELBOW    = 13
_R_ELBOW    = 14
_L_WRIST    = 15
_R_WRIST    = 16
_L_HIP      = 23
_R_HIP      = 24

_REQUIRED = [_L_SHOULDER, _R_SHOULDER, _L_ELBOW, _R_ELBOW,
             _L_WRIST, _R_WRIST, _L_HIP, _R_HIP]


class CrossedArms(BaseGesture):

    def __init__(self, thresholds: dict):
        super().__init__("crossed_arms", thresholds)

    # ------------------------------------------------------------------
    def detect(self, landmarks: dict) -> bool:
        pose = landmarks.get("pose")
        if pose is None:
            return False

        # ---- visibility gate -----------------------------------------
        min_vis = self.thresholds.get("min_landmark_visibility", 0.5)
        try:
            for idx in _REQUIRED:
                if pose[idx].visibility < min_vis:
                    return False
        except IndexError:
            return False

        lw = pose[_L_WRIST]
        rw = pose[_R_WRIST]
        ls = pose[_L_SHOULDER]
        rs = pose[_R_SHOULDER]
        lh = pose[_L_HIP]
        rh = pose[_R_HIP]

        # ---- crossing check ------------------------------------------
        # In normalised image coords x grows left→right.
        # Crossed arms: right wrist ends up on the LEFT side of the body,
        # so right_wrist.x < left_wrist.x.
        margin = self.thresholds.get("wrist_cross_margin", 0.05)
        if not (rw.x < lw.x - margin):
            return False

        # ---- height check (relative to torso) ------------------------
        shoulder_y = (ls.y + rs.y) / 2.0
        hip_y      = (lh.y + rh.y) / 2.0
        torso_h    = hip_y - shoulder_y

        if torso_h <= 0:
            return False

        h_min = self.thresholds.get("wrist_height_min_ratio", 0.0)
        h_max = self.thresholds.get("wrist_height_max_ratio", 1.1)

        for wrist in (lw, rw):
            ratio = (wrist.y - shoulder_y) / torso_h
            if not (h_min <= ratio <= h_max):
                return False

        return True
