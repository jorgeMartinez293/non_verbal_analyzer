"""
Touch Face gesture detector.

Detection logic
---------------
A hand is touching the face when any hand landmark is within a threshold
fraction of the inter-shoulder distance from the nose.

    shoulder_dist = dist(pose[11], pose[12])
    min_dist      = min distance from any landmark of any detected hand to pose[0]
    ratio         = min_dist / shoulder_dist

Detected when ratio < hand_face_ratio threshold.

Normalization by inter-shoulder distance makes the threshold
camera-distance-invariant and person-independent.

Both left_hand and right_hand landmarks (when present) are checked
simultaneously.  The gesture fires if either hand is close enough.

Configurable thresholds (config/thresholds.json → "touch_face")
----------------------------------------------------------------
hand_face_ratio          (float, default 0.35)
    Hand must be within 35% of shoulder-width from the nose.

min_landmark_visibility  (float, default 0.3)
    MediaPipe visibility score gate for nose and shoulder landmarks.

confirmation_window / confirmation_ratio / cooldown_frames
    Handled by BaseGesture (sliding-window confirmation).
"""

import math
from gestures.base_gesture import BaseGesture

_NOSE       = 0
_L_SHOULDER = 11
_R_SHOULDER = 12

_REQUIRED_POSE = [_NOSE, _L_SHOULDER, _R_SHOULDER]


def _dist(a, b) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


class TouchFace(BaseGesture):

    def __init__(self, thresholds: dict):
        super().__init__("touch_face", thresholds)
        self._last_metrics: dict = {}

    def detect(self, landmarks: dict) -> bool:
        pose = landmarks.get("pose")
        if pose is None:
            return False

        # ---- visibility gate for required pose landmarks ---------------
        min_vis = self.thresholds.get("min_landmark_visibility", 0.3)
        try:
            for idx in _REQUIRED_POSE:
                if pose[idx].visibility < min_vis:
                    return False
        except IndexError:
            return False

        nose = pose[_NOSE]
        shoulder_dist = _dist(pose[_L_SHOULDER], pose[_R_SHOULDER])
        if shoulder_dist <= 0:
            return False

        # ---- collect hand landmarks ------------------------------------
        left_hand  = landmarks.get("left_hand")
        right_hand = landmarks.get("right_hand")
        if left_hand is None and right_hand is None:
            return False

        # ---- find minimum distance from any hand landmark to nose ------
        min_dist = float("inf")
        for hand in (left_hand, right_hand):
            if hand is None:
                continue
            for lm in hand:
                d = _dist(lm, nose)
                if d < min_dist:
                    min_dist = d

        ratio = min_dist / shoulder_dist
        self._last_metrics = {"hand_face_ratio": ratio}

        return ratio < self.thresholds.get("hand_face_ratio", 0.35)

    @property
    def state(self) -> dict:
        s = super().state
        s.update(self._last_metrics)
        return s
