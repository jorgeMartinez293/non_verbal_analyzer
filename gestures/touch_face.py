"""
Touch Face gesture detector.

Detection logic
---------------
A hand is touching the face when any hand landmark is within a threshold
fraction of the inter-shoulder distance from the nose, AND the corresponding
arm does not appear foreshortened (which would indicate it is extended toward
the camera rather than actually touching the face).

    shoulder_dist = dist(pose[11], pose[12])
    min_dist      = min distance from any landmark of the hand to pose[0] (nose)
    ratio         = min_dist / shoulder_dist

Step 1 — proximity check:
    ratio < hand_face_ratio

Step 2 — arm foreshortening gate:
    arm_len_2d = dist(shoulder→elbow) + dist(elbow→wrist)  [2D pose landmarks]
    arm_len_ratio = arm_len_2d / shoulder_dist

    When a straight arm points toward the camera, its 2D apparent length shrinks
    dramatically (perspective foreshortening).  A very low arm_len_ratio therefore
    means the arm is extended at the camera, not reaching the face.

    Gesture is accepted only when arm_len_ratio >= min_arm_ratio.
    If elbow/wrist visibility is below the threshold the arm gate is skipped
    (the gesture is accepted) to avoid rejecting real face-touches that happen
    with partial occlusion of the arm.

Both left_hand and right_hand are checked independently.  Each hand is paired
with its corresponding arm (left_hand ↔ L_SHOULDER/L_ELBOW/L_WRIST, etc.).
The gesture fires as soon as one hand passes both checks.

Configurable thresholds (config/thresholds.json → "touch_face")
----------------------------------------------------------------
hand_face_ratio          (float, default 0.35)
    Hand must be within 35 % of shoulder-width from the nose.

min_arm_ratio            (float, default 0.50)
    Minimum apparent arm length (as fraction of shoulder-width) required to
    accept a proximity hit.  Below this the arm is considered foreshortened
    (pointing toward the camera).  Typical values when touching face: 0.7–1.4.
    Typical values when arm stretched at camera: 0.05–0.25.

min_landmark_visibility  (float, default 0.3)
    MediaPipe visibility score gate for nose, shoulders, and (when available)
    elbow/wrist landmarks.

confirmation_window / confirmation_ratio / cooldown_frames
    Handled by BaseGesture (sliding-window confirmation).
"""

import math
from gestures.base_gesture import BaseGesture

_NOSE       = 0
_L_SHOULDER = 11
_R_SHOULDER = 12
_L_ELBOW    = 13
_R_ELBOW    = 14
_L_WRIST    = 15
_R_WRIST    = 16

_REQUIRED_POSE = [_NOSE, _L_SHOULDER, _R_SHOULDER]

# Maps each hand key to its (shoulder, elbow, wrist) pose indices
_HAND_ARMS = {
    "left_hand":  (_L_SHOULDER, _L_ELBOW, _L_WRIST),
    "right_hand": (_R_SHOULDER, _R_ELBOW, _R_WRIST),
}


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

        prox_thr      = self.thresholds.get("hand_face_ratio", 0.35)
        min_arm_ratio = self.thresholds.get("min_arm_ratio", 0.5)

        # Track the global minimum distance ratio for the HUD graph,
        # independently of whether the arm gate passes or fails.
        overall_min_ratio = float("inf")

        for hand_key, (sh_idx, el_idx, wr_idx) in _HAND_ARMS.items():
            hand_lms = landmarks.get(hand_key)
            if hand_lms is None:
                continue

            hand_min_dist = min(_dist(lm, nose) for lm in hand_lms)
            ratio = hand_min_dist / shoulder_dist

            if ratio < overall_min_ratio:
                overall_min_ratio = ratio

            if ratio >= prox_thr:
                continue  # this hand is not near the face

            # ---- arm foreshortening gate --------------------------------
            # Skip the gate if arm landmarks are not visible (partial occlusion).
            try:
                el = pose[el_idx]
                wr = pose[wr_idx]
                if el.visibility >= min_vis and wr.visibility >= min_vis:
                    arm_len = _dist(pose[sh_idx], el) + _dist(el, wr)
                    if arm_len / shoulder_dist < min_arm_ratio:
                        continue  # arm appears foreshortened → pointing at camera
            except IndexError:
                pass  # can't check, accept anyway

            # Both proximity and arm checks passed
            self._last_metrics = {"hand_face_ratio": overall_min_ratio}
            return True

        # No hand passed both checks
        if overall_min_ratio < float("inf"):
            self._last_metrics = {"hand_face_ratio": overall_min_ratio}
        return False

    @property
    def state(self) -> dict:
        s = super().state
        s.update(self._last_metrics)
        return s
