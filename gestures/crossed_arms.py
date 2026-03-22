"""
Crossed Arms gesture detector.

Detection logic
---------------
Rather than comparing raw wrist x-positions, we measure how far each wrist
has travelled toward the OPPOSITE shoulder, normalised by the inter-shoulder
distance.  This single normalisation compensates for both:

  • Camera angle   — if the person is turned, the shoulder span shrinks and
                     so do the wrist-to-shoulder distances proportionally.
  • Camera distance — if the person steps back, everything scales down
                     equally, so the ratio stays the same.

For each wrist we compute a "cross ratio":

    right_ratio = (rw.x - rs.x) / shoulder_dist
    left_ratio  = (ls.x - lw.x) / shoulder_dist

    shoulder_dist = ls.x - rs.x   (positive when facing the camera)

Interpretation (image x-axis: 0 = left edge, 1 = right edge):
    ratio = 0  →  wrist is at its own shoulder (neutral)
    ratio = 1  →  wrist has reached the opposite shoulder (fully crossed)
    ratio > 1  →  wrist is past the opposite shoulder

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

wrist_cross_ratio    (float, default 0.50)
    Both wrists must reach at least this ratio toward the opposite
    shoulder.  0.5 = halfway, 1.0 = fully at the opposite shoulder.

wrist_height_min_ratio / wrist_height_max_ratio  (float, defaults 0.0 / 1.1)
    Wrists must sit within this vertical band of the torso
    (0 = shoulder level, 1 = hip level).

min_landmark_visibility  (float, default 0.5)
    MediaPipe visibility score gate for all required landmarks.

confirmation_window / confirmation_ratio / cooldown_frames
    Handled by BaseGesture (sliding-window confirmation).
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

    def detect(self, landmarks: dict) -> bool:
        pose = landmarks.get("pose")
        if pose is None:
            return False

        # ---- visibility gate ----------------------------------------
        min_vis = self.thresholds.get("min_landmark_visibility", 0.5)
        try:
            for idx in _REQUIRED:
                if pose[idx].visibility < min_vis:
                    return False
        except IndexError:
            return False

        lw = pose[_L_WRIST];   rw = pose[_R_WRIST]
        ls = pose[_L_SHOULDER]; rs = pose[_R_SHOULDER]
        lh = pose[_L_HIP];     rh = pose[_R_HIP]

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

        # ---- crossing ratios (scale & angle invariant) --------------
        right_ratio = (rw.x - rs.x) / shoulder_dist
        left_ratio  = (ls.x - lw.x) / shoulder_dist

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
