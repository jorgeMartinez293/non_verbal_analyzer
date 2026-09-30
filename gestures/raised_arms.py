"""
Raised Arms gesture detector.

Detection logic
---------------
Each wrist is considered "raised" when it sits above its own shoulder in the
image (MediaPipe Y increases downward, so raised ↔ lower Y value):

    raise_l = L_SHOULDER.y - L_WRIST.y   (positive = wrist above shoulder)
    raise_r = R_SHOULDER.y - R_WRIST.y

Both values are in the normalised [0,1] MediaPipe coordinate space, so no
additional normalization is needed: the comparison is already invariant to
the subject's distance from the camera.

A minimum margin (min_raise_margin) prevents the gesture from triggering
when the wrist is only marginally above the shoulder due to noise.

Configurable thresholds (config/thresholds.json → "raised_arms")
-----------------------------------------------------------------
min_raise_margin         (float, default 0.05)
    How many normalised units the wrist must be above the shoulder.
    ~0.05 ≈ 5 % of frame height, enough to require a clear intentional raise.

require_both_arms        (bool, default true)
    If true, both wrists must exceed the threshold simultaneously.
    If false, either wrist suffices (useful for single-arm raise).

min_shoulder_torso_ratio (float, default 0.40)
    Facing-camera gate: (L_SHOULDER.x - R_SHOULDER.x) / torso_height.
    Suppresses the gesture when the subject is turned sideways or away.

min_landmark_visibility  (float, default 0.3)
    MediaPipe visibility score gate for required landmarks.

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

_REQUIRED = [_L_SHOULDER, _R_SHOULDER, _L_WRIST, _R_WRIST]


class RaisedArms(BaseGesture):

    def __init__(self, thresholds: dict):
        super().__init__("raised_arms", thresholds)
        self._last_metrics: dict = {}

    def detect(self, landmarks: dict) -> bool:
        pose = landmarks.get("pose")
        if pose is None:
            return False

        # ---- always compute metrics so graphs update even on gate failure --
        try:
            ls = pose[_L_SHOULDER]; rs = pose[_R_SHOULDER]
            lw = pose[_L_WRIST];    rw = pose[_R_WRIST]
            self._last_metrics = {
                "raise_l": ls.y - lw.y,
                "raise_r": rs.y - rw.y,
            }
        except (IndexError, AttributeError):
            pass

        # ---- visibility gate --------------------------------------------
        min_vis = self.thresholds.get("min_landmark_visibility", 0.3)
        try:
            for idx in _REQUIRED:
                if pose[idx].visibility < min_vis:
                    return False
        except IndexError:
            return False

        ls = pose[_L_SHOULDER]; rs = pose[_R_SHOULDER]
        lw = pose[_L_WRIST];    rw = pose[_R_WRIST]

        # ---- facing-camera gate -----------------------------------------
        try:
            lh = pose[_L_HIP]; rh = pose[_R_HIP]
            shoulder_y = (ls.y + rs.y) / 2.0
            hip_y      = (lh.y + rh.y) / 2.0
            torso_h    = hip_y - shoulder_y
            if torso_h > 0:
                shoulder_dist = ls.x - rs.x
                min_ratio = self.thresholds.get("min_shoulder_torso_ratio", 0.40)
                if shoulder_dist / torso_h < min_ratio:
                    return False
        except (IndexError, AttributeError):
            pass   # hips not visible — skip the gate

        # ---- raise check ------------------------------------------------
        margin        = self.thresholds.get("min_raise_margin", 0.05)
        require_both  = self.thresholds.get("require_both_arms", True)
        raise_l       = ls.y - lw.y
        raise_r       = rs.y - rw.y

        if require_both:
            return raise_l >= margin and raise_r >= margin
        else:
            return raise_l >= margin or raise_r >= margin

    @property
    def state(self) -> dict:
        s = super().state
        s.update(self._last_metrics)
        return s
