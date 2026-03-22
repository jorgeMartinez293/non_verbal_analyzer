"""
Raised Eyebrows gesture detector.

Detection logic
---------------
For each eyebrow we measure the vertical gap between the top of the eyebrow
arch and the upper eyelid, then normalise by the horizontal inter-iris
distance.  This single normalisation compensates for:

  • Camera distance — both the gap and the iris separation shrink equally
                      as the person moves away, so the ratio stays constant.
  • Small head turns — a moderate turn shrinks the iris distance but also
                       compresses the face vertically less, giving a mild
                       false-positive resistance.

    raise_ratio_right = (eye_upper_right.y - brow_top_right.y) / inter_iris_dist
    raise_ratio_left  = (eye_upper_left.y  - brow_top_left.y)  / inter_iris_dist

Image y-axis: 0 = top, 1 = bottom.
Because the eyebrows sit ABOVE the eyes, brow.y < eye.y, so the gap is
positive.  A higher ratio means more raised eyebrows.

Both sides must exceed `eyebrow_raise_ratio` simultaneously so that a
unilateral raise (e.g., one-eyebrow expressions) is not triggered.

Configurable thresholds (config/thresholds.json → "raised_eyebrows")
----------------------------------------------------------------------
eyebrow_raise_ratio    (float, default 0.35)
    Normalised gap threshold.  Lower → more sensitive.

min_inter_iris_dist    (float, default 0.06)
    Minimum horizontal distance between iris centres in normalised coords.
    Guards against near-profile views where the geometry breaks down.

confirmation_window / confirmation_ratio / cooldown_frames
    Handled by BaseGesture (sliding-window confirmation).
"""

from __future__ import annotations
import statistics
from gestures.base_gesture import BaseGesture

# Top-arch landmarks for each eyebrow (MediaPipe 478-point face mesh)
_R_BROW_TOP = [70, 63, 105, 66, 107]    # right eyebrow upper edge
_L_BROW_TOP = [300, 293, 334, 296, 336]  # left eyebrow upper edge

# Upper eyelid centre for each eye
_R_EYE_UPPER = 159
_L_EYE_UPPER = 386

# Iris centres (landmarks 468-477, added in the 478-point model)
_R_IRIS = 468
_L_IRIS = 473

_REQUIRED = _R_BROW_TOP + _L_BROW_TOP + [
    _R_EYE_UPPER, _L_EYE_UPPER, _R_IRIS, _L_IRIS
]


class RaisedEyebrows(BaseGesture):

    def __init__(self, thresholds: dict):
        super().__init__("raised_eyebrows", thresholds)

    def detect(self, landmarks: dict) -> bool:
        face = landmarks.get("face")
        if face is None or len(face) <= max(_REQUIRED):
            return False

        # ---- inter-iris distance gate (frontal-view guard) ----------
        inter_iris = abs(face[_R_IRIS].x - face[_L_IRIS].x)
        min_dist   = self.thresholds.get("min_inter_iris_dist", 0.06)
        if inter_iris < min_dist:
            return False

        # ---- eyebrow-to-eye gap, normalised by inter-iris distance --
        brow_y_right = statistics.mean(face[i].y for i in _R_BROW_TOP)
        brow_y_left  = statistics.mean(face[i].y for i in _L_BROW_TOP)

        eye_y_right = face[_R_EYE_UPPER].y
        eye_y_left  = face[_L_EYE_UPPER].y

        ratio_right = (eye_y_right - brow_y_right) / inter_iris
        ratio_left  = (eye_y_left  - brow_y_left)  / inter_iris

        threshold = self.thresholds.get("eyebrow_raise_ratio", 0.35)
        return ratio_right >= threshold and ratio_left >= threshold
