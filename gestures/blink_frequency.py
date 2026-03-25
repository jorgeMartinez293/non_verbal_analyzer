"""
Blink Frequency detector.

Detection logic
---------------
Uses the Eye Aspect Ratio (EAR, Soukupová 2016) to detect individual blink
events and compute a rolling blinks-per-minute (BPM) metric.

    EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)

The EAR formula is already scale-invariant (ratio of the eye's own geometry),
so no per-person calibration is needed.  A fixed threshold (~0.20) works
reliably across most people, matching the original paper and all published
implementations.

MediaPipe FaceMesh 478-pt landmark indices used:
    Right eye: outer=33, inner=133, top=(160, 158), bottom=(144, 153)
    Left eye:  outer=263, inner=362, top=(385, 387), bottom=(380, 373)

Blink event detection (state machine)
--------------------------------------
OPEN    → EAR_avg drops below ear_threshold → CLOSING (record start frame)
CLOSING → EAR_avg rises back               → blink valid if duration ∈ [min_frames, max_frames]
        → duration > max_blink_frames       → voluntary eye hold, discard → OPEN

Rolling BPM
-----------
A deque stores frame numbers of confirmed blinks.  Each frame, entries older
than `rate_window_frames` are purged.

    bpm = len(deque) * (fps * 60 / rate_window_frames)

This detector never triggers (update() always returns False).
Its BPM metric is exposed via the `state` property for the debug overlay.

Configurable thresholds (config/thresholds.json → "blink_frequency")
----------------------------------------------------------------------
ear_threshold        (float, default 0.20) — EAR below this = eye closed
detect_min_ear       (float, default 0.10) — sanity gate: skip frame if below (tracking loss)
rate_window_frames   (int,   default 900)  — rolling window length in frames
blink_min_frames     (int,   default 2)    — min blink duration (noise filter)
blink_max_frames     (int,   default 12)   — max blink duration (eye hold filter)
fps                  (float, default 30)   — video fps for BPM conversion
max_bpm_display      (float, default 40)   — overlay bar scale max
"""

from __future__ import annotations
import math
from collections import deque
from gestures.base_gesture import BaseGesture

# --------------------------------------------------------------------------
# Landmark indices (MediaPipe FaceMesh 478-pt)
# Each eye: (outer_corner, inner_corner, top_a, top_b, bottom_a, bottom_b)
# --------------------------------------------------------------------------
_R_EYE = (33,  133, 160, 158, 144, 153)
_L_EYE = (263, 362, 385, 387, 380, 373)

_REQUIRED = list(_R_EYE) + list(_L_EYE)


def _dist(a, b) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def _ear(face, outer, inner, top_a, top_b, bot_a, bot_b) -> float:
    vertical   = _dist(face[top_a], face[bot_a]) + _dist(face[top_b], face[bot_b])
    horizontal = 2.0 * _dist(face[outer], face[inner])
    return vertical / horizontal if horizontal > 0 else 0.0


# --------------------------------------------------------------------------

class BlinkFrequency(BaseGesture):

    def __init__(self, thresholds: dict):
        super().__init__("blink_frequency", thresholds)

        self._ear_threshold = thresholds.get("ear_threshold", 0.20)

        # Blink state machine
        self._in_blink    = False
        self._blink_start = 0

        # Rolling blink counter
        self._blink_frames: deque = deque()   # valid-frame numbers of confirmed blinks
        self._frame_count       = 0   # total frames (for update() bookkeeping)
        self._valid_frame_count = 0   # frames where face was visible and tracked

        # Last computed metrics (for state exposure)
        self._ear_r = 0.0
        self._ear_l = 0.0
        self._bpm   = 0.0

    # ------------------------------------------------------------------
    def update(self, landmarks: dict) -> bool:
        self._frame_count += 1
        self._run_detection(landmarks)
        return False     # metric only — never triggers

    # ------------------------------------------------------------------
    def detect(self, landmarks: dict) -> bool:
        """Returns True when current EAR is below the blink threshold."""
        face = landmarks.get("face")
        if face is None or len(face) <= max(_REQUIRED):
            return False
        ear_avg = (_ear(face, *_R_EYE) + _ear(face, *_L_EYE)) / 2.0
        return ear_avg < self._ear_threshold

    # ------------------------------------------------------------------
    @property
    def state(self) -> dict:
        max_bpm = self.thresholds.get("max_bpm_display", 40.0)
        return {
            # BaseGesture-compatible fields (dummies — never fires)
            "positive_frames":    0,
            "window_total":       1,
            "window_size":        1,
            "ratio":              min(1.0, self._bpm / max_bpm) if max_bpm > 0 else 0.0,
            "required_ratio":     1.0,
            "cooldown_remaining": 0,
            "in_cooldown":        False,
            # Blink-specific fields
            "ear_r":              self._ear_r,
            "ear_l":              self._ear_l,
            "ear_threshold":      self._ear_threshold,
            "blink_rate":         self._bpm,
            "blink_count":        len(self._blink_frames),
            "in_blink":           self._in_blink,
        }

    # ------------------------------------------------------------------
    def _run_detection(self, landmarks: dict) -> None:
        face = landmarks.get("face")
        if face is None or len(face) <= max(_REQUIRED):
            # Face lost — reset blink state so a half-blink is not counted later
            self._in_blink = False
            return

        detect_min = self.thresholds.get("detect_min_ear", 0.10)
        ear_r = _ear(face, *_R_EYE)
        ear_l = _ear(face, *_L_EYE)

        # Skip frame if tracking has collapsed (landmark positions invalid).
        # Do NOT reset _in_blink here: a hard blink can legitimately drop the EAR
        # below detect_min. Only a full face loss (handled above) should abort a
        # blink in progress.
        if ear_r < detect_min or ear_l < detect_min:
            return

        # Frame is valid: advance the visible-frame counter
        self._valid_frame_count += 1
        self._ear_r = ear_r
        self._ear_l = ear_l
        ear_avg = (ear_r + ear_l) / 2.0

        # ---- blink state machine (counts in valid frames) -----------
        thr        = self._ear_threshold
        min_frames = self.thresholds.get("blink_min_frames", 2)
        max_frames = self.thresholds.get("blink_max_frames", 12)

        if not self._in_blink:
            if ear_avg < thr:
                self._in_blink    = True
                self._blink_start = self._valid_frame_count
        else:
            duration = self._valid_frame_count - self._blink_start
            if ear_avg >= thr:
                if min_frames <= duration <= max_frames:
                    self._blink_frames.append(self._valid_frame_count)
                self._in_blink = False
            elif duration > max_frames:
                # Voluntary eye closure — discard
                self._in_blink = False

        # ---- rolling BPM (based on visible frames only) -------------
        window = self.thresholds.get("rate_window_frames", 900)
        fps    = self.thresholds.get("fps", 30.0)
        cutoff = self._valid_frame_count - window
        while self._blink_frames and self._blink_frames[0] < cutoff:
            self._blink_frames.popleft()
        # Extrapolate when the window isn't full yet (avoids artificially low BPM at start)
        elapsed   = min(self._valid_frame_count, window)
        self._bpm = len(self._blink_frames) * (fps * 60.0 / elapsed)
