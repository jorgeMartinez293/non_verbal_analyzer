from abc import ABC, abstractmethod
from collections import deque


class BaseGesture(ABC):
    """
    Abstract base class for all gesture detectors.

    Confirmation uses a sliding window instead of a strict consecutive-frame
    counter.  Within the last `confirmation_window` frames, at least
    `confirmation_ratio` of them must return True from detect() for the
    gesture to be confirmed.  This tolerates brief single-frame noise without
    resetting the entire accumulation.

    Thresholds read from thresholds.json (gesture sub-dict):
        confirmation_window  (int,   default 20)  — sliding window length
        confirmation_ratio   (float, default 0.70) — fraction required
        cooldown_frames      (int,   default 300)  — frames to skip after trigger
    """

    def __init__(self, name: str, thresholds: dict):
        self.name       = name
        self.thresholds = thresholds

        window_size   = thresholds.get("confirmation_window", 20)
        self._window  = deque(maxlen=window_size)
        self._cooldown_remaining: int = 0

    # ------------------------------------------------------------------
    @abstractmethod
    def detect(self, landmarks: dict) -> bool:
        """
        Return True if the gesture is present in this single frame.

        landmarks keys (present only when the model found something):
            'pose'       -> list[NormalizedLandmark]  (33 points)
            'face'       -> list[NormalizedLandmark]  (478 points)
            'left_hand'  -> list[NormalizedLandmark]  (21 points)
            'right_hand' -> list[NormalizedLandmark]  (21 points)
        """
        ...

    # ------------------------------------------------------------------
    @property
    def state(self) -> dict:
        """Current internal state exposed for debug overlays."""
        window_size = self.thresholds.get("confirmation_window", 20)
        filled      = sum(self._window)
        total       = len(self._window)
        return {
            "positive_frames":    filled,
            "window_total":       total,
            "window_size":        window_size,
            "ratio":              filled / total if total else 0.0,
            "required_ratio":     self.thresholds.get("confirmation_ratio", 0.70),
            "cooldown_remaining": self._cooldown_remaining,
            "in_cooldown":        self._cooldown_remaining > 0,
        }

    # ------------------------------------------------------------------
    def update(self, landmarks: dict) -> bool:
        """
        Called once per frame.  Returns True exactly once when the sliding
        window reaches the required positive ratio and the cooldown has
        expired.
        """
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            self._window.clear()
            return False

        self._window.append(1 if self.detect(landmarks) else 0)

        window_size = self.thresholds.get("confirmation_window", 20)
        min_ratio   = self.thresholds.get("confirmation_ratio", 0.70)

        if (
            len(self._window) >= window_size
            and sum(self._window) / len(self._window) >= min_ratio
        ):
            self._cooldown_remaining = self.thresholds.get("cooldown_frames", 300)
            self._window.clear()
            return True

        return False
