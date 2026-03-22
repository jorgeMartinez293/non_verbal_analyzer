from abc import ABC, abstractmethod


class BaseGesture(ABC):
    """
    Abstract base class for all gesture detectors.

    Each gesture file in this directory should define one class that inherits
    from BaseGesture. The gesture manager will discover and instantiate it
    automatically, passing the relevant section from thresholds.json.

    Subclasses must implement:
        detect(landmarks) -> bool
    """

    def __init__(self, name: str, thresholds: dict):
        self.name = name
        self.thresholds = thresholds
        self._active_frames: int = 0
        self._cooldown_remaining: int = 0

    @abstractmethod
    def detect(self, landmarks: dict) -> bool:
        """
        Examine the current frame's landmarks and return True if the gesture
        is present in this single frame.

        Args:
            landmarks: dict with optional keys:
                'pose'        -> list of pose NormalizedLandmark (33 points)
                'face'        -> list of face NormalizedLandmark (468 points)
                'left_hand'   -> list of hand NormalizedLandmark (21 points)
                'right_hand'  -> list of hand NormalizedLandmark (21 points)
        """
        ...

    @property
    def state(self) -> dict:
        """Current internal state, useful for debug overlays."""
        return {
            "active_frames":     self._active_frames,
            "cooldown_remaining": self._cooldown_remaining,
            "in_cooldown":       self._cooldown_remaining > 0,
        }

    def update(self, landmarks: dict) -> bool:
        """
        Called once per frame. Handles the confirmation buffer and cooldown
        logic so individual gesture classes only need to implement detect().

        Returns True exactly once when the gesture has been confirmed for
        `confirmation_frames` consecutive frames and the cooldown has expired.
        """
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            self._active_frames = 0
            return False

        if self.detect(landmarks):
            self._active_frames += 1
            confirmation = self.thresholds.get("confirmation_frames", 10)
            if self._active_frames >= confirmation:
                self._cooldown_remaining = self.thresholds.get("cooldown_frames", 90)
                self._active_frames = 0
                return True
        else:
            self._active_frames = 0

        return False
