"""
Video Processor

Drives the main analysis loop:
  1. Opens the MP4 with OpenCV.
  2. Runs every frame through MediaPipe Holistic (model_complexity=2 for
     maximum accuracy).
  3. Packages the resulting landmarks into a dict and passes them to the
     GestureManager.
  4. Maintains a rolling frame buffer (deque) of length `frames_before`
     for pre-trigger clip assembly.
  5. Coordinates with ClipSaver to collect post-trigger frames and write
     the final clips.

Frame processing order (per iteration)
---------------------------------------
  a) Read frame from VideoCapture
  b) Call clip_saver.add_post_frame(frame) → feeds any pending clips
  c) Run MediaPipe inference → extract landmarks
  d) Append frame to rolling buffer (so trigger frame is the last entry)
  e) Ask GestureManager whether any gesture fired
  f) For each fired gesture → clip_saver.trigger(pre_frames=list(buffer))
"""

import cv2
import mediapipe as mp
from collections import deque
from pathlib import Path

from core.gesture_manager import GestureManager
from core.clip_saver import ClipSaver


class VideoProcessor:

    def __init__(
        self,
        gesture_manager: GestureManager,
        clip_saver: ClipSaver,
        config: dict,
    ):
        self.gesture_manager = gesture_manager
        self.clip_saver      = clip_saver
        self.config          = config

        frames_before = config.get("clip", {}).get("frames_before", 10)
        self._buffer  = deque(maxlen=frames_before)

        self._mp_holistic  = mp.solutions.holistic
        self._mp_drawing   = mp.solutions.drawing_utils
        self._mp_draw_styles = mp.solutions.drawing_styles

    # ------------------------------------------------------------------
    def process(self, video_path: str) -> None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_stem   = Path(video_path).stem

        print(f"\n[VideoProcessor] File   : {video_path}")
        print(f"[VideoProcessor] Size   : {width}x{height}  |  FPS: {fps:.2f}  |  Frames: {total_frames}\n")

        holistic_cfg = dict(
            model_complexity          = 2,      # highest accuracy
            smooth_landmarks          = True,
            enable_segmentation       = False,
            smooth_segmentation       = False,
            refine_face_landmarks     = True,   # 478-point face mesh
            min_detection_confidence  = 0.6,
            min_tracking_confidence   = 0.6,
        )

        with self._mp_holistic.Holistic(**holistic_cfg) as holistic:
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # (b) feed post-trigger frames to any pending clips FIRST
                self.clip_saver.add_post_frame(frame)

                # (c) MediaPipe inference
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = holistic.process(rgb)
                rgb.flags.writeable = True

                landmarks = self._extract_landmarks(results)

                # (d) append to rolling buffer so trigger frame is last
                self._buffer.append(frame.copy())

                # (e) gesture detection
                triggered = self.gesture_manager.process_frame(landmarks)

                # (f) trigger clip collection
                for gesture_name in triggered:
                    print(f"[VideoProcessor] ✓ Gesture '{gesture_name}' confirmed at frame {frame_idx}")
                    self.clip_saver.trigger(
                        gesture_name      = gesture_name,
                        pre_frames        = list(self._buffer),
                        trigger_frame_idx = frame_idx,
                        video_stem        = video_stem,
                        fps               = fps,
                        width             = width,
                        height            = height,
                    )

                frame_idx += 1
                if frame_idx % 100 == 0:
                    pct = frame_idx / total_frames * 100 if total_frames else 0
                    print(f"[VideoProcessor] {frame_idx}/{total_frames} frames processed ({pct:.1f}%)")

        cap.release()
        self.clip_saver.flush()
        print(f"\n[VideoProcessor] Done. {frame_idx} frames processed.")

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_landmarks(results) -> dict:
        """
        Converts MediaPipe Holistic results into a plain dict so gesture
        classes stay decoupled from the MediaPipe API.

        Keys present only when the corresponding model produced output:
            'pose'       → list of 33  NormalizedLandmark  (x, y, z, visibility)
            'face'       → list of 478 NormalizedLandmark  (x, y, z)
            'left_hand'  → list of 21  NormalizedLandmark  (x, y, z)
            'right_hand' → list of 21  NormalizedLandmark  (x, y, z)
        """
        landmarks = {}

        if results.pose_landmarks:
            landmarks["pose"] = results.pose_landmarks.landmark

        if results.face_landmarks:
            landmarks["face"] = results.face_landmarks.landmark

        if results.left_hand_landmarks:
            landmarks["left_hand"] = results.left_hand_landmarks.landmark

        if results.right_hand_landmarks:
            landmarks["right_hand"] = results.right_hand_landmarks.landmark

        return landmarks
