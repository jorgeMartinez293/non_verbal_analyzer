"""
Video Processor  (MediaPipe Tasks API — mediapipe >= 0.10)

Runs three separate landmarkers in VIDEO mode:
  • PoseLandmarker   (heavy model) — 33 pose landmarks + 3-D world landmarks
  • FaceLandmarker                 — 478-point face mesh
  • HandLandmarker                 — 21 landmarks per hand, up to 2 hands

Landmarks are packaged into a plain dict and forwarded to GestureManager.

Frame processing order
----------------------
  a) Read BGR frame from VideoCapture
  b) clip_saver.add_post_frame(frame) — feeds any pending clips
  c) Run all three landmarkers on the frame
  d) Append frame to rolling pre-trigger buffer
  e) Ask GestureManager if any gesture fired
  f) For each fired gesture → clip_saver.trigger(pre_frames=list(buffer))

Model files
-----------
Expected in the models/ directory next to this package root:
  models/pose_landmarker_heavy.task
  models/face_landmarker.task
  models/hand_landmarker.task
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)

from core.gesture_manager import GestureManager
from core.clip_saver import ClipSaver
from core import debug_overlay


class VideoProcessor:

    def __init__(
        self,
        gesture_manager: GestureManager,
        clip_saver: ClipSaver,
        config: dict,
        debug: bool = False,
    ):
        self.gesture_manager = gesture_manager
        self.clip_saver      = clip_saver
        self.config          = config
        self.debug           = debug

        frames_before = config.get("clip", {}).get("frames_before", 10)
        self._buffer  = deque(maxlen=frames_before)

        self._models_dir = Path(__file__).parent.parent / "models"
        self._pose_lm    = None
        self._face_lm    = None
        self._hand_lm    = None

    # ------------------------------------------------------------------
    def _build_landmarkers(self, fps: float):
        """Instantiate the three landmarkers in VIDEO running mode."""
        BaseOptions  = mp_tasks.BaseOptions
        RunningMode  = VisionTaskRunningMode

        # ---- Pose (heavy = most accurate) ----------------------------
        pose_opts = mp_vision.PoseLandmarkerOptions(
            base_options     = BaseOptions(
                model_asset_path = str(self._models_dir / "pose_landmarker_heavy.task")
            ),
            running_mode         = RunningMode.VIDEO,
            num_poses            = 1,
            min_pose_detection_confidence  = 0.6,
            min_pose_presence_confidence   = 0.6,
            min_tracking_confidence        = 0.6,
            output_segmentation_masks      = False,
        )
        self._pose_lm = mp_vision.PoseLandmarker.create_from_options(pose_opts)

        # ---- Face ----------------------------------------------------
        face_opts = mp_vision.FaceLandmarkerOptions(
            base_options = BaseOptions(
                model_asset_path = str(self._models_dir / "face_landmarker.task")
            ),
            running_mode                   = RunningMode.VIDEO,
            num_faces                      = 1,
            min_face_detection_confidence  = 0.6,
            min_face_presence_confidence   = 0.6,
            min_tracking_confidence        = 0.6,
            output_face_blendshapes        = False,
            output_facial_transformation_matrixes = False,
        )
        self._face_lm = mp_vision.FaceLandmarker.create_from_options(face_opts)

        # ---- Hands ---------------------------------------------------
        hand_opts = mp_vision.HandLandmarkerOptions(
            base_options = BaseOptions(
                model_asset_path = str(self._models_dir / "hand_landmarker.task")
            ),
            running_mode                    = RunningMode.VIDEO,
            num_hands                       = 2,
            min_hand_detection_confidence   = 0.6,
            min_hand_presence_confidence    = 0.6,
            min_tracking_confidence         = 0.6,
        )
        self._hand_lm = mp_vision.HandLandmarker.create_from_options(hand_opts)

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

        self._build_landmarkers(fps)

        crossed_arms_thresholds = self.config.get("crossed_arms", {})

        try:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # (b) run inference first so we can annotate before buffering
                timestamp_ms = int(frame_idx * 1000 / fps)
                landmarks    = self._run_inference(frame, timestamp_ms)

                # (c) annotate frame if debug mode is on
                display_frame = (
                    debug_overlay.draw(frame, landmarks, crossed_arms_thresholds)
                    if self.debug else frame
                )

                # (d) feed post-trigger frames to pending clips
                self.clip_saver.add_post_frame(display_frame)

                # (e) append to rolling buffer (trigger frame ends up last)
                self._buffer.append(display_frame.copy())

                # (f) gesture detection + clip trigger
                triggered = self.gesture_manager.process_frame(landmarks)
                for gesture_name in triggered:
                    print(
                        f"[VideoProcessor] ✓ Gesture '{gesture_name}' "
                        f"confirmed at frame {frame_idx}"
                    )
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
                    print(
                        f"[VideoProcessor] {frame_idx}/{total_frames} "
                        f"frames processed ({pct:.1f}%)"
                    )

        finally:
            cap.release()
            self._pose_lm.close()
            self._face_lm.close()
            self._hand_lm.close()

        self.clip_saver.flush()
        print(f"\n[VideoProcessor] Done. {frame_idx} frames processed.")

    # ------------------------------------------------------------------
    def _run_inference(self, bgr_frame, timestamp_ms: int) -> dict:
        """
        Run all three landmarkers and return a unified landmarks dict:
            'pose'       → list[NormalizedLandmark] (33)   or None
            'face'       → list[NormalizedLandmark] (478)  or None
            'left_hand'  → list[NormalizedLandmark] (21)   or None
            'right_hand' → list[NormalizedLandmark] (21)   or None
        """
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        pose_result = self._pose_lm.detect_for_video(mp_image, timestamp_ms)
        face_result = self._face_lm.detect_for_video(mp_image, timestamp_ms)
        hand_result = self._hand_lm.detect_for_video(mp_image, timestamp_ms)

        landmarks: dict = {}

        # Pose — take first detected person
        if pose_result.pose_landmarks:
            landmarks["pose"] = pose_result.pose_landmarks[0]

        # Face — take first detected face
        if face_result.face_landmarks:
            landmarks["face"] = face_result.face_landmarks[0]

        # Hands — split by handedness label ("Left" / "Right")
        # Note: MediaPipe reports handedness from the *camera's* perspective,
        # which is mirrored vs the subject; for pre-recorded video the raw
        # label is used as-is.  Gesture files that care about which hand is
        # which should inspect the 'handedness' key if needed.
        if hand_result.hand_landmarks:
            for hand_lms, handedness_list in zip(
                hand_result.hand_landmarks, hand_result.handedness
            ):
                label = handedness_list[0].category_name  # "Left" or "Right"
                key   = "left_hand" if label == "Left" else "right_hand"
                landmarks[key] = hand_lms

        return landmarks
