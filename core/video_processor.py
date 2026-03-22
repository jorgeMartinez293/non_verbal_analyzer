"""
Video Processor — debug-viz branch

Processes every frame, draws landmarks + gesture state on each one,
and writes the full annotated video to:
    <output_dir>/<video_stem>_annotated.mp4

No clip saving in this branch — the goal is full-video visualisation
so gesture detection logic can be debugged frame by frame.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)

from core.gesture_manager import GestureManager
from core import debug_overlay

# How many frames to keep the DETECTED banner visible after a trigger
_ALERT_FRAMES = 45


class VideoProcessor:

    def __init__(
        self,
        gesture_manager: GestureManager,
        config: dict,
        output_dir: str = "output",
    ):
        self.gesture_manager = gesture_manager
        self.config          = config
        self.output_dir      = Path(output_dir)

        self._models_dir = Path(__file__).parent.parent / "models"
        self._pose_lm    = None
        self._face_lm    = None
        self._hand_lm    = None

    # ------------------------------------------------------------------
    def _build_landmarkers(self):
        BaseOptions = mp_tasks.BaseOptions
        RunningMode = VisionTaskRunningMode

        self._pose_lm = mp_vision.PoseLandmarker.create_from_options(
            mp_vision.PoseLandmarkerOptions(
                base_options = BaseOptions(
                    model_asset_path=str(self._models_dir / "pose_landmarker_heavy.task")
                ),
                running_mode                   = RunningMode.VIDEO,
                num_poses                      = 1,
                min_pose_detection_confidence  = 0.6,
                min_pose_presence_confidence   = 0.6,
                min_tracking_confidence        = 0.6,
                output_segmentation_masks      = False,
            )
        )
        self._face_lm = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options = BaseOptions(
                    model_asset_path=str(self._models_dir / "face_landmarker.task")
                ),
                running_mode                   = RunningMode.VIDEO,
                num_faces                      = 1,
                min_face_detection_confidence  = 0.6,
                min_face_presence_confidence   = 0.6,
                min_tracking_confidence        = 0.6,
                output_face_blendshapes        = False,
                output_facial_transformation_matrixes = False,
            )
        )
        self._hand_lm = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options = BaseOptions(
                    model_asset_path=str(self._models_dir / "hand_landmarker.task")
                ),
                running_mode                  = RunningMode.VIDEO,
                num_hands                     = 2,
                min_hand_detection_confidence = 0.6,
                min_hand_presence_confidence  = 0.6,
                min_tracking_confidence       = 0.6,
            )
        )

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

        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / f"{video_stem}_annotated.mp4"
        writer   = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        print(f"\n[VideoProcessor] Input  : {video_path}")
        print(f"[VideoProcessor] Output : {out_path}")
        print(f"[VideoProcessor] Size   : {width}x{height}  |  FPS: {fps:.2f}  |  Frames: {total_frames}\n")

        self._build_landmarkers()

        # alert_states tracks how many frames to keep each gesture's banner visible
        alert_states: dict[str, int] = {
            g.name: 0 for g in self.gesture_manager.gestures
        }

        crossed_arms_cfg = self.config.get("crossed_arms", {})

        try:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # --- inference ----------------------------------------
                timestamp_ms = int(frame_idx * 1000 / fps)
                landmarks    = self._run_inference(frame, timestamp_ms)

                # --- gesture detection (updates internal state) -------
                triggered = self.gesture_manager.process_frame(landmarks)
                for name in triggered:
                    alert_states[name] = _ALERT_FRAMES
                    print(f"[VideoProcessor] ✓ '{name}' at frame {frame_idx}")

                # --- draw landmarks -----------------------------------
                debug_overlay.draw(frame, landmarks, crossed_arms_cfg)

                # --- draw gesture state panel -------------------------
                debug_overlay.draw_gesture_state(
                    frame,
                    self.gesture_manager.get_states(),
                    alert_states,
                    self.config,
                )

                # --- tick down alert counters -------------------------
                for name in alert_states:
                    if alert_states[name] > 0:
                        alert_states[name] -= 1

                writer.write(frame)
                frame_idx += 1

                if frame_idx % 100 == 0:
                    pct = frame_idx / total_frames * 100 if total_frames else 0
                    print(f"[VideoProcessor] {frame_idx}/{total_frames}  ({pct:.1f}%)")

        finally:
            cap.release()
            writer.release()
            self._pose_lm.close()
            self._face_lm.close()
            self._hand_lm.close()

        print(f"\n[VideoProcessor] Done. Annotated video saved to: {out_path}")

    # ------------------------------------------------------------------
    def _run_inference(self, bgr_frame, timestamp_ms: int) -> dict:
        rgb      = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        pose_result = self._pose_lm.detect_for_video(mp_image, timestamp_ms)
        face_result = self._face_lm.detect_for_video(mp_image, timestamp_ms)
        hand_result = self._hand_lm.detect_for_video(mp_image, timestamp_ms)

        landmarks: dict = {}

        if pose_result.pose_landmarks:
            landmarks["pose"] = pose_result.pose_landmarks[0]
        if face_result.face_landmarks:
            landmarks["face"] = face_result.face_landmarks[0]
        if hand_result.hand_landmarks:
            for hand_lms, handedness in zip(
                hand_result.hand_landmarks, hand_result.handedness
            ):
                label = handedness[0].category_name
                landmarks["left_hand" if label == "Left" else "right_hand"] = hand_lms

        return landmarks
