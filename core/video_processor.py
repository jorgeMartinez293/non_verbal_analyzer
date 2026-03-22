"""
Video Processor

Two operating modes selected at construction time:

  Normal mode  (debug=False)
      Runs gesture detection on every frame.  When a gesture is confirmed
      a ~20-frame clip (pre + post trigger) is saved via ClipSaver.
      Frames are stored raw — no annotation overhead.

  Debug mode   (debug=True)
      Runs gesture detection AND draws landmarks + gesture state on every
      frame, then writes the full annotated video to:
          <output_dir>/<video_stem>_annotated.mp4
      No clips are saved in this mode.
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

_ALERT_FRAMES = 45   # how long the DETECTED banner stays visible


class VideoProcessor:

    def __init__(
        self,
        gesture_manager: GestureManager,
        config: dict,
        clip_saver: ClipSaver | None = None,
        output_dir: str = "output",
        debug: bool = False,
    ):
        self.gesture_manager = gesture_manager
        self.config          = config
        self.clip_saver      = clip_saver
        self.output_dir      = Path(output_dir)
        self.debug           = debug

        frames_before    = config.get("clip", {}).get("frames_before", 10)
        self._buffer     = deque(maxlen=frames_before)
        self._models_dir = Path(__file__).parent.parent / "models"
        self._pose_lm = self._face_lm = self._hand_lm = None
        self._last_face_lms = None   # cache for debug inset continuity

    # ------------------------------------------------------------------
    def _build_landmarkers(self):
        BaseOptions = mp_tasks.BaseOptions
        RunningMode = VisionTaskRunningMode

        self._pose_lm = mp_vision.PoseLandmarker.create_from_options(
            mp_vision.PoseLandmarkerOptions(
                base_options = BaseOptions(
                    model_asset_path=str(self._models_dir / "pose_landmarker_heavy.task")
                ),
                running_mode                  = RunningMode.VIDEO,
                num_poses                     = 1,
                min_pose_detection_confidence = 0.6,
                min_pose_presence_confidence  = 0.6,
                min_tracking_confidence       = 0.6,
                output_segmentation_masks     = False,
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

        print(f"\n[VideoProcessor] Input  : {video_path}")
        print(f"[VideoProcessor] Mode   : {'DEBUG (annotated video)' if self.debug else 'NORMAL (clips)'}")
        print(f"[VideoProcessor] Size   : {width}x{height}  |  FPS: {fps:.2f}  |  Frames: {total_frames}\n")

        # Debug mode: open a VideoWriter for the full annotated output
        writer = None
        if self.debug:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            out_path = self.output_dir / f"{video_stem}_annotated.mp4"
            writer   = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps, (width, height),
            )
            print(f"[VideoProcessor] Output : {out_path}\n")

        self._build_landmarkers()

        crossed_arms_cfg = self.config.get("crossed_arms", {})
        alert_states: dict[str, int] = {g.name: 0 for g in self.gesture_manager.gestures}

        try:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp_ms = int(frame_idx * 1000 / fps)
                landmarks    = self._run_inference(frame, timestamp_ms)

                # ---- normal mode: feed post-trigger frames first -----
                if not self.debug and self.clip_saver:
                    self.clip_saver.add_post_frame(frame)

                # ---- gesture detection (updates internal state) ------
                triggered = self.gesture_manager.process_frame(landmarks)

                for name in triggered:
                    print(f"[VideoProcessor] ✓ '{name}' at frame {frame_idx}")
                    if self.debug:
                        alert_states[name] = _ALERT_FRAMES
                    elif self.clip_saver:
                        self.clip_saver.trigger(
                            gesture_name      = name,
                            pre_frames        = list(self._buffer),
                            trigger_frame_idx = frame_idx,
                            video_stem        = video_stem,
                            fps               = fps,
                            width             = width,
                            height            = height,
                        )

                # ---- normal mode: add frame to pre-trigger buffer ----
                if not self.debug:
                    self._buffer.append(frame.copy())

                # ---- debug mode: annotate and write frame ------------
                if self.debug:
                    if landmarks.get("face"):
                        self._last_face_lms = landmarks["face"]

                    debug_overlay.draw(frame, landmarks, crossed_arms_cfg)
                    debug_overlay.draw_face_inset(
                        frame, landmarks, cached_face_lms=self._last_face_lms
                    )
                    debug_overlay.draw_gesture_state(
                        frame,
                        self.gesture_manager.get_states(),
                        alert_states,
                        self.config,
                    )
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
            if writer:
                writer.release()
            self._pose_lm.close()
            self._face_lm.close()
            self._hand_lm.close()

        if not self.debug and self.clip_saver:
            self.clip_saver.flush()

        print(f"\n[VideoProcessor] Done. {frame_idx} frames processed.")

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
