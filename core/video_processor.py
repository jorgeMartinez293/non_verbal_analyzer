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
from types import SimpleNamespace

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
        self._pose_lm = None  # set by _build_landmarkers()
        self._face_lm = None  # set by _build_landmarkers()
        self._hand_lm = None  # set by _build_landmarkers()
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
                        # Copy coordinates into plain Python objects so they
                        # remain valid after MediaPipe recycles its internal
                        # C++ buffers on the next inference call.
                        self._last_face_lms = [
                            SimpleNamespace(x=lm.x, y=lm.y, z=lm.z)
                            for lm in landmarks["face"]
                        ]

                    clean_frame = frame.copy()  # snapshot before any drawing
                    debug_overlay.draw(frame, landmarks, crossed_arms_cfg)
                    debug_overlay.draw_face_inset(
                        frame, landmarks,
                        cached_face_lms=self._last_face_lms,
                        source_frame=clean_frame,
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
        h, w   = bgr_frame.shape[:2]
        rgb    = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        pose_result = self._pose_lm.detect_for_video(mp_img, timestamp_ms)
        hand_result = self._hand_lm.detect_for_video(mp_img, timestamp_ms)

        # ---- face: run on a zoomed-in crop around the head ----------
        # The face landmarker struggles with small faces (subject far
        # from camera). We derive a head bounding box from the already-
        # computed pose landmarks (stable in VIDEO mode), crop that
        # region, feed it to the face model, then remap the resulting
        # normalised coordinates back to the full-frame space.
        face_result, crop_bbox = self._face_inference_on_crop(
            bgr_frame, pose_result, timestamp_ms, h, w
        )

        landmarks: dict = {}

        if pose_result.pose_landmarks:
            landmarks["pose"] = pose_result.pose_landmarks[0]

        if face_result.face_landmarks:
            if crop_bbox:
                cx0, cy0, cx1, cy1 = crop_bbox
                cw, ch = cx1 - cx0, cy1 - cy0
                # Remap from crop-normalised → full-frame-normalised
                landmarks["face"] = [
                    SimpleNamespace(
                        x = (lm.x * cw + cx0) / w,
                        y = (lm.y * ch + cy0) / h,
                        z = lm.z,
                    )
                    for lm in face_result.face_landmarks[0]
                ]
            else:
                landmarks["face"] = [
                    SimpleNamespace(x=lm.x, y=lm.y, z=lm.z)
                    for lm in face_result.face_landmarks[0]
                ]

        if hand_result.hand_landmarks:
            for hand_lms, handedness in zip(
                hand_result.hand_landmarks, hand_result.handedness
            ):
                label = handedness[0].category_name
                landmarks["left_hand" if label == "Left" else "right_hand"] = hand_lms

        return landmarks

    # ------------------------------------------------------------------
    def _face_inference_on_crop(self, bgr_frame, pose_result, timestamp_ms, h, w):
        """
        Crop the head region from bgr_frame using pose head landmarks,
        run the face landmarker on the crop, and return the result plus
        the crop bounding box (x0, y0, x1, y1) in pixels.

        Falls back to the full frame if pose is unavailable.
        """
        crop_bbox = None

        if pose_result.pose_landmarks:
            head = [
                pose_result.pose_landmarks[0][i]
                for i in range(11)
                if pose_result.pose_landmarks[0][i].visibility > 0.3
            ]
            if head:
                xs = [lm.x * w for lm in head]
                ys = [lm.y * h for lm in head]
                face_w = max(xs) - min(xs)
                face_h = max(ys) - min(ys)
                pad_x  = int(face_w * 0.55)
                pad_y  = int(face_h * 0.75)
                x0 = max(0, int(min(xs)) - pad_x)
                y0 = max(0, int(min(ys)) - pad_y)
                x1 = min(w, int(max(xs)) + pad_x)
                y1 = min(h, int(max(ys)) + pad_y)
                if x1 > x0 and y1 > y0:
                    crop_bbox = (x0, y0, x1, y1)

        if crop_bbox:
            x0, y0, x1, y1 = crop_bbox
            crop     = bgr_frame[y0:y1, x0:x1]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mp_crop  = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
            result   = self._face_lm.detect_for_video(mp_crop, timestamp_ms)
        else:
            rgb    = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._face_lm.detect_for_video(mp_img, timestamp_ms)

        return result, crop_bbox
