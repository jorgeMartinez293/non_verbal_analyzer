"""
Clip Saver

When a gesture is confirmed, the VideoProcessor calls trigger() with the
pre-trigger frames already buffered.  From that point, add_post_frame()
is called for every subsequent frame until the required number of
post-trigger frames have been collected, at which point the clip is
written to disk automatically.

flush() should be called at the end of the video to save any pending
clips that did not accumulate enough post-trigger frames (e.g. gesture
occurred near the end of the file).

Output path:
    <output_dir>/<video_stem>/<gesture_name>_frame<XXXXXX>.mp4
"""

import os
import time
from pathlib import Path
from dataclasses import dataclass, field
import cv2


@dataclass
class _PendingClip:
    gesture_name:    str
    pre_frames:      list           # frames already captured (includes trigger frame)
    post_frames:     list = field(default_factory=list)
    frames_remaining: int = 0
    trigger_frame_idx: int = 0
    video_stem:      str = ""
    fps:             float = 25.0
    width:           int = 640
    height:          int = 480


class ClipSaver:

    def __init__(self, frames_before: int, frames_after: int, output_dir: str):
        self.frames_before  = frames_before
        self.frames_after   = frames_after
        self.output_dir     = Path(output_dir)
        self._pending: list[_PendingClip] = []

    # ------------------------------------------------------------------
    def trigger(
        self,
        gesture_name: str,
        pre_frames: list,
        trigger_frame_idx: int,
        video_stem: str,
        fps: float,
        width: int,
        height: int,
    ) -> None:
        """
        Called by VideoProcessor when a gesture is confirmed.
        pre_frames should contain up to frames_before frames ending at
        (and including) the trigger frame.
        """
        clip = _PendingClip(
            gesture_name      = gesture_name,
            pre_frames        = list(pre_frames),
            frames_remaining  = self.frames_after,
            trigger_frame_idx = trigger_frame_idx,
            video_stem        = video_stem,
            fps               = fps,
            width             = width,
            height            = height,
        )
        self._pending.append(clip)
        print(
            f"[ClipSaver] Collecting post-trigger frames for '{gesture_name}' "
            f"(trigger frame {trigger_frame_idx})"
        )

    # ------------------------------------------------------------------
    def add_post_frame(self, frame) -> None:
        """
        Called by VideoProcessor for every frame AFTER a trigger has been
        registered.  Automatically saves the clip once enough frames are
        collected.
        """
        completed = []
        for clip in self._pending:
            if clip.frames_remaining > 0:
                clip.post_frames.append(frame.copy())
                clip.frames_remaining -= 1
                if clip.frames_remaining == 0:
                    completed.append(clip)

        for clip in completed:
            self._save(clip)
            self._pending.remove(clip)

    # ------------------------------------------------------------------
    def flush(self) -> None:
        """Save any clips that are still pending (e.g. video ended early)."""
        for clip in self._pending:
            print(
                f"[ClipSaver] Flushing incomplete clip for '{clip.gesture_name}' "
                f"(only {len(clip.post_frames)}/{self.frames_after} post-frames captured)"
            )
            self._save(clip)
        self._pending.clear()

    # ------------------------------------------------------------------
    def _save(self, clip: _PendingClip) -> None:
        all_frames = clip.pre_frames + clip.post_frames
        if not all_frames:
            return

        dest_dir = self.output_dir / clip.video_stem
        dest_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{clip.gesture_name}_frame{clip.trigger_frame_idx:06d}.mp4"
        out_path = dest_dir / filename

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(out_path), fourcc, clip.fps, (clip.width, clip.height)
        )

        for frame in all_frames:
            writer.write(frame)
        writer.release()

        print(
            f"[ClipSaver] Saved {len(all_frames)}-frame clip → {out_path}"
        )
