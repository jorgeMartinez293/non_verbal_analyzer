"""
Debug Overlay

Public functions:

  draw(frame, landmarks, thresholds)
      Draws the 8 pose landmarks used by CrossedArms on the frame.

  draw_gesture_state(frame, gesture_states, alert_states, thresholds)
      Draws the gesture status panel: confirmation progress bar,
      cooldown countdown, and a flashing DETECTED banner.

  draw_face_inset(frame, landmarks)
      Renders a zoomed face crop with all 478 face landmarks in the
      top-right corner of the frame.
"""

from __future__ import annotations
import cv2
import numpy as np

# MediaPipe Pose landmark indices used by CrossedArms
_IDX = {
    "L_SHOULDER": 11, "R_SHOULDER": 12,
    "L_ELBOW":    13, "R_ELBOW":    14,
    "L_WRIST":    15, "R_WRIST":    16,
    "L_HIP":      23, "R_HIP":      24,
}

_COL_SHOULDER = (255, 255,   0)
_COL_ELBOW    = (  0, 255, 255)
_COL_WRIST    = (255, 255, 255)
_COL_HIP      = (160, 160, 160)
_COL_BONE     = ( 80, 200,  80)
_COL_MET      = (  0, 220,   0)
_COL_FAIL     = (  0,   0, 220)
_COL_BAND     = (200, 120,   0)


def _px(lm, w: int, h: int) -> tuple[int, int]:
    return int(lm.x * w), int(lm.y * h)


# -----------------------------------------------------------------------
def draw(frame: np.ndarray, landmarks: dict, thresholds: dict) -> np.ndarray:
    """Draw the 8 pose keypoints and diagnostic lines onto *frame* (in-place)."""
    h, w = frame.shape[:2]
    pose = landmarks.get("pose")

    if pose is None:
        cv2.putText(frame, "no pose", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
        return frame

    try:
        lm = {name: pose[idx] for name, idx in _IDX.items()}
    except IndexError:
        return frame

    # ---- torso band --------------------------------------------------
    shoulder_y = (lm["L_SHOULDER"].y + lm["R_SHOULDER"].y) / 2.0
    hip_y      = (lm["L_HIP"].y     + lm["R_HIP"].y)      / 2.0
    torso_h    = hip_y - shoulder_y
    h_min      = thresholds.get("wrist_height_min_ratio", 0.0)
    h_max      = thresholds.get("wrist_height_max_ratio", 1.1)

    band_top    = int((shoulder_y + h_min * torso_h) * h)
    band_bottom = int((shoulder_y + h_max * torso_h) * h)
    band_left   = int(min(lm["L_SHOULDER"].x, lm["R_SHOULDER"].x) * w) - 20
    band_right  = int(max(lm["L_SHOULDER"].x, lm["R_SHOULDER"].x) * w) + 20

    overlay = frame.copy()
    cv2.rectangle(overlay, (band_left, band_top), (band_right, band_bottom),
                  _COL_BAND, -1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    cv2.rectangle(frame, (band_left, band_top), (band_right, band_bottom),
                  _COL_BAND, 1)

    # ---- arm skeleton ------------------------------------------------
    for side in ("L", "R"):
        cv2.line(frame, _px(lm[f"{side}_SHOULDER"], w, h),
                        _px(lm[f"{side}_ELBOW"],    w, h), _COL_BONE, 2)
        cv2.line(frame, _px(lm[f"{side}_ELBOW"],    w, h),
                        _px(lm[f"{side}_WRIST"],    w, h), _COL_BONE, 2)

    # ---- crossing ratios (new scale-invariant metric) ----------------
    shoulder_dist      = lm["L_SHOULDER"].x - lm["R_SHOULDER"].x  # >0 facing cam
    min_shoulder_ratio = thresholds.get("min_shoulder_torso_ratio", 0.40)
    cross_thresh       = thresholds.get("wrist_cross_ratio", 0.50)
    shoulder_torso_r   = shoulder_dist / torso_h if torso_h > 0 else 0.0

    if shoulder_dist > 0:
        right_ratio = (lm["R_WRIST"].x - lm["R_SHOULDER"].x) / shoulder_dist
        left_ratio  = (lm["L_SHOULDER"].x - lm["L_WRIST"].x) / shoulder_dist
    else:
        right_ratio = left_ratio = 0.0

    facing_ok   = shoulder_torso_r >= min_shoulder_ratio
    right_ok    = right_ratio >= cross_thresh
    left_ok     = left_ratio  >= cross_thresh

    # Wrist connector: green only when both ratios meet the threshold
    cv2.line(frame, _px(lm["L_WRIST"], w, h), _px(lm["R_WRIST"], w, h),
             _COL_MET if (right_ok and left_ok) else _COL_FAIL, 3)

    # ---- wrist dots (coloured by their own crossing ratio) ----------
    for key, ratio_val, ratio_ok in [
        ("R_WRIST", right_ratio, right_ok),
        ("L_WRIST", left_ratio,  left_ok),
    ]:
        col = _COL_MET if ratio_ok else _COL_FAIL
        cv2.circle(frame, _px(lm[key], w, h), 8, col,            -1)
        cv2.circle(frame, _px(lm[key], w, h), 8, (255, 255, 255), 1)

    # ---- remaining landmark dots ------------------------------------
    for name, col in [
        ("L_SHOULDER", _COL_SHOULDER), ("R_SHOULDER", _COL_SHOULDER),
        ("L_HIP",      _COL_HIP),      ("R_HIP",      _COL_HIP),
    ]:
        cv2.circle(frame, _px(lm[name], w, h), 6, col,           -1)
        cv2.circle(frame, _px(lm[name], w, h), 6, (255, 255, 255), 1)

    # ---- HUD values --------------------------------------------------
    lw_h = (lm["L_WRIST"].y - shoulder_y) / torso_h if torso_h > 0 else 0
    rw_h = (lm["R_WRIST"].y - shoulder_y) / torso_h if torso_h > 0 else 0
    hud_lines = [
        (f"shoulder/torso: {shoulder_torso_r:.2f}  (min {min_shoulder_ratio:.2f})",
         (0, 220, 0) if facing_ok else (0, 80, 220)),
        (f"R wrist ratio: {right_ratio:.2f}  (need >= {cross_thresh:.2f})",
         (0, 220, 0) if right_ok else (0, 80, 220)),
        (f"L wrist ratio: {left_ratio:.2f}  (need >= {cross_thresh:.2f})",
         (0, 220, 0) if left_ok else (0, 80, 220)),
        (f"L wrist height: {lw_h:.2f}  [{h_min:.1f}-{h_max:.1f}]",
         (0, 220, 0) if h_min <= lw_h <= h_max else (0, 80, 220)),
        (f"R wrist height: {rw_h:.2f}  [{h_min:.1f}-{h_max:.1f}]",
         (0, 220, 0) if h_min <= rw_h <= h_max else (0, 80, 220)),
    ]
    _put_hud(frame, hud_lines, w, h, y_offset=0)
    return frame


# -----------------------------------------------------------------------
def draw_gesture_state(
    frame:          np.ndarray,
    gesture_states: dict,          # {name: {active_frames, cooldown_remaining, ...}}
    alert_states:   dict,          # {name: frames_remaining}  for DETECTED banner
    thresholds:     dict,          # full config dict (not just crossed_arms sub-dict)
) -> np.ndarray:
    """
    Draws the gesture status panel at the bottom of the frame:
      • Yellow progress bar — confirmation frames accumulating
      • Blue bar           — cooldown countdown
      • Green flashing banner — "DETECTED" for ~1 second after trigger
    """
    h, w = frame.shape[:2]
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, w / 1400)
    thickness  = 1
    bar_h      = 18
    pad        = 6
    panel_x    = 0

    y_cursor = h  # build upwards from the bottom

    for gesture_name, state in gesture_states.items():
        cfg          = thresholds.get(gesture_name, {})
        cooldown_tot = cfg.get("cooldown_frames", 300)
        cooldown_rem = state["cooldown_remaining"]
        alert_rem    = alert_states.get(gesture_name, 0)

        # ---- DETECTED banner ----------------------------------------
        if alert_rem > 0:
            alpha   = min(1.0, alert_rem / 15)           # fade out
            banner_h = int(44 * font_scale / 0.45)
            y_cursor -= banner_h + pad
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, y_cursor), (w, y_cursor + banner_h),
                          (0, 140, 0), -1)
            cv2.addWeighted(overlay, alpha * 0.8, frame, 1 - alpha * 0.8, 0, frame)
            label = f"DETECTED: {gesture_name.replace('_',' ').upper()}"
            tw = cv2.getTextSize(label, font, font_scale * 1.4, 2)[0][0]
            cv2.putText(frame, label,
                        ((w - tw) // 2, y_cursor + banner_h - pad),
                        font, font_scale * 1.4, (255, 255, 255), 2, cv2.LINE_AA)

        # ---- Cooldown bar -------------------------------------------
        elif cooldown_rem > 0:
            y_cursor -= bar_h + pad
            ratio = cooldown_rem / cooldown_tot
            filled = int(w * ratio)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, y_cursor), (w, y_cursor + bar_h), (80, 60, 0), -1)
            cv2.rectangle(overlay, (0, y_cursor), (filled, y_cursor + bar_h), (200, 120, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            label = f"cooldown  {cooldown_rem} frames"
            cv2.putText(frame, label, (pad, y_cursor + bar_h - 4),
                        font, font_scale, (220, 180, 80), thickness, cv2.LINE_AA)

        # ---- Confirmation progress bar ------------------------------
        else:
            y_cursor -= bar_h + pad
            ratio        = state.get("ratio", 0.0)
            required     = state.get("required_ratio", 0.70)
            positive     = state.get("positive_frames", 0)
            window_total = state.get("window_total", 0)
            window_size  = state.get("window_size", 20)
            bar_fill     = int(w * ratio)
            threshold_x  = int(w * required)
            col_bar      = (0, 200, 200) if ratio >= required else (0, 130, 200)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, y_cursor), (w, y_cursor + bar_h), (30, 30, 0), -1)
            cv2.rectangle(overlay, (0, y_cursor), (bar_fill, y_cursor + bar_h), col_bar, -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            # threshold marker
            cv2.line(frame, (threshold_x, y_cursor), (threshold_x, y_cursor + bar_h),
                     (255, 255, 255), 1)
            label = (f"{gesture_name.replace('_',' ')}  "
                     f"{positive}/{window_total} frames  "
                     f"({ratio*100:.0f}% / need {required*100:.0f}%)")
            cv2.putText(frame, label, (pad, y_cursor + bar_h - 4),
                        font, font_scale, (200, 230, 230), thickness, cv2.LINE_AA)

    return frame


# -----------------------------------------------------------------------
def draw_face_inset(frame: np.ndarray, landmarks: dict) -> np.ndarray:
    """
    Crop the face region from *frame*, draw all detected face landmarks
    on the crop, and embed it as a picture-in-picture in the top-right
    corner.

    Bounding box is derived from the 478 face mesh landmarks when
    available, falling back to the 11 head pose landmarks (nose, eyes,
    ears, mouth corners) if the face landmarker produced no output.
    """
    h, w = frame.shape[:2]

    face_lms = landmarks.get("face")
    pose_lms = landmarks.get("pose")

    if face_lms is None and pose_lms is None:
        return frame

    # ---- compute face bounding box in pixel coords ------------------
    if face_lms:
        xs = [lm.x * w for lm in face_lms]
        ys = [lm.y * h for lm in face_lms]
    else:
        # Head landmarks: nose(0) … mouth_right(10)
        head = [pose_lms[i] for i in range(11) if pose_lms[i].visibility > 0.3]
        if not head:
            return frame
        xs = [lm.x * w for lm in head]
        ys = [lm.y * h for lm in head]

    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))

    # Add padding (~30 % of face size on each side)
    pad_x = max(10, int((x_max - x_min) * 0.35))
    pad_y = max(10, int((y_max - y_min) * 0.35))
    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(w, x_max + pad_x)
    y_max = min(h, y_max + pad_y)

    if x_max <= x_min or y_max <= y_min:
        return frame

    # ---- crop -------------------------------------------------------
    crop = frame[y_min:y_max, x_min:x_max].copy()
    ch, cw = crop.shape[:2]
    if cw <= 0 or ch <= 0:
        return frame

    # ---- scale inset to 1/3 of frame width, keep aspect ratio ------
    inset_w = max(1, w // 3)
    inset_h = max(1, int(inset_w * ch / cw))
    inset_h = min(inset_h, h // 2)
    inset   = cv2.resize(crop, (inset_w, inset_h), interpolation=cv2.INTER_LINEAR)

    # ---- draw landmarks AFTER resize so dots stay 1 px -------------
    # Drawing on the large crop before scaling turns 1-px dots into
    # sub-pixel artefacts that vanish during interpolation.
    if face_lms:
        scale_x = inset_w / (x_max - x_min)
        scale_y = inset_h / (y_max - y_min)
        for lm in face_lms:
            px = int((lm.x * w - x_min) * scale_x)
            py = int((lm.y * h - y_min) * scale_y)
            if 0 <= px < inset_w and 0 <= py < inset_h:
                cv2.circle(inset, (px, py), 1, (0, 230, 0), -1)

    # ---- border -----------------------------------------------------
    cv2.rectangle(inset, (0, 0), (inset_w - 1, inset_h - 1), (200, 200, 200), 2)

    # ---- place in top-right corner ----------------------------------
    margin = 10
    x_off  = w - inset_w - margin
    y_off  = margin

    if x_off >= 0 and y_off + inset_h <= h:
        frame[y_off: y_off + inset_h, x_off: x_off + inset_w] = inset

    return frame


# -----------------------------------------------------------------------
def _put_hud(
    frame: np.ndarray,
    lines: list,
    w: int,
    h: int,
    y_offset: int = 0,
) -> None:
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, w / 1600)
    thickness  = 1
    pad        = 6
    line_h     = int(18 * font_scale / 0.4)

    max_tw = max(
        cv2.getTextSize(l if isinstance(l, str) else l[0],
                        font, font_scale, thickness)[0][0]
        for l in lines
    )
    box_w = max_tw + pad * 2
    box_h = line_h * len(lines) + pad * 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y_offset), (box_w, y_offset + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, line in enumerate(lines):
        text  = line if isinstance(line, str) else line[0]
        color = (200, 200, 200) if isinstance(line, str) else line[1]
        y     = y_offset + pad + (i + 1) * line_h
        cv2.putText(frame, text, (pad, y), font, font_scale,
                    color, thickness, cv2.LINE_AA)
