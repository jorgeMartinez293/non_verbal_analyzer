"""
Debug Overlay

Two public functions:

  draw(frame, landmarks, thresholds)
      Draws the 8 pose landmarks used by CrossedArms on the frame.

  draw_gesture_state(frame, gesture_states, alert_states, thresholds)
      Draws the gesture status panel: confirmation progress bar,
      cooldown countdown, and a flashing DETECTED banner.
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

    # ---- wrist-to-wrist crossing line --------------------------------
    margin      = thresholds.get("wrist_cross_margin", 0.05)
    cross_value = lm["R_WRIST"].x - lm["L_WRIST"].x
    cross_met   = cross_value >= margin
    cv2.line(frame, _px(lm["L_WRIST"], w, h), _px(lm["R_WRIST"], w, h),
             _COL_MET if cross_met else _COL_FAIL, 3)

    # ---- wrist dots (coloured by height check) -----------------------
    for key in ("L_WRIST", "R_WRIST"):
        wrist = lm[key]
        ratio = (wrist.y - shoulder_y) / torso_h if torso_h > 0 else 0
        col   = _COL_MET if h_min <= ratio <= h_max else _COL_FAIL
        cv2.circle(frame, _px(wrist, w, h), 8, col,           -1)
        cv2.circle(frame, _px(wrist, w, h), 8, (255,255,255),  1)

    # ---- remaining landmark dots ------------------------------------
    for name, col in [
        ("L_SHOULDER", _COL_SHOULDER), ("R_SHOULDER", _COL_SHOULDER),
        ("L_ELBOW",    _COL_ELBOW),    ("R_ELBOW",    _COL_ELBOW),
        ("L_HIP",      _COL_HIP),      ("R_HIP",      _COL_HIP),
    ]:
        cv2.circle(frame, _px(lm[name], w, h), 6, col,          -1)
        cv2.circle(frame, _px(lm[name], w, h), 6, (255,255,255), 1)

    # ---- HUD values --------------------------------------------------
    lw_ratio = (lm["L_WRIST"].y - shoulder_y) / torso_h if torso_h > 0 else 0
    rw_ratio = (lm["R_WRIST"].y - shoulder_y) / torso_h if torso_h > 0 else 0
    hud_lines = [
        (f"R-L wrist: {cross_value:+.3f}  (need >= {margin:.3f})",
         (0,220,0) if cross_met else (0,80,220)),
        (f"L wrist height: {lw_ratio:.2f}  [{h_min:.1f}-{h_max:.1f}]",
         (0,220,0) if h_min <= lw_ratio <= h_max else (0,80,220)),
        (f"R wrist height: {rw_ratio:.2f}  [{h_min:.1f}-{h_max:.1f}]",
         (0,220,0) if h_min <= rw_ratio <= h_max else (0,80,220)),
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
