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
import math
import numpy as np
from collections import deque

# ---- Rolling history for temporal graphs --------------------------------
_HISTORY_LEN = 200
_GRAPH_W     = 130   # width of graph plot area
_GRAPH_H     = 35    # height of graph plot area
_GRAPH_GAP   = 4     # vertical gap between graphs
_LABEL_H     = 12    # height of label row above each graph
_VAL_W       = 75    # extra width to the right for current-value text
_history: dict[str, deque] = {}
_scale:   dict[str, float] = {}   # hi per metric — fixed at thr*1.3; expands up for refs

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


def _dist_lm(a, b) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


# -----------------------------------------------------------------------
def draw(frame: np.ndarray, landmarks: dict, thresholds: dict,
         gesture_states: dict | None = None) -> np.ndarray:
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
    crossed_cfg = thresholds.get("crossed_arms", thresholds)  # full config or sub-dict
    h_min      = crossed_cfg.get("wrist_height_min_ratio", 0.0)
    h_max      = crossed_cfg.get("wrist_height_max_ratio", 1.1)

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

    # ---- crossing ratios (arm-length normalised) ---------------------
    cross_thresh = crossed_cfg.get("wrist_cross_ratio", 0.40)

    arm_len_r = (_dist_lm(lm["R_SHOULDER"], lm["R_ELBOW"]) +
                 _dist_lm(lm["R_ELBOW"],    lm["R_WRIST"]))
    arm_len_l = (_dist_lm(lm["L_SHOULDER"], lm["L_ELBOW"]) +
                 _dist_lm(lm["L_ELBOW"],    lm["L_WRIST"]))

    if arm_len_r > 0 and arm_len_l > 0:
        right_ratio = (lm["R_WRIST"].x - lm["R_SHOULDER"].x) / arm_len_r
        left_ratio  = (lm["L_SHOULDER"].x - lm["L_WRIST"].x) / arm_len_l
    else:
        right_ratio = left_ratio = 0.0

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

    # ---- touch_face: nose anchor + threshold radius + hand landmarks ---
    tf_cfg    = thresholds.get("touch_face", thresholds)
    tf_ratio  = tf_cfg.get("hand_face_ratio", 0.35)
    nose_lm   = pose[0]
    nx, ny    = _px(nose_lm, w, h)

    # threshold radius in pixels (ratio * shoulder_dist_px)
    s_dist_px = int(_dist_lm(lm["L_SHOULDER"], lm["R_SHOULDER"]) * w)
    radius_px = max(1, int(tf_ratio * s_dist_px))

    # semi-transparent orange circle showing the proximity zone
    overlay2 = frame.copy()
    cv2.circle(overlay2, (nx, ny), radius_px, (0, 165, 255), 2)
    cv2.addWeighted(overlay2, 0.5, frame, 0.5, 0, frame)

    # nose dot
    cv2.circle(frame, (nx, ny), 6, (0, 200, 255), -1)
    cv2.circle(frame, (nx, ny), 6, (255, 255, 255), 1)

    # hand landmarks
    s_dist_norm = _dist_lm(lm["L_SHOULDER"], lm["R_SHOULDER"])
    for hand_key in ("left_hand", "right_hand"):
        hand = landmarks.get(hand_key)
        if hand is None:
            continue
        hand_active = any(
            _dist_lm(lm2, nose_lm) / max(s_dist_norm, 1e-4) < tf_ratio
            for lm2 in hand
        )
        col = (0, 230, 0) if hand_active else (200, 200, 200)
        for lm2 in hand:
            px2, py2 = int(lm2.x * w), int(lm2.y * h)
            cv2.circle(frame, (px2, py2), 4, col, -1)

    _update_and_draw_graphs(frame, gesture_states, thresholds, landmarks)
    return frame


# -----------------------------------------------------------------------
def _update_and_draw_graphs(
    frame: np.ndarray,
    gesture_states: dict | None,
    thresholds: dict,
    landmarks: dict,
) -> None:
    """Update rolling metric history and render mini time-series graphs."""
    ca_state  = (gesture_states or {}).get("crossed_arms", {})
    oa_state  = (gesture_states or {}).get("open_arms", {})
    re_state  = (gesture_states or {}).get("raised_eyebrows", {})
    bf_state  = (gesture_states or {}).get("blink_frequency", {})
    tf_state  = (gesture_states or {}).get("touch_face", {})

    cross_thr = thresholds.get("crossed_arms", {}).get("wrist_cross_ratio", 0.50)
    open_thr  = thresholds.get("open_arms", {}).get("min_open_ratio", 2.0)
    calib     = re_state.get("calibrating", False)
    brow_thr  = re_state.get("personalized_thr") if not calib else None

    ear_thr   = thresholds.get("blink_frequency", {}).get("ear_threshold", 0.20)
    tf_thr    = thresholds.get("touch_face", {}).get("hand_face_ratio", 0.35)

    shoulder_eu = oa_state.get("shoulder_dist")
    if shoulder_eu is None:
        pose = landmarks.get("pose")
        if pose:
            try:
                ls = pose[11]; rs = pose[12]
                shoulder_eu = math.sqrt((ls.x - rs.x) ** 2 + (ls.y - rs.y) ** 2)
            except (IndexError, AttributeError):
                pass

    inter_iris = re_state.get("inter_iris")
    if inter_iris is None:
        face = landmarks.get("face")
        if face and len(face) > 473:
            v = abs(face[468].x - face[473].x)
            inter_iris = v if v > 0 else None

    # (key, value, threshold, is_ref, calibrating)
    specs = [
        ("ls-rs", shoulder_eu,                None,      True,  False),
        ("rw-ls", ca_state.get("right_ratio"), cross_thr, False, False),
        ("lw-rs", ca_state.get("left_ratio"),  cross_thr, False, False),
        ("lw-rw", oa_state.get("open_ratio"),  open_thr,  False, False),
        ("iris",  inter_iris,                  None,      True,  False),
        ("rb-ri", re_state.get("ratio_r"),     brow_thr,  False, calib),
        ("lb-li", re_state.get("ratio_l"),     brow_thr,  False, calib),
        ("ear-r", bf_state.get("ear_r"),       ear_thr,   False, False),
        ("ear-l", bf_state.get("ear_l"),       ear_thr,   False, False),
        ("bpm",   bf_state.get("blink_rate"),  None,      True,  False),
        ("tf-hf", tf_state.get("hand_face_ratio"), tf_thr, False, False),
    ]

    for key, val, *_ in specs:
        if key not in _history:
            _history[key] = deque(maxlen=_HISTORY_LEN)
        if val is not None:
            _history[key].append(float(val))

    y = 0
    for key, _val, thr, is_ref, is_calib in specs:
        buf = _history.get(key, deque())
        _draw_one_graph(frame, 0, y, key, buf, thr, is_ref, is_calib)
        y += _LABEL_H + _GRAPH_H + _GRAPH_GAP

    # ---- horizontal confirmation bars below the 7 graphs ------------
    if not gesture_states:
        return
    _ROW_H   = _LABEL_H + _GRAPH_H + _GRAPH_GAP   # 51px
    panel_w  = _GRAPH_W + _VAL_W                   # 205px
    n_gest   = len(gesture_states)
    bar_gap  = 5
    bar_w    = (panel_w - bar_gap * (n_gest - 1)) // max(n_gest, 1)
    bar_h    = 40
    bar_y    = 11 * _ROW_H + bar_gap               # just below the 11th graph

    for i, (g_name, g_state) in enumerate(gesture_states.items()):
        bx = i * (bar_w + bar_gap)
        by = bar_y

        # background
        ov = frame.copy()
        cv2.rectangle(ov, (bx, by), (bx + bar_w, by + bar_h), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

        required     = g_state.get("required_ratio", 0.70)
        calib        = g_state.get("calibrating", False)
        cooldown_rem = g_state.get("cooldown_remaining", 0)
        cooldown_tot = max(1, thresholds.get(g_name, {}).get("cooldown_frames", 50))

        if cooldown_rem > 0:
            # drain slowly as cooldown expires
            fill_ratio = cooldown_rem / cooldown_tot
            col        = (0, 140, 0)
        elif calib:
            calib_n   = g_state.get("calib_samples", 0)
            calib_tot = max(1, g_state.get("calib_target", 200))
            fill_ratio = calib_n / calib_tot
            col        = (200, 200, 200)
        else:
            fill_ratio = g_state.get("ratio", 0.0)
            col        = (0, 200, 0)

        fill_w = int(fill_ratio * bar_w)
        if fill_w > 0:
            cv2.rectangle(frame, (bx, by), (bx + fill_w, by + bar_h), col, -1)

        # white threshold marker
        thr_x = bx + int(required * bar_w)
        thr_x = max(bx, min(bx + bar_w - 1, thr_x))
        cv2.line(frame, (thr_x, by), (thr_x, by + bar_h), (255, 255, 255), 1)

        # label
        short = {"crossed_arms": "ca", "open_arms": "oa", "raised_eyebrows": "re", "blink_frequency": "bf", "touch_face": "tf"}.get(g_name) or g_name[:2]
        cv2.putText(frame, short, (bx + 3, by + bar_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)


# -----------------------------------------------------------------------
def _draw_one_graph(
    frame: np.ndarray,
    x0: int, y0: int,
    label: str,
    buf,               # deque[float]
    threshold,         # float | None
    is_ref: bool,
    calibrating: bool,
) -> None:
    total_w = _GRAPH_W + _VAL_W
    total_h = _LABEL_H + _GRAPH_H

    # semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + total_w, y0 + total_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # label
    cv2.putText(frame, label, (x0 + 3, y0 + _LABEL_H - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1, cv2.LINE_AA)

    if not buf:
        return

    y_vals  = list(buf)
    current = y_vals[-1]
    n       = len(y_vals)

    # y-range: lo always 0 (graph clips below); hi fixed at thr*1.3 for gesture
    # metrics, or expands up for reference metrics. Numeric text is unclamped.
    if threshold is not None:
        _scale[label] = threshold * 1.3
    elif label not in _scale:
        _scale[label] = max(current, 1e-4)
    else:
        _scale[label] = max(_scale[label], current)
    lo = 0.0
    hi = max(_scale[label], 1e-4)

    def _to_py(v: float) -> int:
        t = max(0.0, min(1.0, (v - lo) / (hi - lo)))
        return y0 + _LABEL_H + _GRAPH_H - int(t * _GRAPH_H)

    # line color
    if calibrating:
        color = (255, 255, 255)
    elif is_ref:
        t = max(0.0, min(1.0, (current - lo) / (hi - lo)))
        color = (0, int(80 + 140 * t), int(220 * (1 - t)))
    else:
        color = (0, 220, 0) if (threshold is not None and current >= threshold) else (0, 80, 220)

    # polyline
    pts = np.array(
        [[x0 + int(i / max(n - 1, 1) * (_GRAPH_W - 1)), _to_py(v)]
         for i, v in enumerate(y_vals)],
        dtype=np.int32,
    ).reshape(-1, 1, 2)
    cv2.polylines(frame, [pts], False, color, 1, cv2.LINE_AA)

    # dashed threshold line
    if threshold is not None and not calibrating:
        y_thr = max(y0 + _LABEL_H, min(y0 + _LABEL_H + _GRAPH_H, _to_py(threshold)))
        dash  = 5
        for xd in range(x0, x0 + _GRAPH_W, dash * 2):
            cv2.line(frame, (xd, y_thr), (min(xd + dash, x0 + _GRAPH_W), y_thr),
                     (180, 180, 60), 1)

    # current value / threshold
    val_text = (f"{current:.2f}/{threshold:.2f}" if threshold is not None
                else f"{current:.2f}")
    cv2.putText(frame, val_text,
                (x0 + _GRAPH_W + 3, y0 + _LABEL_H + _GRAPH_H - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)


# -----------------------------------------------------------------------
def draw_gesture_state(
    frame:          np.ndarray,
    gesture_states: dict,
    alert_states:   dict,
    thresholds:     dict,
) -> np.ndarray:
    """Draws the DETECTED banner at the bottom when a gesture fires."""
    h, w = frame.shape[:2]
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, w / 1400)
    pad        = 6
    y_cursor   = h

    for gesture_name in gesture_states:
        alert_rem = alert_states.get(gesture_name, 0)
        if alert_rem > 0:
            alpha    = min(1.0, alert_rem / 15)
            banner_h = int(44 * font_scale / 0.45)
            y_cursor -= banner_h + pad
            overlay  = frame.copy()
            cv2.rectangle(overlay, (0, y_cursor), (w, y_cursor + banner_h), (0, 140, 0), -1)
            cv2.addWeighted(overlay, alpha * 0.8, frame, 1 - alpha * 0.8, 0, frame)
            label = f"DETECTED: {gesture_name.replace('_', ' ').upper()}"
            tw = cv2.getTextSize(label, font, font_scale * 1.4, 2)[0][0]
            cv2.putText(frame, label, ((w - tw) // 2, y_cursor + banner_h - pad),
                        font, font_scale * 1.4, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


# -----------------------------------------------------------------------
def draw_face_inset(
    frame: np.ndarray,
    landmarks: dict,
    cached_face_lms=None,
    source_frame: np.ndarray | None = None,
) -> np.ndarray:
    """
    Crop the face region and embed it as a picture-in-picture in the
    top-right corner of *frame*.

    source_frame: the clean (un-annotated) frame to crop from.  If None,
                  falls back to *frame* itself (legacy behaviour).

    Bounding box is ALWAYS derived from pose head landmarks (indices 0–10)
    so the window keeps following the face even when the face landmarker
    loses tracking.

    Dots are drawn from the current face landmarks when available, or from
    cached_face_lms when the face landmarker has dropped out temporarily.
    """
    h, w = frame.shape[:2]

    pose_lms = landmarks.get("pose")
    face_lms = landmarks.get("face") or cached_face_lms

    if pose_lms is None:
        return frame

    # ---- bounding box always from current pose head landmarks -------
    # Pose tracking in VIDEO mode is much more stable than face tracking.
    # Using it for the bbox guarantees the window keeps moving with the
    # subject regardless of face landmarker state.
    head = [pose_lms[i] for i in range(11) if pose_lms[i].visibility > 0.3]
    if not head:
        return frame
    xs = [lm.x * w for lm in head]
    ys = [lm.y * h for lm in head]

    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))

    # Add padding: more vertical room (forehead + chin tend to get clipped)
    pad_x = max(10, int((x_max - x_min) * 0.35))
    pad_y = max(10, int((y_max - y_min) * 0.60))
    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(w, x_max + pad_x)
    y_max = min(h, y_max + pad_y)

    if x_max <= x_min or y_max <= y_min:
        return frame

    # ---- crop -------------------------------------------------------
    src  = source_frame if source_frame is not None else frame
    crop = src[y_min:y_max, x_min:x_max].copy()
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