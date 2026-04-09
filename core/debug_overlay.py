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
_COL_GAP     = 10   # horizontal gap between the two graph columns
_COL_W       = _GRAPH_W + _VAL_W          # 205 — width of one graph column
SIDEBAR_W    = _COL_W * 2 + _COL_GAP      # 420 — total sidebar width (public)
_history: dict[str, deque] = {}
_scale:   dict[str, float] = {}   # hi per metric — fixed at thr*1.3; expands up for refs

# MediaPipe Pose landmark indices used by CrossedArms
_IDX = {
    "L_SHOULDER": 11, "R_SHOULDER": 12,
    "L_WRIST":    15, "R_WRIST":    16,
    "L_HIP":      23, "R_HIP":      24,
}

_COL_SHOULDER = (255, 255,   0)
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
         gesture_states: dict | None = None, tracking_stable: bool = True,
         sidebar_frame: np.ndarray | None = None) -> np.ndarray:
    """Draw the 8 pose keypoints and diagnostic lines onto *frame* (in-place).

    *frame* must be a view of the video area only (video_w × video_h).
    Drawing is automatically clipped to the video bounds by numpy slicing.
    Pass *sidebar_frame* (the full wide canvas) to render metric graphs in
    the sidebar; if omitted the graphs are drawn on *frame* itself.
    """
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

    # ---- arm lines (shoulder → wrist, no elbow needed) ---------------
    for side in ("L", "R"):
        cv2.line(frame, _px(lm[f"{side}_SHOULDER"], w, h),
                        _px(lm[f"{side}_WRIST"],    w, h), _COL_BONE, 2)

    # ---- wrist crossing (shoulder-width normalised) ------------------
    cross_thresh  = crossed_cfg.get("wrist_cross_ratio", 0.10)
    shoulder_dist = lm["L_SHOULDER"].x - lm["R_SHOULDER"].x
    cross_ratio   = ((lm["R_WRIST"].x - lm["L_WRIST"].x) / shoulder_dist
                     if shoulder_dist > 0 else 0.0)
    crossed_ok    = cross_ratio >= cross_thresh

    # Wrist connector: green when wrists are crossed past the threshold
    cv2.line(frame, _px(lm["L_WRIST"], w, h), _px(lm["R_WRIST"], w, h),
             _COL_MET if crossed_ok else _COL_FAIL, 3)

    # ---- wrist dots --------------------------------------------------
    for key in ("L_WRIST", "R_WRIST"):
        col = _COL_MET if crossed_ok else _COL_FAIL
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
            cv2.circle(frame, _px(lm2, w, h), 4, col, -1)

    _update_and_draw_graphs(
        sidebar_frame if sidebar_frame is not None else frame,
        gesture_states, thresholds, landmarks, tracking_stable,
    )
    return frame


# -----------------------------------------------------------------------
def _update_and_draw_graphs(
    frame: np.ndarray,
    gesture_states: dict | None,
    thresholds: dict,
    landmarks: dict,
    tracking_stable: bool = True,
) -> None:
    """Update rolling metric history and render mini time-series graphs."""
    ca_state  = (gesture_states or {}).get("crossed_arms", {})
    oa_state  = (gesture_states or {}).get("open_arms", {})
    ra_state  = (gesture_states or {}).get("raised_arms", {})
    re_state  = (gesture_states or {}).get("raised_eyebrows", {})
    bf_state  = (gesture_states or {}).get("blink_frequency", {})
    tf_state  = (gesture_states or {}).get("touch_face", {})

    cross_thr = thresholds.get("crossed_arms", {}).get("wrist_cross_ratio", 0.10)
    open_thr  = thresholds.get("open_arms", {}).get("min_open_ratio", 2.0)
    ra_thr    = thresholds.get("raised_arms", {}).get("min_raise_margin", 0.05)
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

    # (key, value, threshold, is_ref, calibrating, display_label)
    specs = [
        ("ls-rs", shoulder_eu,                     None,      True,  False,    "shoulder dist"),
        ("ca-cr", ca_state.get("cross_ratio"),      cross_thr, False, False,    "wrist cross"),
        ("lw-rw", oa_state.get("open_ratio"),       open_thr,  False, False,    "arm spread"),
        ("ra-l",  ra_state.get("raise_l"),          ra_thr,    False, False,    "raise left"),
        ("ra-r",  ra_state.get("raise_r"),          ra_thr,    False, False,    "raise right"),
        ("iris",  inter_iris,                       None,      True,  False,    "inter-iris"),
        ("rb-ri", re_state.get("ratio_r"),          brow_thr,  False, calib,    "eyebrow right"),
        ("lb-li", re_state.get("ratio_l"),          brow_thr,  False, calib,    "eyebrow left"),
        ("ear-r", bf_state.get("ear_r"),            ear_thr,   False, False,    "eye AR right"),
        ("ear-l", bf_state.get("ear_l"),            ear_thr,   False, False,    "eye AR left"),
        ("bpm",   bf_state.get("blink_rate"),       None,      True,  False,    "blink rate"),
        ("tf-hf", tf_state.get("hand_face_ratio"),  tf_thr,    False, False,    "hand-face"),
    ]

    for key, val, *_ in specs:
        if key not in _history:
            _history[key] = deque(maxlen=_HISTORY_LEN)
        if val is not None:
            _history[key].append(float(val))

    # ---- sidebar background + separator line -------------------------
    h_frame = frame.shape[0]
    cv2.rectangle(frame, (0, 0), (SIDEBAR_W - 1, h_frame), (18, 18, 18), -1)
    cv2.line(frame, (SIDEBAR_W - 1, 0), (SIDEBAR_W - 1, h_frame), (60, 60, 60), 1)

    # ---- 2-column graph layout ---------------------------------------
    col_xs = [0, _COL_W + _COL_GAP]   # x-start for each column
    col_ys = [0, 0]                    # current y for each column
    half   = (len(specs) + 1) // 2    # first column gets ceil(n/2)

    for i, (key, _val, thr, is_ref, is_calib, label) in enumerate(specs):
        col = 0 if i < half else 1
        x0  = col_xs[col]
        y0  = col_ys[col]
        if y0 + _LABEL_H + _GRAPH_H > h_frame:
            continue
        buf = _history.get(key, deque())
        _draw_one_graph(frame, x0, y0, label, buf, thr, is_ref, is_calib)
        col_ys[col] += _LABEL_H + _GRAPH_H + _GRAPH_GAP

    # ---- horizontal confirmation bars below the graphs ----------------
    if not gesture_states:
        return
    # metric-only gestures: exclude from confirmation bars
    _NO_BARS = {"blink_frequency"}
    bar_states = {k: v for k, v in gesture_states.items() if k not in _NO_BARS}
    n_gest    = len(bar_states)
    if n_gest == 0:
        return
    bar_gap   = 4
    bar_h     = 40
    bar_y     = min(max(col_ys) + bar_gap, h_frame - bar_h - 2)
    if bar_y < 0:
        return
    available_w = SIDEBAR_W - 2 * bar_gap
    bar_w       = (available_w - bar_gap * (n_gest - 1)) // max(n_gest, 1)

    for i, (g_name, g_state) in enumerate(bar_states.items()):
        bx = bar_gap + i * (bar_w + bar_gap)
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
            fill_ratio = cooldown_rem / cooldown_tot
            col        = (0, 140, 0)
        elif calib:
            calib_n   = g_state.get("calib_samples", 0)
            calib_tot = max(1, g_state.get("calib_target", 200))
            fill_ratio = calib_n / calib_tot
            col        = (200, 200, 200)
        else:
            pos = g_state.get("positive_frames", 0)
            wsz = max(1, g_state.get("window_size", 1))
            fill_ratio = pos / wsz
            col        = (0, 200, 0)

        fill_w = int(fill_ratio * bar_w)
        if fill_w > 0:
            cv2.rectangle(frame, (bx, by), (bx + fill_w, by + bar_h), col, -1)

        # white threshold marker
        thr_x = bx + int(required * bar_w)
        thr_x = max(bx, min(bx + bar_w - 1, thr_x))
        cv2.line(frame, (thr_x, by), (thr_x, by + bar_h), (255, 255, 255), 1)

        # label
        short = {
            "crossed_arms": "ca", "open_arms": "oa", "raised_arms": "ra",
            "raised_eyebrows": "re", "touch_face": "tf",
        }.get(g_name) or g_name[:2]
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
    video_x_offset: int = 0,
    video_w: int | None = None,
    video_h: int | None = None,
) -> np.ndarray:
    """Draws the DETECTED banner at the bottom of the video area when a gesture fires."""
    h_total, w_total = frame.shape[:2]
    w  = video_w if video_w is not None else w_total
    h  = video_h if video_h is not None else h_total   # banners anchor to bottom of video area
    xo = video_x_offset
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
            cv2.rectangle(overlay, (xo, y_cursor), (xo + w, y_cursor + banner_h), (0, 140, 0), -1)
            cv2.addWeighted(overlay, alpha * 0.8, frame, 1 - alpha * 0.8, 0, frame)
            label = f"DETECTED: {gesture_name.replace('_', ' ').upper()}"
            tw = cv2.getTextSize(label, font, font_scale * 1.4, 2)[0][0]
            cv2.putText(frame, label, (xo + (w - tw) // 2, y_cursor + banner_h - pad),
                        font, font_scale * 1.4, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


# ── fixed proportions for the bottom-strip insets ────────────────────────
_INSET_FACE_ASPECT = 0.80   # face crop  width / height
_INSET_HAND_ASPECT = 0.75   # hand crop  width / height


def _placeholder(w: int, h: int, label: str) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (55, 55, 55), 1)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.putText(img, label, ((w - tw) // 2, (h + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 90, 90), 1, cv2.LINE_AA)
    return img


def _crop_face(
    src: np.ndarray,
    landmarks: dict,
    cached_face_lms,
    vw: int, vh: int, xo: int,
    out_w: int, out_h: int,
) -> np.ndarray:
    pose_lms = landmarks.get("pose")
    face_lms = landmarks.get("face") or cached_face_lms

    if pose_lms is None:
        return _placeholder(out_w, out_h, "face")

    head = [pose_lms[i] for i in range(11) if pose_lms[i].visibility > 0.3]
    if not head:
        return _placeholder(out_w, out_h, "face")

    xs = [lm.x * vw for lm in head]
    ys = [lm.y * vh for lm in head]
    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))
    pad_x = max(10, int((x_max - x_min) * 0.35))
    pad_y = max(10, int((y_max - y_min) * 0.60))
    x_min = max(0, x_min - pad_x);  x_max = min(vw, x_max + pad_x)
    y_min = max(0, y_min - pad_y);  y_max = min(vh, y_max + pad_y)

    if x_max <= x_min or y_max <= y_min:
        return _placeholder(out_w, out_h, "face")

    crop_x0 = max(0, xo + x_min);  crop_x1 = min(src.shape[1], xo + x_max)
    if crop_x1 <= crop_x0:
        return _placeholder(out_w, out_h, "face")

    crop = src[y_min:y_max, crop_x0:crop_x1]
    if crop.size == 0:
        return _placeholder(out_w, out_h, "face")

    result = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    if face_lms:
        sx = out_w / max(x_max - x_min, 1)
        sy = out_h / max(y_max - y_min, 1)
        for lm in face_lms:
            px = int((lm.x * vw - x_min) * sx)
            py = int((lm.y * vh - y_min) * sy)
            if 0 <= px < out_w and 0 <= py < out_h:
                cv2.circle(result, (px, py), 1, (0, 230, 0), -1)

    cv2.rectangle(result, (0, 0), (out_w - 1, out_h - 1), (200, 200, 200), 2)
    return result


def _crop_hand(
    src: np.ndarray,
    landmarks: dict,
    hand_key: str,
    vw: int, vh: int, xo: int,
    out_w: int, out_h: int,
) -> np.ndarray:
    hand = landmarks.get(hand_key)
    lbl  = "L hand" if hand_key == "left_hand" else "R hand"
    if hand is None:
        return _placeholder(out_w, out_h, lbl)

    xs = [lm.x * vw for lm in hand]
    ys = [lm.y * vh for lm in hand]
    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))
    pad = max(15, int(max(x_max - x_min, y_max - y_min) * 0.35))
    x_min = max(0, x_min - pad);  x_max = min(vw, x_max + pad)
    y_min = max(0, y_min - pad);  y_max = min(vh, y_max + pad)

    if x_max <= x_min or y_max <= y_min:
        return _placeholder(out_w, out_h, lbl)

    crop_x0 = max(0, xo + x_min);  crop_x1 = min(src.shape[1], xo + x_max)
    if crop_x1 <= crop_x0:
        return _placeholder(out_w, out_h, lbl)

    crop = src[y_min:y_max, crop_x0:crop_x1]
    if crop.size == 0:
        return _placeholder(out_w, out_h, lbl)

    result = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    sx = out_w / max(x_max - x_min, 1)
    sy = out_h / max(y_max - y_min, 1)
    for lm in hand:
        px = int((lm.x * vw - x_min) * sx)
        py = int((lm.y * vh - y_min) * sy)
        if 0 <= px < out_w and 0 <= py < out_h:
            cv2.circle(result, (px, py), 3, (0, 200, 255), -1)

    cv2.rectangle(result, (0, 0), (out_w - 1, out_h - 1), (200, 200, 200), 2)
    return result


# -----------------------------------------------------------------------
def draw_bottom_insets(
    frame: np.ndarray,
    landmarks: dict,
    cached_face_lms=None,
    source_frame: np.ndarray | None = None,
    video_x_offset: int = 0,
    video_w: int | None = None,
    video_h: int | None = None,
) -> np.ndarray:
    """Render face + hand crops at fixed proportions in the bottom strip."""
    h_total, w_total = frame.shape[:2]
    vw  = video_w  if video_w  is not None else w_total
    vh  = video_h  if video_h  is not None else h_total
    xo  = video_x_offset
    src = source_frame if source_frame is not None else frame

    bottom_h = h_total - vh
    if bottom_h <= 4:
        return frame

    margin  = 8
    gap     = 12
    inset_h = bottom_h - 2 * margin
    if inset_h <= 0:
        return frame

    face_w = max(1, int(inset_h * _INSET_FACE_ASPECT))
    hand_w = max(1, int(inset_h * _INSET_HAND_ASPECT))
    y_top  = vh + margin

    crops = [
        _crop_face(src, landmarks, cached_face_lms, vw, vh, xo, face_w, inset_h),
        _crop_hand(src, landmarks, "left_hand",     vw, vh, xo, hand_w, inset_h),
        _crop_hand(src, landmarks, "right_hand",    vw, vh, xo, hand_w, inset_h),
    ]

    total_w = sum(c.shape[1] for c in crops) + gap * (len(crops) - 1)
    x = (w_total - total_w) // 2

    for crop in crops:
        ch, cw = crop.shape[:2]
        if x >= 0 and x + cw <= w_total and y_top + ch <= h_total:
            frame[y_top:y_top + ch, x:x + cw] = crop
        x += cw + gap

    return frame