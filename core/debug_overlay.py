"""
Debug Overlay

Draws the 8 pose landmarks used by CrossedArms detection onto a frame,
plus diagnostic text showing the live wrist-cross value vs the threshold.

Visual legend
-------------
  Cyan circles    — shoulder landmarks (L/R)
  Yellow circles  — elbow landmarks (L/R)
  White circles   — wrist landmarks (L/R)  ← the primary decision points
  Grey circles    — hip landmarks (L/R)
  Green lines     — arm skeleton (shoulder→elbow→wrist)
  Green line      — wrist-to-wrist connector  (condition MET)
  Red line        — wrist-to-wrist connector  (condition NOT met)
  Blue rectangle  — torso height band used for the height check
  HUD text        — live values of every threshold condition
"""

from __future__ import annotations
import cv2
import numpy as np

# MediaPipe Pose landmark indices used by CrossedArms
_IDX = {
    "L_SHOULDER": 11,
    "R_SHOULDER": 12,
    "L_ELBOW":    13,
    "R_ELBOW":    14,
    "L_WRIST":    15,
    "R_WRIST":    16,
    "L_HIP":      23,
    "R_HIP":      24,
}

# BGR colours
_COL_SHOULDER = (255, 255,   0)   # cyan
_COL_ELBOW    = (  0, 255, 255)   # yellow
_COL_WRIST    = (255, 255, 255)   # white
_COL_HIP      = (160, 160, 160)   # grey
_COL_BONE     = ( 80, 200,  80)   # green skeleton
_COL_MET      = (  0, 220,   0)   # green  — condition met
_COL_FAIL     = (  0,   0, 220)   # red    — condition not met
_COL_BAND     = (200, 120,   0)   # blue   — torso band
_COL_HUD_BG   = (  0,   0,   0)
_COL_HUD_OK   = (  0, 220,   0)
_COL_HUD_FAIL = (  0,  80, 220)


def _px(landmark, w: int, h: int) -> tuple[int, int]:
    """Convert normalised landmark to pixel coords."""
    return int(landmark.x * w), int(landmark.y * h)


def draw(frame: np.ndarray, landmarks: dict, thresholds: dict) -> np.ndarray:
    """
    Return a copy of *frame* with the debug overlay applied.

    Parameters
    ----------
    frame      : BGR numpy array
    landmarks  : dict as produced by VideoProcessor._extract_landmarks()
    thresholds : the 'crossed_arms' sub-dict from thresholds.json
    """
    out = frame.copy()
    h, w = out.shape[:2]

    pose = landmarks.get("pose")
    if pose is None:
        _put_hud(out, ["[no pose detected]"], w, h)
        return out

    # ---- gather the 8 points ----------------------------------------
    try:
        lm = {name: pose[idx] for name, idx in _IDX.items()}
    except IndexError:
        _put_hud(out, ["[landmark index out of range]"], w, h)
        return out

    # ---- torso band rectangle ----------------------------------------
    shoulder_y = (lm["L_SHOULDER"].y + lm["R_SHOULDER"].y) / 2.0
    hip_y      = (lm["L_HIP"].y     + lm["R_HIP"].y)      / 2.0
    torso_h    = hip_y - shoulder_y

    h_min = thresholds.get("wrist_height_min_ratio", 0.0)
    h_max = thresholds.get("wrist_height_max_ratio", 1.1)

    band_top    = int((shoulder_y + h_min * torso_h) * h)
    band_bottom = int((shoulder_y + h_max * torso_h) * h)
    band_left   = int(min(lm["L_SHOULDER"].x, lm["R_SHOULDER"].x) * w) - 20
    band_right  = int(max(lm["L_SHOULDER"].x, lm["R_SHOULDER"].x) * w) + 20

    overlay = out.copy()
    cv2.rectangle(overlay, (band_left, band_top), (band_right, band_bottom),
                  _COL_BAND, -1)
    cv2.addWeighted(overlay, 0.15, out, 0.85, 0, out)
    cv2.rectangle(out, (band_left, band_top), (band_right, band_bottom),
                  _COL_BAND, 1)

    # ---- arm skeleton ------------------------------------------------
    for side in ("L", "R"):
        p_sh = _px(lm[f"{side}_SHOULDER"], w, h)
        p_el = _px(lm[f"{side}_ELBOW"],    w, h)
        p_wr = _px(lm[f"{side}_WRIST"],    w, h)
        cv2.line(out, p_sh, p_el, _COL_BONE, 2)
        cv2.line(out, p_el, p_wr, _COL_BONE, 2)

    # ---- wrist-to-wrist line (colour = crossing condition) -----------
    margin       = thresholds.get("wrist_cross_margin", 0.05)
    # Positive when R_WRIST is to the RIGHT of L_WRIST in image coords = crossed
    cross_value  = lm["R_WRIST"].x - lm["L_WRIST"].x
    cross_met    = cross_value >= margin

    p_lw = _px(lm["L_WRIST"], w, h)
    p_rw = _px(lm["R_WRIST"], w, h)
    wrist_col = _COL_MET if cross_met else _COL_FAIL
    cv2.line(out, p_lw, p_rw, wrist_col, 3)

    # ---- height check per wrist --------------------------------------
    for side, key in (("L", "L_WRIST"), ("R", "R_WRIST")):
        wrist     = lm[key]
        ratio     = (wrist.y - shoulder_y) / torso_h if torso_h > 0 else 0
        in_band   = h_min <= ratio <= h_max
        dot_col   = _COL_MET if in_band else _COL_FAIL
        cv2.circle(out, _px(wrist, w, h), 8, dot_col, -1)
        cv2.circle(out, _px(wrist, w, h), 8, (255, 255, 255), 1)

    # ---- remaining landmark dots ------------------------------------
    dot_map = [
        ("L_SHOULDER", _COL_SHOULDER), ("R_SHOULDER", _COL_SHOULDER),
        ("L_ELBOW",    _COL_ELBOW),    ("R_ELBOW",    _COL_ELBOW),
        ("L_HIP",      _COL_HIP),      ("R_HIP",      _COL_HIP),
    ]
    for name, col in dot_map:
        cv2.circle(out, _px(lm[name], w, h), 6, col, -1)
        cv2.circle(out, _px(lm[name], w, h), 6, (255, 255, 255), 1)

    # ---- HUD text ---------------------------------------------------
    lw_ratio = (lm["L_WRIST"].y - shoulder_y) / torso_h if torso_h > 0 else 0
    rw_ratio = (lm["R_WRIST"].y - shoulder_y) / torso_h if torso_h > 0 else 0

    lines = [
        (f"R-L wrist: {cross_value:+.3f}  (need >= {margin:.3f})",
         _COL_HUD_OK if cross_met else _COL_HUD_FAIL),
        (f"L_wrist height ratio: {lw_ratio:.2f}  [{h_min:.1f} – {h_max:.1f}]",
         _COL_HUD_OK if h_min <= lw_ratio <= h_max else _COL_HUD_FAIL),
        (f"R_wrist height ratio: {rw_ratio:.2f}  [{h_min:.1f} – {h_max:.1f}]",
         _COL_HUD_OK if h_min <= rw_ratio <= h_max else _COL_HUD_FAIL),
        (f"visibility threshold: {thresholds.get('min_landmark_visibility', 0.5):.2f}",
         _COL_HUD_OK),
    ]
    _put_hud(out, lines, w, h)
    return out


def _put_hud(
    frame: np.ndarray,
    lines: list,
    w: int,
    h: int,
) -> None:
    """Render a semi-transparent HUD box in the top-left corner."""
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, w / 1600)
    thickness  = 1
    pad        = 6
    line_h     = int(18 * font_scale / 0.4)

    # Measure widest line
    max_tw = max(
        cv2.getTextSize(l if isinstance(l, str) else l[0], font, font_scale, thickness)[0][0]
        for l in lines
    )
    box_w = max_tw + pad * 2
    box_h = line_h * len(lines) + pad * 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, line in enumerate(lines):
        text  = line if isinstance(line, str) else line[0]
        color = (200, 200, 200) if isinstance(line, str) else line[1]
        y     = pad + (i + 1) * line_h
        cv2.putText(frame, text, (pad, y), font, font_scale, color, thickness, cv2.LINE_AA)
