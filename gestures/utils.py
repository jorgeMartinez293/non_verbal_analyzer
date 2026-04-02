"""Shared utility functions for gesture detectors."""

from __future__ import annotations
import math


def dist_2d(a, b) -> float:
    """Euclidean distance between two landmarks in 2D normalised space."""
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def visibility_gate(pose, indices: list[int], min_vis: float) -> bool:
    """Return True if ALL listed pose landmarks exceed *min_vis*."""
    try:
        return all(pose[i].visibility >= min_vis for i in indices)
    except (IndexError, AttributeError):
        return False


def facing_camera(pose, min_ratio: float,
                  shoulder_idx=(11, 12), hip_idx=(23, 24)) -> bool | None:
    """Check shoulder-span / torso-height ratio (distance-invariant).

    Returns True if the subject appears to face the camera, False if turned
    too far, and None if hips are not visible (caller decides whether to
    skip or fail the gate).
    """
    try:
        ls, rs = pose[shoulder_idx[0]], pose[shoulder_idx[1]]
        lh, rh = pose[hip_idx[0]], pose[hip_idx[1]]
    except (IndexError, AttributeError):
        return None

    if lh.visibility < 0.3 or rh.visibility < 0.3:
        return None

    shoulder_y = (ls.y + rs.y) / 2.0
    hip_y      = (lh.y + rh.y) / 2.0
    torso_h    = hip_y - shoulder_y
    if torso_h <= 0:
        return False

    shoulder_dist = ls.x - rs.x
    return shoulder_dist / torso_h >= min_ratio
