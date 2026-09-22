"""Pitch geometry and coordinate conventions.

Canonical internal representation: **metres, centre-origin**.

    x in [-L/2, +L/2]   negative = own half, positive = attacking half
    y in [-W/2, +W/2]   negative = left, positive = right

Metres rather than a per-axis normalisation to [-1, 1], because normalising each
axis independently destroys the aspect ratio: one x-unit would be 52.5m and one
y-unit 34m, making every distance, speed and pitch-control value anisotropic.
Storing metres keeps distances real and velocities in m/s, directly comparable to
the max-speed figures the player capability model reads from GPS vests.

Normalised views are provided for model input (isotropic, divided by a single
scale) and for the ROLES.md channel lookups (per-axis, where that is correct).
"""

from __future__ import annotations

import numpy as np

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

HALF_LENGTH = PITCH_LENGTH / 2
HALF_WIDTH = PITCH_WIDTH / 2

OWN_GOAL = np.array([-HALF_LENGTH, 0.0], dtype=np.float32)
OPP_GOAL = np.array([+HALF_LENGTH, 0.0], dtype=np.float32)

# Five vertical channels (ROLES.md §1), as fractions of half-width.
CHANNEL_EDGES = np.array([-1.0, -0.6, -0.2, 0.2, 0.6, 1.0], dtype=np.float32)
CHANNEL_NAMES = ("left_wing", "left_half", "centre", "right_half", "right_wing")

# Thirds, as fractions of half-length.
THIRD_EDGES = np.array([-1.0, -1 / 3, 1 / 3, 1.0], dtype=np.float32)
THIRD_NAMES = ("defensive", "middle", "attacking")


def from_kloppy(xy: np.ndarray) -> np.ndarray:
    """kloppy normalised [0, 1] on both axes -> metres, centre-origin.

    kloppy's y axis points down the screen; ours points right from the attacking
    team's perspective, so y is inverted here.
    """
    out = np.empty_like(xy, dtype=np.float32)
    out[..., 0] = (xy[..., 0].astype(np.float32) - 0.5) * PITCH_LENGTH
    out[..., 1] = (0.5 - xy[..., 1].astype(np.float32)) * PITCH_WIDTH
    return out


def to_kloppy(xy: np.ndarray) -> np.ndarray:
    """Inverse of :func:`from_kloppy`, for plotting against mplsoccer."""
    out = np.empty_like(xy, dtype=np.float32)
    out[..., 0] = xy[..., 0] / PITCH_LENGTH + 0.5
    out[..., 1] = 0.5 - xy[..., 1] / PITCH_WIDTH
    return out


def to_model_scale(xy: np.ndarray) -> np.ndarray:
    """Metres -> isotropic normalised units (both axes divided by HALF_LENGTH).

    Aspect ratio preserved: x spans [-1, 1], y spans about [-0.648, 0.648].
    This is what the model consumes.
    """
    return (xy / HALF_LENGTH).astype(np.float32)


def to_axis_norm(xy: np.ndarray) -> np.ndarray:
    """Metres -> per-axis normalised [-1, 1], for channel/third lookups only.

    Anisotropic by construction. Never feed this to a distance calculation.
    """
    out = np.empty_like(xy, dtype=np.float32)
    out[..., 0] = xy[..., 0] / HALF_LENGTH
    out[..., 1] = xy[..., 1] / HALF_WIDTH
    return out


def channel_of(y_metres: np.ndarray) -> np.ndarray:
    """Vertical channel index 0..4 (left wing -> right wing)."""
    y_norm = np.asarray(y_metres, dtype=np.float32) / HALF_WIDTH
    return np.clip(np.searchsorted(CHANNEL_EDGES[1:-1], y_norm, side="right"), 0, 4)


def third_of(x_metres: np.ndarray) -> np.ndarray:
    """Pitch third index 0..2 (defensive -> attacking)."""
    x_norm = np.asarray(x_metres, dtype=np.float32) / HALF_LENGTH
    return np.clip(np.searchsorted(THIRD_EDGES[1:-1], x_norm, side="right"), 0, 2)


def dist_to(xy: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Euclidean distance in metres from each position to a fixed point."""
    return np.linalg.norm(xy - point, axis=-1).astype(np.float32)


def bearing_to(xy: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Unit vector from each position toward a point, as (cos, sin).

    Returned as a direction pair rather than an angle: raw radians have a
    discontinuity at +/-pi that poisons any model consuming them.
    """
    delta = (point - xy).astype(np.float32)
    norm = np.linalg.norm(delta, axis=-1, keepdims=True)
    return np.divide(delta, norm, out=np.zeros_like(delta), where=norm > 1e-6)
