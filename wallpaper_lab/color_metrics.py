from __future__ import annotations

import numpy as np


def lab_chroma(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float32)
    return np.sqrt(np.square(lab[..., 1]) + np.square(lab[..., 2])).astype(np.float32)


def lab_hue_degrees(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float32)
    hue = np.degrees(np.arctan2(lab[..., 2], lab[..., 1]))
    return np.mod(hue, 360.0).astype(np.float32)


def delta_e_cie1976(lab_a: np.ndarray, lab_b: np.ndarray) -> np.ndarray:
    lab_a = np.asarray(lab_a, dtype=np.float32)
    lab_b = np.asarray(lab_b, dtype=np.float32)
    return np.linalg.norm(lab_a - lab_b, axis=-1).astype(np.float32)


def weighted_circular_mean_degrees(
    angles_degrees: np.ndarray,
    weights: np.ndarray | None = None,
    min_weight_sum: float = 1e-6,
) -> float:
    angles = np.asarray(angles_degrees, dtype=np.float32)
    if angles.size == 0:
        return float("nan")

    if weights is None:
        weights_array = np.ones_like(angles, dtype=np.float32)
    else:
        weights_array = np.asarray(weights, dtype=np.float32)

    weight_sum = float(np.sum(weights_array))
    if weight_sum <= min_weight_sum:
        return float("nan")

    radians = np.deg2rad(angles)
    sin_mean = float(np.sum(np.sin(radians) * weights_array) / weight_sum)
    cos_mean = float(np.sum(np.cos(radians) * weights_array) / weight_sum)
    if abs(sin_mean) <= min_weight_sum and abs(cos_mean) <= min_weight_sum:
        return float("nan")
    return float(np.mod(np.degrees(np.arctan2(sin_mean, cos_mean)), 360.0))


def weighted_circular_std_degrees(
    angles_degrees: np.ndarray,
    weights: np.ndarray | None = None,
    min_weight_sum: float = 1e-6,
) -> float:
    angles = np.asarray(angles_degrees, dtype=np.float32)
    if angles.size == 0:
        return float("nan")

    if weights is None:
        weights_array = np.ones_like(angles, dtype=np.float32)
    else:
        weights_array = np.asarray(weights, dtype=np.float32)

    weight_sum = float(np.sum(weights_array))
    if weight_sum <= min_weight_sum:
        return float("nan")

    radians = np.deg2rad(angles)
    sin_mean = float(np.sum(np.sin(radians) * weights_array) / weight_sum)
    cos_mean = float(np.sum(np.cos(radians) * weights_array) / weight_sum)
    resultant_length = min(1.0, max(0.0, float(np.hypot(sin_mean, cos_mean))))
    if resultant_length <= min_weight_sum:
        return 180.0
    return float(np.degrees(np.sqrt(-2.0 * np.log(resultant_length))))
