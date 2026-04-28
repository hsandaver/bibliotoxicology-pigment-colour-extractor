from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from skimage import draw

from wallpaper_lab.color_metrics import (
    lab_chroma,
    lab_hue_degrees,
    weighted_circular_mean_degrees,
    weighted_circular_std_degrees,
)


@dataclass
class PolygonOperation:
    name: str
    points: list[tuple[float, float]]
    operation: str = "include"


def polygon_to_mask(shape: tuple[int, int], points: list[tuple[float, float]]) -> np.ndarray:
    if len(points) < 3:
        return np.zeros(shape, dtype=bool)
    rows = np.array([point[1] for point in points], dtype=np.float32)
    cols = np.array([point[0] for point in points], dtype=np.float32)
    rr, cc = draw.polygon(rows, cols, shape)
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    return mask


def apply_polygon_operations(
    base_mask: np.ndarray,
    operations: list[PolygonOperation],
) -> np.ndarray:
    adjusted = base_mask.copy()
    for operation in operations:
        polygon_mask = polygon_to_mask(base_mask.shape, operation.points)
        if operation.operation == "include":
            adjusted |= polygon_mask
        elif operation.operation == "exclude":
            adjusted &= ~polygon_mask
    return adjusted


def build_manual_roi_masks(
    image_shape: tuple[int, int],
    polygons: list[PolygonOperation],
    pigment_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    roi_masks: dict[str, np.ndarray] = {}
    for polygon in polygons:
        polygon_mask = polygon_to_mask(image_shape, polygon.points)
        roi_masks[polygon.name] = polygon_mask & pigment_mask
    return roi_masks


def build_fragment_roi_masks(
    fragment_labels: np.ndarray,
    pigment_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    roi_masks: dict[str, np.ndarray] = {}
    for label_id in np.unique(fragment_labels):
        if label_id == 0:
            continue
        mask = (fragment_labels == label_id) & pigment_mask
        if int(mask.sum()) == 0:
            continue
        roi_masks[f"fragment_{label_id}"] = mask
    return roi_masks


def compute_lab_statistics(
    lab_image: np.ndarray,
    roi_masks: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for roi_name, mask in roi_masks.items():
        pixels = lab_image[mask]
        if pixels.size == 0:
            continue
        chroma = lab_chroma(pixels)
        hue = lab_hue_degrees(pixels)
        rows.append(
            {
                "roi": roi_name,
                "pixel_count": int(pixels.shape[0]),
                "mean_L": float(np.mean(pixels[:, 0])),
                "mean_a": float(np.mean(pixels[:, 1])),
                "mean_b": float(np.mean(pixels[:, 2])),
                "mean_C": float(np.mean(chroma)),
                "mean_h": weighted_circular_mean_degrees(hue, chroma),
                "median_L": float(np.median(pixels[:, 0])),
                "median_a": float(np.median(pixels[:, 1])),
                "median_b": float(np.median(pixels[:, 2])),
                "median_C": float(np.median(chroma)),
                "std_L": float(np.std(pixels[:, 0], ddof=0)),
                "std_a": float(np.std(pixels[:, 1], ddof=0)),
                "std_b": float(np.std(pixels[:, 2], ddof=0)),
                "std_C": float(np.std(chroma, ddof=0)),
                "std_h": weighted_circular_std_degrees(hue, chroma),
            }
        )
    return pd.DataFrame(rows)
