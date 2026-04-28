from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np

from wallpaper_lab.colorchecker import order_corners
from wallpaper_lab.io_utils import as_uint8


def overlay_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    color_rgb: tuple[int, int, int] = (0, 210, 80),
    alpha: float = 0.45,
) -> np.ndarray:
    base = as_uint8(image_rgb).astype(np.float32)
    overlay = base.copy()
    overlay[mask] = (
        (1.0 - alpha) * overlay[mask]
        + alpha * np.array(color_rgb, dtype=np.float32)
    )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def draw_colorchecker_overlay(
    image_rgb: np.ndarray,
    corners: np.ndarray,
    color_rgb: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    overlay = as_uint8(image_rgb).copy()
    ordered = np.round(order_corners(corners)).astype(np.int32)
    cv2.polylines(overlay, [ordered], isClosed=True, color=color_rgb, thickness=6)
    labels = ["TL", "TR", "BR", "BL"]
    for label, point in zip(labels, ordered):
        cv2.circle(overlay, tuple(point), 18, color_rgb, thickness=-1)
        cv2.putText(
            overlay,
            label,
            (int(point[0] + 12), int(point[1] - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color_rgb,
            2,
            cv2.LINE_AA,
        )
    return overlay


def draw_polygon_overlays(
    image_rgb: np.ndarray,
    operations: list,
) -> np.ndarray:
    overlay = as_uint8(image_rgb).copy()
    for operation in operations:
        if len(operation.points) < 2:
            continue
        points = np.round(np.array(operation.points, dtype=np.float32)).astype(np.int32)
        color = (0, 210, 80) if operation.operation == "include" else (255, 80, 80)
        cv2.polylines(overlay, [points], isClosed=True, color=color, thickness=4)
        cv2.putText(
            overlay,
            operation.name,
            tuple(points[0] + np.array([10, -10])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    return overlay


def draw_points(
    image_rgb: np.ndarray,
    points: list[tuple[float, float]],
    color_rgb: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    overlay = as_uint8(image_rgb).copy()
    for index, point in enumerate(points, start=1):
        px, py = int(round(point[0])), int(round(point[1]))
        cv2.circle(overlay, (px, py), 14, color_rgb, thickness=-1)
        cv2.putText(
            overlay,
            str(index),
            (px + 10, py - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color_rgb,
            2,
            cv2.LINE_AA,
        )
    return overlay


def create_lab_distribution_figure(
    lab_image: np.ndarray,
    mask: np.ndarray,
    point_color: tuple[int, int, int] = (0, 210, 80),
) -> plt.Figure:
    pixels = lab_image[mask]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    if pixels.size == 0:
        axes[0].text(0.5, 0.5, "No sampled pixels", ha="center", va="center")
        axes[1].text(0.5, 0.5, "No sampled pixels", ha="center", va="center")
        for axis in axes:
            axis.set_axis_off()
        figure.tight_layout()
        return figure

    if pixels.shape[0] > 12000:
        rng = np.random.default_rng(42)
        sample = pixels[rng.choice(pixels.shape[0], size=12000, replace=False)]
    else:
        sample = pixels

    scatter_color = np.array(point_color, dtype=np.float32) / 255.0
    axes[0].scatter(sample[:, 1], sample[:, 2], s=5, alpha=0.3, c=[scatter_color])
    axes[0].set_xlabel("a*")
    axes[0].set_ylabel("b*")
    axes[0].set_title("Selected Pixels: a* vs b*")
    axes[0].grid(alpha=0.2)

    axes[1].hist(sample[:, 0], bins=30, color="#4f8f4f", alpha=0.7, label="L*")
    axes[1].hist(sample[:, 1], bins=30, color="#7c3f98", alpha=0.5, label="a*")
    axes[1].hist(sample[:, 2], bins=30, color="#cc8a00", alpha=0.5, label="b*")
    axes[1].set_title("Lab Channel Distributions")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    figure.tight_layout()
    return figure
