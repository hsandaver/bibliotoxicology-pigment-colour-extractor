from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from wallpaper_lab.io_utils import as_uint8


@dataclass
class ColorCheckerDetection:
    corners: np.ndarray
    score: float
    method: str
    debug_mask: np.ndarray | None = None


def _score_candidate_rect(
    contour: np.ndarray,
    rect: tuple[tuple[float, float], tuple[float, float], float],
    gray: np.ndarray,
    saturation: np.ndarray,
    hue: np.ndarray,
) -> float:
    (rect_w, rect_h) = rect[1]
    rect_area = float(rect_w * rect_h)
    if rect_area <= 1.0:
        return 0.0

    area = float(cv2.contourArea(contour))
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    fill_ratio = area / rect_area
    hull_fill_ratio = area / max(hull_area, 1.0)

    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    vertex_penalty = 1.0 if len(approx) == 4 else max(0.2, 4.0 / max(len(approx), 4))

    box = cv2.boxPoints(rect)
    box_int = np.round(box).astype(np.int32)
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.fillConvexPoly(mask, box_int, 255)
    masked_gray = gray[mask > 0]
    masked_sat = saturation[mask > 0]
    masked_hue = hue[mask > 0]
    if masked_gray.size == 0:
        return 0.0

    mean_darkness = 255.0 - float(masked_gray.mean())
    mean_saturation = float(masked_sat.mean())
    hue_std = float(masked_hue.std())

    long_side = max(rect_w, rect_h)
    short_side = min(rect_w, rect_h)
    aspect = long_side / max(short_side, 1.0)
    aspect_score = 1.0 / (1.0 + abs(aspect - 1.5) * 2.5)

    rectangularity = max(0.0, min(fill_ratio, 1.0))
    convexity = max(0.0, min(hull_fill_ratio, 1.0))
    color_variation = min(hue_std, 60.0) / 60.0

    return (
        area
        * (rectangularity ** 3)
        * (convexity ** 2)
        * vertex_penalty
        * aspect_score
        * (0.5 + mean_darkness / 255.0)
        * (0.35 + mean_saturation / 255.0)
        * (0.35 + color_variation)
    )


def _is_image_sized_candidate(
    contour: np.ndarray,
    rect: tuple[tuple[float, float], tuple[float, float], float],
    image_shape: tuple[int, int],
) -> bool:
    """Reject dark-scene masks that span the photograph rather than the chart."""
    height, width = image_shape
    image_area = float(height * width)
    rect_w, rect_h = rect[1]
    rect_area_fraction = float(rect_w * rect_h) / max(image_area, 1.0)
    if rect_area_fraction >= 0.82:
        return True

    x, y, box_w, box_h = cv2.boundingRect(contour)
    border_margin = 2
    touches_left = x <= border_margin
    touches_top = y <= border_margin
    touches_right = x + box_w >= width - border_margin
    touches_bottom = y + box_h >= height - border_margin
    touches = sum(
        (
            touches_left,
            touches_top,
            touches_right,
            touches_bottom,
        )
    )
    bbox_area_fraction = float(box_w * box_h) / max(image_area, 1.0)
    spans_full_height = touches_top and touches_bottom and box_h >= height * 0.92
    spans_full_width = touches_left and touches_right and box_w >= width * 0.92
    if (spans_full_height or spans_full_width) and bbox_area_fraction >= 0.22:
        return True

    return touches >= 3 and bbox_area_fraction >= 0.70


def order_corners(corners: np.ndarray) -> np.ndarray:
    """Order arbitrary four-point corners as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(corners, dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).ravel()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def detect_colorchecker(
    image_rgb: np.ndarray,
    max_width: int = 1600,
) -> ColorCheckerDetection | None:
    """Detect a likely ColorChecker Classic rectangle without relying on image position."""
    image_u8 = as_uint8(image_rgb)
    height, width = image_u8.shape[:2]
    scale = min(1.0, max_width / float(width))
    resized_size = (int(round(width * scale)), int(round(height * scale)))
    small = cv2.resize(image_u8, resized_size, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    hue = hsv[:, :, 0]

    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, threshold = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV)
    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_CLOSE,
        np.ones((11, 11), dtype=np.uint8),
        iterations=2,
    )
    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_OPEN,
        np.ones((5, 5), dtype=np.uint8),
    )

    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best: ColorCheckerDetection | None = None
    image_area = float(threshold.shape[0] * threshold.shape[1])

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.01:
            continue

        rect = cv2.minAreaRect(contour)
        (_, _), (rect_w, rect_h), _ = rect
        if rect_w <= 1 or rect_h <= 1:
            continue
        if _is_image_sized_candidate(contour, rect, threshold.shape):
            continue

        long_side = max(rect_w, rect_h)
        short_side = min(rect_w, rect_h)
        aspect = long_side / short_side
        if not 1.15 <= aspect <= 1.95:
            continue

        score = _score_candidate_rect(contour, rect, gray, saturation, hue)
        if score <= 0.0:
            continue

        corners = order_corners(cv2.boxPoints(rect) / scale)
        candidate = ColorCheckerDetection(
            corners=corners,
            score=score,
            method="automatic contour detection",
            debug_mask=threshold,
        )
        if best is None or candidate.score > best.score:
            best = candidate

    patch_grid_detection = _detect_colorchecker_from_patch_grid(scale, hsv[:, :, 1], hsv[:, :, 2])
    if patch_grid_detection is not None and (best is None or patch_grid_detection.score > best.score):
        return patch_grid_detection

    return best


def _detect_colorchecker_from_patch_grid(
    scale: float,
    saturation: np.ndarray,
    value: np.ndarray,
) -> ColorCheckerDetection | None:
    """Infer the chart rectangle from the visible ColorChecker patch grid.

    Dark bookcloth photographs can make the entire scene one dark contour. In
    that case the coloured patch grid is a more reliable registration signal.
    """
    patch_mask = (((saturation > 45) & (value > 40)) | (value > 48)).astype(np.uint8) * 255
    patch_mask = cv2.morphologyEx(
        patch_mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
    )
    patch_mask = cv2.morphologyEx(
        patch_mask,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8),
    )

    contours, _ = cv2.findContours(patch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    height, width = patch_mask.shape
    image_area = float(height * width)
    boxes: list[tuple[float, float, float, float, float]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < image_area * 0.00005 or area > image_area * 0.04:
            continue
        rect = cv2.minAreaRect(contour)
        rect_w, rect_h = rect[1]
        if rect_w <= 1 or rect_h <= 1:
            continue
        aspect = max(rect_w, rect_h) / max(min(rect_w, rect_h), 1.0)
        if not 0.65 <= aspect <= 1.55:
            continue
        fill_ratio = area / max(rect_w * rect_h, 1.0)
        if fill_ratio < 0.45:
            continue
        x, y, box_w, box_h = cv2.boundingRect(contour)
        boxes.append((float(x), float(y), float(box_w), float(box_h), area))

    if len(boxes) < 8:
        return None

    median_box_area = float(np.median([area for _, _, _, _, area in boxes]))
    boxes = [box for box in boxes if box[4] >= median_box_area * 0.35]
    median_patch_size = float(np.median([min(box_w, box_h) for _, _, box_w, box_h, _ in boxes]))
    boxes = _keep_largest_axis_cluster(boxes, axis=0, max_gap=median_patch_size * 2.2)
    boxes = _keep_largest_axis_cluster(boxes, axis=1, max_gap=median_patch_size * 2.2)
    if len(boxes) < 8:
        return None

    candidates = [
        _fit_patch_grid_detection(
            boxes,
            column_count=6,
            row_count=4,
            x_fractions=np.linspace(0.13, 0.87, 6, dtype=np.float32),
            y_fractions=np.linspace(0.12, 0.88, 4, dtype=np.float32),
            image_shape=patch_mask.shape,
            scale=scale,
            debug_mask=patch_mask,
            method="automatic patch-grid detection",
        ),
        _fit_patch_grid_detection(
            boxes,
            column_count=4,
            row_count=6,
            x_fractions=np.linspace(0.17, 0.83, 4, dtype=np.float32),
            y_fractions=np.linspace(0.15, 0.85, 6, dtype=np.float32),
            image_shape=patch_mask.shape,
            scale=scale,
            debug_mask=patch_mask,
            method="automatic rotated patch-grid detection",
        ),
    ]
    candidates = [candidate for candidate in candidates if candidate is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate.score)


def _keep_largest_axis_cluster(
    boxes: list[tuple[float, float, float, float, float]],
    axis: int,
    max_gap: float,
) -> list[tuple[float, float, float, float, float]]:
    if len(boxes) < 2:
        return boxes
    centers = np.array(
        [box[axis] + box[axis + 2] * 0.5 for box in boxes],
        dtype=np.float32,
    )
    order = np.argsort(centers)
    ordered_centers = centers[order]
    split_after = np.flatnonzero(np.diff(ordered_centers) > max_gap) + 1
    groups = np.split(order, split_after)
    largest_group = max(groups, key=len)
    return [boxes[int(index)] for index in np.sort(largest_group)]


def _fit_patch_grid_detection(
    boxes: list[tuple[float, float, float, float, float]],
    column_count: int,
    row_count: int,
    x_fractions: np.ndarray,
    y_fractions: np.ndarray,
    image_shape: tuple[int, int],
    scale: float,
    debug_mask: np.ndarray,
    method: str,
) -> ColorCheckerDetection | None:
    centers = np.array(
        [[x + box_w * 0.5, y + box_h * 0.5] for x, y, box_w, box_h, _ in boxes],
        dtype=np.float32,
    )
    if centers.shape[0] < max(column_count, row_count):
        return None

    x_centers = _cluster_1d(centers[:, 0], column_count)
    y_centers = _cluster_1d(centers[:, 1], row_count)
    x_spacing = np.diff(x_centers)
    y_spacing = np.diff(y_centers)
    if x_spacing.size == 0 or y_spacing.size == 0:
        return None
    if np.min(x_spacing) <= 0 or np.min(y_spacing) <= 0:
        return None

    median_x_spacing = float(np.median(x_spacing))
    median_y_spacing = float(np.median(y_spacing))
    x_regularity = float(np.min(x_spacing) / max(np.max(x_spacing), 1.0))
    y_regularity = float(np.min(y_spacing) / max(np.max(y_spacing), 1.0))
    if min(x_regularity, y_regularity) < 0.35:
        return None

    patch_sizes = np.array([min(box_w, box_h) for _, _, box_w, box_h, _ in boxes], dtype=np.float32)
    median_patch_size = float(np.median(patch_sizes))
    max_x_error = max(median_patch_size * 0.85, median_x_spacing * 0.32)
    max_y_error = max(median_patch_size * 0.85, median_y_spacing * 0.32)

    x_labels = np.argmin(np.abs(centers[:, 0, None] - x_centers[None, :]), axis=1)
    y_labels = np.argmin(np.abs(centers[:, 1, None] - y_centers[None, :]), axis=1)
    dx = np.abs(centers[:, 0] - x_centers[x_labels])
    dy = np.abs(centers[:, 1] - y_centers[y_labels])
    close = (dx <= max_x_error) & (dy <= max_y_error)

    occupied_cells = {(int(col), int(row)) for col, row in zip(x_labels[close], y_labels[close])}
    occupied_columns = {col for col, _ in occupied_cells}
    occupied_rows = {row for _, row in occupied_cells}
    cell_count = column_count * row_count
    if len(occupied_cells) < max(8, int(round(cell_count * 0.45))):
        return None
    if len(occupied_columns) < max(3, column_count - 1):
        return None
    if len(occupied_rows) < max(3, row_count - 1):
        return None

    source_points: list[list[float]] = []
    observed_points: list[list[float]] = []
    for index in np.flatnonzero(close):
        source_points.append([float(x_fractions[x_labels[index]]), float(y_fractions[y_labels[index]]), 1.0])
        observed_points.append([float(centers[index, 0]), float(centers[index, 1])])

    source = np.asarray(source_points, dtype=np.float32)
    observed = np.asarray(observed_points, dtype=np.float32)
    if np.linalg.matrix_rank(source) < 3:
        return None

    affine_x, _, _, _ = np.linalg.lstsq(source, observed[:, 0], rcond=None)
    affine_y, _, _, _ = np.linalg.lstsq(source, observed[:, 1], rcond=None)
    chart_points = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    corners_small = np.column_stack((chart_points @ affine_x, chart_points @ affine_y)).astype(np.float32)

    chart_area = float(cv2.contourArea(corners_small))
    image_area = float(image_shape[0] * image_shape[1])
    if chart_area <= image_area * 0.01 or chart_area >= image_area * 0.82:
        return None

    side_lengths = [
        float(np.linalg.norm(corners_small[(index + 1) % 4] - corners_small[index]))
        for index in range(4)
    ]
    long_side = max(side_lengths)
    short_side = max(min(side_lengths), 1.0)
    aspect = long_side / short_side
    if not 1.15 <= aspect <= 1.95:
        return None

    prediction = np.column_stack((source @ affine_x, source @ affine_y))
    residual = np.median(np.linalg.norm(observed - prediction, axis=1) / max(median_patch_size, 1.0))
    occupancy_score = len(occupied_cells) / float(cell_count)
    regularity_score = x_regularity * y_regularity
    score = (
        chart_area
        * (occupancy_score ** 2)
        * regularity_score
        / (1.0 + float(residual))
    )

    return ColorCheckerDetection(
        corners=order_corners(corners_small / scale),
        score=float(score),
        method=method,
        debug_mask=debug_mask,
    )


def warp_colorchecker(
    image_rgb: np.ndarray,
    corners: np.ndarray,
    output_size: tuple[int, int] = (1200, 800),
) -> tuple[np.ndarray, np.ndarray]:
    """Perspective-warp the chart into a canonical rectangle."""
    ordered = order_corners(corners).astype(np.float32)
    width, height = output_size
    destination = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(ordered, destination)
    warped = cv2.warpPerspective(
        as_uint8(image_rgb),
        transform,
        output_size,
        flags=cv2.INTER_LINEAR,
    )
    return warped.astype(np.float32) / 255.0, transform


def sample_colorchecker_patches(
    warped_chart_rgb: np.ndarray,
    sample_fraction: float = 0.45,
    column_count: int = 6,
    row_count: int = 4,
) -> np.ndarray:
    """Sample median RGB values from a canonical ColorChecker patch grid."""
    height, width = warped_chart_rgb.shape[:2]

    image_u8 = as_uint8(warped_chart_rgb)
    hsv = cv2.cvtColor(image_u8, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(image_u8, cv2.COLOR_RGB2GRAY)

    threshold = (((gray > 28) & (hsv[:, :, 2] > 28)) | (hsv[:, :, 1] > 35)).astype(np.uint8) * 255
    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
    )
    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_CLOSE,
        np.ones((7, 7), dtype=np.uint8),
    )

    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    image_area = float(height * width)
    for contour in contours:
        x, y, box_w, box_h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect = box_w / max(box_h, 1)
        if area < image_area * 0.004 or area > image_area * 0.05:
            continue
        if not 0.65 <= aspect <= 1.35:
            continue
        boxes.append((x, y, box_w, box_h))

    if len(boxes) >= max(12, int(round(column_count * row_count * 0.5))):
        x_centers = _cluster_1d(
            np.array([x + box_w / 2.0 for x, _, box_w, _ in boxes], dtype=np.float32),
            column_count,
        )
        y_centers = _cluster_1d(
            np.array([y + box_h / 2.0 for _, y, _, box_h in boxes], dtype=np.float32),
            row_count,
        )
        patch_size = float(np.median([min(box_w, box_h) for _, _, box_w, box_h in boxes]))
    else:
        x_margin = 0.13 if column_count == 6 else 0.17 if column_count == 4 else 0.5 / column_count
        y_margin = 0.12 if row_count == 4 else 0.15 if row_count == 6 else 0.5 / row_count
        x_centers = np.linspace(width * x_margin, width * (1.0 - x_margin), column_count, dtype=np.float32)
        y_centers = np.linspace(height * y_margin, height * (1.0 - y_margin), row_count, dtype=np.float32)
        patch_size = min(width / (column_count + 1.5), height / (row_count + 1.5))

    sample_half = patch_size * sample_fraction * 0.5
    patch_values: list[np.ndarray] = []
    for center_y in y_centers:
        for center_x in x_centers:
            x0 = max(0, int(round(center_x - sample_half)))
            x1 = min(width, int(round(center_x + sample_half)))
            y0 = max(0, int(round(center_y - sample_half)))
            y1 = min(height, int(round(center_y + sample_half)))
            patch = warped_chart_rgb[y0:y1, x0:x1]
            patch_values.append(np.median(patch.reshape(-1, 3), axis=0))
    return np.asarray(patch_values, dtype=np.float32)


def _cluster_1d(values: np.ndarray, cluster_count: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        raise ValueError("Cannot cluster empty values.")
    centers = np.linspace(float(values.min()), float(values.max()), cluster_count, dtype=np.float32)
    for _ in range(25):
        distances = np.abs(values[:, None] - centers[None, :])
        labels = np.argmin(distances, axis=1)
        updated = centers.copy()
        for index in range(cluster_count):
            group = values[labels == index]
            if group.size:
                updated[index] = float(group.mean())
        if np.allclose(updated, centers):
            break
        centers = updated
    return np.sort(centers)


def corners_to_bbox(corners: np.ndarray) -> tuple[int, int, int, int]:
    ordered = order_corners(corners)
    xmin = int(np.floor(np.min(ordered[:, 0])))
    ymin = int(np.floor(np.min(ordered[:, 1])))
    xmax = int(np.ceil(np.max(ordered[:, 0])))
    ymax = int(np.ceil(np.max(ordered[:, 1])))
    return xmin, ymin, xmax, ymax
