from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage import color

from wallpaper_lab.colorchecker import sample_colorchecker_patches, warp_colorchecker
from wallpaper_lab.color_metrics import delta_e_cie1976
from wallpaper_lab.references import CALIBRITE_DISPLAY_PATCH_NAMES, CALIBRITE_DISPLAY_SRGB


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * np.power(rgb, 1 / 2.4) - 0.055)


def _smoothstep(edge0: float, edge1: float, values: np.ndarray) -> np.ndarray:
    if edge1 <= edge0:
        return (values >= edge1).astype(np.float32)
    t = np.clip((values - edge0) / (edge1 - edge0), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _luminance(rgb_linear: np.ndarray) -> np.ndarray:
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return rgb_linear @ weights


def _channel_spread(rgb_linear: np.ndarray) -> np.ndarray:
    return np.ptp(rgb_linear, axis=-1)


def _fit_monotonic_curve(
    source_values: np.ndarray,
    target_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.concatenate(([0.0], np.asarray(source_values, dtype=np.float32), [1.0]))
    target = np.concatenate(([0.0], np.asarray(target_values, dtype=np.float32), [1.0]))

    order = np.argsort(source)
    source = source[order]
    target = target[order]

    unique_source = np.unique(source)
    averaged_target = np.array([target[source == value].mean() for value in unique_source], dtype=np.float32)
    monotonic_target = np.maximum.accumulate(averaged_target)
    return unique_source.astype(np.float32), np.clip(monotonic_target, 0.0, 1.0).astype(np.float32)


def _identify_neutral_patch_indices(reference_patches_rgb: np.ndarray, count: int = 6) -> np.ndarray:
    channel_spread = np.ptp(reference_patches_rgb, axis=1)
    luminance = _luminance(reference_patches_rgb.astype(np.float32))
    candidate_indices = np.argsort(channel_spread)[:count]
    return candidate_indices[np.argsort(luminance[candidate_indices])].astype(np.int32)


def _apply_black_anchored_matrix(
    rgb_linear: np.ndarray,
    matrix: np.ndarray,
    observed_black_linear: np.ndarray,
    reference_black_linear: np.ndarray,
) -> np.ndarray:
    flat = np.asarray(rgb_linear, dtype=np.float32).reshape(-1, 3)
    anchored = np.clip(flat - observed_black_linear, 0.0, None)
    corrected = anchored @ matrix + reference_black_linear
    return np.clip(corrected.reshape(rgb_linear.shape), 0.0, 1.0).astype(np.float32)


@dataclass
class ShadowNeutralizationModel:
    curves: tuple[tuple[np.ndarray, np.ndarray], ...]
    shadow_full_luminance: float
    shadow_end_luminance: float
    chroma_full: float
    chroma_end: float

    def apply(self, rgb_linear: np.ndarray) -> np.ndarray:
        flat = np.clip(np.asarray(rgb_linear, dtype=np.float32).reshape(-1, 3), 0.0, 1.0)
        luminance = _luminance(flat)
        chroma = _channel_spread(flat)

        shadow_weight = 1.0 - _smoothstep(self.shadow_full_luminance, self.shadow_end_luminance, luminance)
        chroma_weight = 1.0 - _smoothstep(self.chroma_full, self.chroma_end, chroma)
        weight = (shadow_weight * chroma_weight).astype(np.float32)

        neutralized = flat.copy()
        for channel, (source_x, target_y) in enumerate(self.curves):
            neutralized[:, channel] = np.interp(flat[:, channel], source_x, target_y).astype(np.float32)

        blended = flat + weight[:, None] * (neutralized - flat)
        return np.clip(blended.reshape(rgb_linear.shape), 0.0, 1.0).astype(np.float32)


@dataclass
class CalibrationModel:
    matrix: np.ndarray
    observed_black_linear: np.ndarray
    reference_black_linear: np.ndarray
    shadow_neutralization: ShadowNeutralizationModel | None = None

    def apply(self, image_rgb: np.ndarray) -> np.ndarray:
        image_rgb = np.asarray(image_rgb, dtype=np.float32)
        if image_rgb.shape[-1] != 3:
            raise ValueError("CalibrationModel.apply expects an array whose last dimension is RGB.")

        image_linear = srgb_to_linear(image_rgb)
        corrected_linear = _apply_black_anchored_matrix(
            image_linear,
            self.matrix,
            self.observed_black_linear,
            self.reference_black_linear,
        )
        if self.shadow_neutralization is not None:
            corrected_linear = self.shadow_neutralization.apply(corrected_linear)

        corrected = linear_to_srgb(corrected_linear)
        return np.clip(corrected, 0.0, 1.0).astype(np.float32)


def _fit_shadow_neutralization_model(
    base_corrected_linear: np.ndarray,
    reference_linear: np.ndarray,
    neutral_indices: np.ndarray,
) -> ShadowNeutralizationModel | None:
    if neutral_indices.size < 3:
        return None

    curves = tuple(
        _fit_monotonic_curve(
            base_corrected_linear[neutral_indices, channel],
            reference_linear[neutral_indices, channel],
        )
        for channel in range(3)
    )

    neutral_luminance = _luminance(reference_linear[neutral_indices])
    shadow_full_luminance = float(neutral_luminance[min(2, neutral_luminance.size - 1)])
    shadow_end_luminance = float(neutral_luminance[min(4, neutral_luminance.size - 1)])

    neutral_spread = _channel_spread(base_corrected_linear[neutral_indices])
    color_mask = np.ones(base_corrected_linear.shape[0], dtype=bool)
    color_mask[neutral_indices] = False
    color_spread = _channel_spread(base_corrected_linear[color_mask])

    chroma_full = float(max(0.02, np.quantile(neutral_spread, 0.75) + 0.01))
    if color_spread.size:
        chroma_end = float(max(chroma_full * 2.5, np.quantile(color_spread, 0.25)))
    else:
        chroma_end = chroma_full + 0.05

    chroma_end = max(chroma_end, chroma_full + 1e-3)
    shadow_end_luminance = max(shadow_end_luminance, shadow_full_luminance + 1e-3)

    return ShadowNeutralizationModel(
        curves=curves,
        shadow_full_luminance=shadow_full_luminance,
        shadow_end_luminance=shadow_end_luminance,
        chroma_full=chroma_full,
        chroma_end=chroma_end,
    )


@dataclass
class CalibrationResult:
    corrected_rgb: np.ndarray
    warped_chart_rgb: np.ndarray
    corrected_chart_rgb: np.ndarray
    chart_rotation_k: int
    chart_layout: str
    patch_grid_shape: tuple[int, int]
    observed_patches_rgb: np.ndarray
    corrected_patches_rgb: np.ndarray
    reference_patches_rgb: np.ndarray
    reference_patch_names: tuple[str, ...]
    patch_delta_e_before: np.ndarray
    patch_delta_e_after: np.ndarray
    patch_delta_e_76_before: np.ndarray
    patch_delta_e_76_after: np.ndarray
    model: CalibrationModel

    @property
    def mean_delta_e_before(self) -> float:
        return float(np.mean(self.patch_delta_e_before))

    @property
    def mean_delta_e_after(self) -> float:
        return float(np.mean(self.patch_delta_e_after))

    @property
    def mean_delta_e_00_before(self) -> float:
        return self.mean_delta_e_before

    @property
    def mean_delta_e_00_after(self) -> float:
        return self.mean_delta_e_after

    @property
    def mean_delta_e_76_before(self) -> float:
        return float(np.mean(self.patch_delta_e_76_before))

    @property
    def mean_delta_e_76_after(self) -> float:
        return float(np.mean(self.patch_delta_e_76_after))


def fit_calibration_model(
    observed_patches_rgb: np.ndarray,
    reference_patches_rgb: np.ndarray,
) -> CalibrationModel:
    observed_linear = srgb_to_linear(observed_patches_rgb)
    reference_linear = srgb_to_linear(reference_patches_rgb)
    black_index = int(np.argmin(_luminance(reference_linear)))
    neutral_indices = _identify_neutral_patch_indices(reference_patches_rgb)

    observed_black_linear = observed_linear[black_index]
    reference_black_linear = reference_linear[black_index]

    design = np.clip(observed_linear - observed_black_linear, 0.0, None)
    targets = np.clip(reference_linear - reference_black_linear, 0.0, None)
    matrix, _, _, _ = np.linalg.lstsq(design, targets, rcond=None)

    base_corrected_linear = _apply_black_anchored_matrix(
        observed_linear,
        matrix.astype(np.float32),
        observed_black_linear.astype(np.float32),
        reference_black_linear.astype(np.float32),
    )
    shadow_neutralization = _fit_shadow_neutralization_model(
        base_corrected_linear,
        reference_linear,
        neutral_indices,
    )

    return CalibrationModel(
        matrix=matrix.astype(np.float32),
        observed_black_linear=observed_black_linear.astype(np.float32),
        reference_black_linear=reference_black_linear.astype(np.float32),
        shadow_neutralization=shadow_neutralization,
    )


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    return color.rgb2lab(np.clip(rgb, 0.0, 1.0))


def orient_colorchecker_to_reference(
    warped_chart_rgb: np.ndarray,
    reference_patches_rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Rotate a warped chart so sampled patches align with the reference patch order."""
    reference_lab = rgb_to_lab(reference_patches_rgb.reshape(1, -1, 3)).reshape(-1, 3)

    best_chart_rgb = warped_chart_rgb
    best_patches_rgb = sample_colorchecker_patches(warped_chart_rgb)
    best_rotation_k = 0
    best_score = np.inf

    for rotation_k in range(4):
        candidate_chart_rgb = np.rot90(warped_chart_rgb, k=rotation_k).copy()
        candidate_patches_rgb = sample_colorchecker_patches(candidate_chart_rgb)
        candidate_lab = rgb_to_lab(candidate_patches_rgb.reshape(1, -1, 3)).reshape(-1, 3)
        candidate_score = float(np.mean(color.deltaE_ciede2000(candidate_lab, reference_lab)))
        if candidate_score < best_score:
            best_chart_rgb = candidate_chart_rgb
            best_patches_rgb = candidate_patches_rgb
            best_rotation_k = rotation_k
            best_score = candidate_score

    return best_chart_rgb, best_patches_rgb, best_rotation_k


@dataclass(frozen=True)
class ColorCheckerLayout:
    name: str
    rotation_k: int
    column_count: int
    row_count: int
    output_size: tuple[int, int]
    reference_patches_rgb: np.ndarray
    reference_patch_names: tuple[str, ...]


@dataclass(frozen=True)
class ColorCheckerLayoutFit:
    layout: ColorCheckerLayout
    warped_chart_rgb: np.ndarray
    observed_patches_rgb: np.ndarray
    mean_delta_e: float


def _rotated_grid(values: np.ndarray, rotation_k: int) -> np.ndarray:
    array = np.asarray(values)
    grid = array.reshape(4, 6, *array.shape[1:])
    return np.rot90(grid, k=rotation_k)


def _rotated_names(names: list[str] | tuple[str, ...], rotation_k: int) -> tuple[str, ...]:
    grid = np.asarray(names, dtype=object).reshape(4, 6)
    return tuple(str(name) for name in np.rot90(grid, k=rotation_k).reshape(-1))


def _layout_candidates(
    reference_patches_rgb: np.ndarray,
    reference_patch_names: list[str] | tuple[str, ...],
) -> tuple[ColorCheckerLayout, ...]:
    layouts: list[ColorCheckerLayout] = []
    for rotation_k in range(4):
        reference_grid = _rotated_grid(reference_patches_rgb, rotation_k)
        row_count, column_count = reference_grid.shape[:2]
        output_size = (1200, 800) if column_count >= row_count else (800, 1200)
        layouts.append(
            ColorCheckerLayout(
                name=f"{column_count}x{row_count}_rotation_{rotation_k}",
                rotation_k=rotation_k,
                column_count=int(column_count),
                row_count=int(row_count),
                output_size=output_size,
                reference_patches_rgb=reference_grid.reshape(-1, 3).astype(np.float32),
                reference_patch_names=_rotated_names(reference_patch_names, rotation_k),
            )
        )
    return tuple(layouts)


def _score_patch_fit(observed_patches_rgb: np.ndarray, reference_patches_rgb: np.ndarray) -> float:
    observed_lab = rgb_to_lab(observed_patches_rgb.reshape(1, -1, 3)).reshape(-1, 3)
    reference_lab = rgb_to_lab(reference_patches_rgb.reshape(1, -1, 3)).reshape(-1, 3)
    return float(np.mean(color.deltaE_ciede2000(observed_lab, reference_lab)))


def _select_colorchecker_layout(
    image_rgb: np.ndarray,
    chart_corners: np.ndarray,
    reference_patches_rgb: np.ndarray,
    reference_patch_names: list[str] | tuple[str, ...],
) -> ColorCheckerLayoutFit:
    best_fit: ColorCheckerLayoutFit | None = None
    for layout in _layout_candidates(reference_patches_rgb, reference_patch_names):
        warped_chart_rgb, _ = warp_colorchecker(image_rgb, chart_corners, output_size=layout.output_size)
        observed_patches_rgb = sample_colorchecker_patches(
            warped_chart_rgb,
            column_count=layout.column_count,
            row_count=layout.row_count,
        )
        mean_delta_e = _score_patch_fit(observed_patches_rgb, layout.reference_patches_rgb)
        fit = ColorCheckerLayoutFit(
            layout=layout,
            warped_chart_rgb=warped_chart_rgb,
            observed_patches_rgb=observed_patches_rgb,
            mean_delta_e=mean_delta_e,
        )
        if best_fit is None or fit.mean_delta_e < best_fit.mean_delta_e:
            best_fit = fit

    if best_fit is None:
        raise ValueError("Could not sample ColorChecker patches for calibration.")
    return best_fit


def calibrate_image_from_colorchecker(
    image_rgb: np.ndarray,
    chart_corners: np.ndarray,
    reference_patches_rgb: np.ndarray = CALIBRITE_DISPLAY_SRGB,
) -> CalibrationResult:
    layout_fit = _select_colorchecker_layout(
        image_rgb,
        chart_corners,
        reference_patches_rgb,
        CALIBRITE_DISPLAY_PATCH_NAMES,
    )
    layout = layout_fit.layout

    model = fit_calibration_model(layout_fit.observed_patches_rgb, layout.reference_patches_rgb)
    corrected_rgb = model.apply(image_rgb)
    corrected_chart_rgb, _ = warp_colorchecker(corrected_rgb, chart_corners, output_size=layout.output_size)
    corrected_patches_rgb = sample_colorchecker_patches(
        corrected_chart_rgb,
        column_count=layout.column_count,
        row_count=layout.row_count,
    )

    reference_lab = rgb_to_lab(layout.reference_patches_rgb.reshape(1, -1, 3)).reshape(-1, 3)
    observed_lab = rgb_to_lab(layout_fit.observed_patches_rgb.reshape(1, -1, 3)).reshape(-1, 3)
    corrected_lab = rgb_to_lab(corrected_patches_rgb.reshape(1, -1, 3)).reshape(-1, 3)

    patch_delta_e_before = color.deltaE_ciede2000(observed_lab, reference_lab)
    patch_delta_e_after = color.deltaE_ciede2000(corrected_lab, reference_lab)
    patch_delta_e_76_before = delta_e_cie1976(observed_lab, reference_lab)
    patch_delta_e_76_after = delta_e_cie1976(corrected_lab, reference_lab)

    return CalibrationResult(
        corrected_rgb=corrected_rgb,
        warped_chart_rgb=layout_fit.warped_chart_rgb,
        corrected_chart_rgb=corrected_chart_rgb,
        chart_rotation_k=layout.rotation_k,
        chart_layout=layout.name,
        patch_grid_shape=(layout.row_count, layout.column_count),
        observed_patches_rgb=layout_fit.observed_patches_rgb,
        corrected_patches_rgb=corrected_patches_rgb,
        reference_patches_rgb=layout.reference_patches_rgb,
        reference_patch_names=layout.reference_patch_names,
        patch_delta_e_before=patch_delta_e_before.astype(np.float32),
        patch_delta_e_after=patch_delta_e_after.astype(np.float32),
        patch_delta_e_76_before=patch_delta_e_76_before.astype(np.float32),
        patch_delta_e_76_after=patch_delta_e_76_after.astype(np.float32),
        model=model,
    )


def calibration_diagnostics_dataframe(calibration: CalibrationResult):
    import pandas as pd

    return pd.DataFrame(
        {
            "patch": calibration.reference_patch_names,
            "delta_e_00_before": calibration.patch_delta_e_before,
            "delta_e_00_after": calibration.patch_delta_e_after,
            "delta_e_76_before": calibration.patch_delta_e_76_before,
            "delta_e_76_after": calibration.patch_delta_e_76_after,
            "observed_r": calibration.observed_patches_rgb[:, 0],
            "observed_g": calibration.observed_patches_rgb[:, 1],
            "observed_b": calibration.observed_patches_rgb[:, 2],
            "corrected_r": calibration.corrected_patches_rgb[:, 0],
            "corrected_g": calibration.corrected_patches_rgb[:, 1],
            "corrected_b": calibration.corrected_patches_rgb[:, 2],
        }
    )
