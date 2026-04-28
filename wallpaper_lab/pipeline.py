from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wallpaper_lab.calibration import (
    CalibrationResult,
    calibrate_image_from_colorchecker,
    calibration_diagnostics_dataframe,
    rgb_to_lab,
)
from wallpaper_lab.roi import (
    PolygonOperation,
    apply_polygon_operations,
    build_fragment_roi_masks,
    build_manual_roi_masks,
    compute_lab_statistics,
    polygon_to_mask,
)
from wallpaper_lab.segmentation import (
    PigmentProfile,
    SegmentationParameters,
    build_fragment_support_mask,
    build_pigment_mask_result,
    get_pigment_profile,
)
from wallpaper_lab.visualization import overlay_mask


@dataclass
class AnalysisResult:
    pigment_profile: PigmentProfile
    calibration: CalibrationResult
    lab_image: np.ndarray
    wallpaper_mask: np.ndarray
    fragment_labels: np.ndarray
    initial_pigment_mask: np.ndarray
    pigment_mask: np.ndarray
    material_masks: dict[str, np.ndarray]
    roi_masks: dict[str, np.ndarray]
    summary_df: pd.DataFrame
    calibration_df: pd.DataFrame
    segmentation_df: pd.DataFrame
    overlay_rgb: np.ndarray

    @property
    def initial_green_mask(self) -> np.ndarray:
        return self.initial_pigment_mask

    @property
    def green_mask(self) -> np.ndarray:
        return self.pigment_mask


def run_analysis(
    image_rgb: np.ndarray,
    chart_corners: np.ndarray,
    segmentation_params: SegmentationParameters,
    cleanup_operations: list[PolygonOperation] | None = None,
    roi_mode: str = "whole_pigment_mask",
    manual_roi_polygons: list[PolygonOperation] | None = None,
) -> AnalysisResult:
    cleanup_operations = cleanup_operations or []
    manual_roi_polygons = manual_roi_polygons or []
    pigment_profile = get_pigment_profile(segmentation_params.pigment_key)

    calibration = calibrate_image_from_colorchecker(image_rgb, chart_corners)
    lab_image = rgb_to_lab(calibration.corrected_rgb)
    mask_result = build_pigment_mask_result(calibration.corrected_rgb, chart_corners, segmentation_params)
    initial_pigment_mask = mask_result.pigment_mask
    material_support_mask = _material_support_mask(mask_result.material_masks)
    support_source_mask = material_support_mask if material_support_mask is not None else initial_pigment_mask
    wallpaper_mask, fragment_labels = build_fragment_support_mask(support_source_mask, segmentation_params)
    pigment_mask = apply_polygon_operations(initial_pigment_mask, cleanup_operations)
    if material_support_mask is not None:
        support_source_mask = apply_polygon_operations(material_support_mask, cleanup_operations)
    else:
        support_source_mask = pigment_mask
    wallpaper_mask, fragment_labels = build_fragment_support_mask(support_source_mask, segmentation_params)
    material_masks = _apply_cleanup_exclusions_to_material_masks(
        mask_result.material_masks,
        cleanup_operations,
    )

    if roi_mode == "per_fragment":
        roi_masks = build_fragment_roi_masks(fragment_labels, pigment_mask)
    elif roi_mode == "manual_polygon":
        roi_masks = build_manual_roi_masks(pigment_mask.shape, manual_roi_polygons, pigment_mask)
    else:
        roi_masks = {f"combined_{pigment_profile.roi_label}": pigment_mask}

    union_mask = np.zeros_like(pigment_mask, dtype=bool)
    for mask in roi_masks.values():
        union_mask |= mask
    if np.any(union_mask):
        roi_masks = {f"combined_{pigment_profile.roi_label}": union_mask, **roi_masks}

    summary_df = compute_lab_statistics(lab_image, roi_masks)
    calibration_df = calibration_diagnostics_dataframe(calibration)
    segmentation_df = _segmentation_diagnostics_dataframe(
        pigment_profile,
        segmentation_params,
        pigment_mask,
        material_masks,
    )
    overlay_rgb = overlay_mask(
        calibration.corrected_rgb,
        pigment_mask,
        color_rgb=pigment_profile.overlay_color_rgb,
    )

    return AnalysisResult(
        pigment_profile=pigment_profile,
        calibration=calibration,
        lab_image=lab_image,
        wallpaper_mask=wallpaper_mask,
        fragment_labels=fragment_labels,
        initial_pigment_mask=initial_pigment_mask,
        pigment_mask=pigment_mask,
        material_masks=material_masks,
        roi_masks=roi_masks,
        summary_df=summary_df,
        calibration_df=calibration_df,
        segmentation_df=segmentation_df,
        overlay_rgb=overlay_rgb,
    )


def _segmentation_diagnostics_dataframe(
    pigment_profile: PigmentProfile,
    params: SegmentationParameters,
    pigment_mask: np.ndarray,
    material_masks: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = [
        {
            "mask": pigment_profile.mask_file_stem,
            "role": "final_sample_mask",
            "material_mode": params.material_mode,
            "pixel_count": int(pigment_mask.sum()),
        }
    ]
    for mask_name, mask in material_masks.items():
        rows.append(
            {
                "mask": mask_name,
                "role": _mask_role(mask_name),
                "material_mode": params.material_mode,
                "pixel_count": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def _material_support_mask(material_masks: dict[str, np.ndarray]) -> np.ndarray | None:
    for mask_name, mask in material_masks.items():
        if mask_name.endswith("_cloth_support"):
            return mask
    return None


def _apply_cleanup_exclusions_to_material_masks(
    material_masks: dict[str, np.ndarray],
    cleanup_operations: list[PolygonOperation],
) -> dict[str, np.ndarray]:
    if not material_masks:
        return {}
    exclusion_mask = np.zeros(next(iter(material_masks.values())).shape, dtype=bool)
    for operation in cleanup_operations:
        if operation.operation == "exclude":
            exclusion_mask |= polygon_to_mask(exclusion_mask.shape, operation.points)
    if not np.any(exclusion_mask):
        return dict(material_masks)
    return {
        mask_name: mask & ~exclusion_mask
        for mask_name, mask in material_masks.items()
    }


def _mask_role(mask_name: str) -> str:
    if mask_name.endswith("_strict_colour_seed"):
        return "strict_colour_seed"
    if mask_name.endswith("_texture_smoothed_seed"):
        return "texture_smoothed_support_seed"
    if mask_name.endswith("_dyed_thread_seed"):
        return "dyed_thread_sample_seed"
    if mask_name.endswith("_thread_shadow"):
        return "local_thread_shadow_recovery"
    if mask_name.endswith("_substrate_showthrough"):
        return "local_substrate_showthrough_recovery"
    if mask_name.endswith("_woven_surface"):
        return "dyed_threads_shadows_and_substrate"
    if mask_name.endswith("_cloth_support"):
        return "local_cloth_support"
    return "material_component"
