from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

import numpy as np
import pandas as pd

from wallpaper_lab.io_utils import save_rgb_image
from wallpaper_lab.segmentation import DEFAULT_PIGMENT_PROFILE, PigmentProfile


def create_output_directory(base_dir: str | Path, image_stem: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"{image_stem}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_analysis_outputs(
    output_dir: str | Path,
    corrected_rgb: np.ndarray,
    pigment_mask: np.ndarray | None = None,
    overlay_rgb: np.ndarray | None = None,
    summary_df: pd.DataFrame | None = None,
    calibration_df: pd.DataFrame | None = None,
    segmentation_df: pd.DataFrame | None = None,
    extra_masks: dict[str, np.ndarray] | None = None,
    pigment_profile: PigmentProfile | None = None,
    green_mask: np.ndarray | None = None,
) -> dict[str, Path]:
    if pigment_mask is None:
        if green_mask is None:
            raise ValueError("save_analysis_outputs requires a pigment_mask.")
        pigment_mask = green_mask
    if overlay_rgb is None or summary_df is None:
        raise ValueError("save_analysis_outputs requires overlay_rgb and summary_df.")

    pigment_profile = pigment_profile or DEFAULT_PIGMENT_PROFILE
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    corrected_path = output_path / "corrected_image.png"
    pigment_mask_path = output_path / f"{pigment_profile.mask_file_stem}.png"
    overlay_path = output_path / "sampling_overlay.png"
    summary_path = output_path / "lab_summary.csv"
    calibration_path = output_path / "calibration_diagnostics.csv"
    segmentation_path = output_path / "segmentation_diagnostics.csv"

    save_rgb_image(corrected_path, corrected_rgb)
    save_rgb_image(pigment_mask_path, np.dstack([pigment_mask] * 3).astype(np.float32))
    save_rgb_image(overlay_path, overlay_rgb)
    summary_df.to_csv(summary_path, index=False)
    if calibration_df is not None:
        calibration_df.to_csv(calibration_path, index=False)
    if segmentation_df is not None:
        segmentation_df.to_csv(segmentation_path, index=False)

    files = {
        "corrected_image": corrected_path,
        pigment_profile.mask_file_stem: pigment_mask_path,
        "sampling_overlay": overlay_path,
        "lab_summary_csv": summary_path,
    }
    if calibration_df is not None:
        files["calibration_diagnostics_csv"] = calibration_path
    if segmentation_df is not None:
        files["segmentation_diagnostics_csv"] = segmentation_path

    for mask_name, mask in (extra_masks or {}).items():
        mask_stem = _safe_file_stem(mask_name)
        mask_path = output_path / f"{mask_stem}.png"
        save_rgb_image(mask_path, np.dstack([mask] * 3).astype(np.float32))
        files[mask_stem] = mask_path
    return files


def _safe_file_stem(stem: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem.strip())
    return normalized.strip("._") or "mask"
