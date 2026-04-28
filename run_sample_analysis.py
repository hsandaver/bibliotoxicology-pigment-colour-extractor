from __future__ import annotations

import argparse
import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
CACHE_ROOT = APP_ROOT / ".cache"
(CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))

from wallpaper_lab.colorchecker import detect_colorchecker
from wallpaper_lab.export import create_output_directory, save_analysis_outputs
from wallpaper_lab.io_utils import load_rgb_image
from wallpaper_lab.pipeline import run_analysis
from wallpaper_lab.references import SAMPLE_IMAGE_PATH, SAMPLE_MANUAL_COLORCHECKER_CORNERS
from wallpaper_lab.segmentation import PIGMENT_PROFILES, SegmentationParameters, get_pigment_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the pigment colour Lab extraction pipeline without the Streamlit UI.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=SAMPLE_IMAGE_PATH,
        help="Path to the source image.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Directory under which a timestamped export folder will be created.",
    )
    parser.add_argument(
        "--chart-mode",
        choices=["auto", "sample-default"],
        default="auto",
        help="How to obtain ColorChecker corners.",
    )
    parser.add_argument(
        "--pigment",
        default="green",
        help=(
            "Pigment extractor to use. Choices: "
            f"{', '.join(PIGMENT_PROFILES)}. Common aliases such as 'vermillion' and 'bookcloth' "
            "are also accepted."
        ),
    )
    parser.add_argument(
        "--material-workflow",
        choices=["auto", "printed", "woven_cloth"],
        default="auto",
        help="Mask workflow. Auto uses the selected extractor's default material workflow.",
    )
    parser.add_argument(
        "--cloth-sample-mode",
        choices=["dyed_threads", "dyed_threads_with_shadows", "surface_appearance"],
        default="dyed_threads",
        help="For woven_cloth workflow, choose which cloth components are included in Lab statistics.",
    )
    parser.add_argument(
        "--roi-mode",
        choices=["whole_pigment_mask", "whole_green_mask", "per_fragment"],
        default="per_fragment",
        help="ROI summary mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        pigment_profile = get_pigment_profile(args.pigment)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    image_rgb = load_rgb_image(args.image)

    if args.chart_mode == "sample-default":
        chart_corners = SAMPLE_MANUAL_COLORCHECKER_CORNERS.copy()
        chart_source = "sample-default"
    else:
        detection = detect_colorchecker(image_rgb)
        if detection is not None:
            chart_corners = detection.corners
            chart_source = detection.method
        else:
            chart_corners = SAMPLE_MANUAL_COLORCHECKER_CORNERS.copy()
            chart_source = "sample-default fallback"

    material_mode = (
        pigment_profile.material_mode
        if args.material_workflow == "auto"
        else args.material_workflow
    )
    analysis = run_analysis(
        image_rgb=image_rgb,
        chart_corners=chart_corners,
        segmentation_params=SegmentationParameters.from_profile(
            pigment_profile,
            material_mode=material_mode,
            cloth_sample_mode=args.cloth_sample_mode,
        ),
        roi_mode=args.roi_mode,
    )

    output_dir = create_output_directory(args.output_root, args.image.stem)
    files = save_analysis_outputs(
        output_dir=output_dir,
        corrected_rgb=analysis.calibration.corrected_rgb,
        pigment_mask=analysis.pigment_mask,
        overlay_rgb=analysis.overlay_rgb,
        summary_df=analysis.summary_df,
        calibration_df=analysis.calibration_df,
        segmentation_df=analysis.segmentation_df,
        extra_masks=analysis.material_masks,
        pigment_profile=analysis.pigment_profile,
    )

    print(f"Image: {args.image}")
    print(f"Pigment extractor: {analysis.pigment_profile.label}")
    print(f"Material workflow: {material_mode}")
    if material_mode == "woven_cloth":
        print(f"Cloth Lab sampling basis: {args.cloth_sample_mode}")
    print(f"Chart registration: {chart_source}")
    print(f"Mean chart Delta E00 before: {analysis.calibration.mean_delta_e_00_before:.2f}")
    print(f"Mean chart Delta E00 after: {analysis.calibration.mean_delta_e_00_after:.2f}")
    print(f"Mean chart Delta E76 after: {analysis.calibration.mean_delta_e_76_after:.2f}")
    print(f"Selected pixel count: {int(analysis.pigment_mask.sum())}")
    print()
    if analysis.summary_df.empty:
        print("No pixels were selected.")
    else:
        print(analysis.summary_df.to_string(index=False))
    print()
    print("Saved files:")
    for label, file_path in files.items():
        print(f"  {label}: {file_path}")


if __name__ == "__main__":
    main()
