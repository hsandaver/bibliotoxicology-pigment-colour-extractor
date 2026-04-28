from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

APP_ROOT = Path(__file__).resolve().parent
CACHE_ROOT = APP_ROOT / ".cache"
(CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))

from wallpaper_lab.colorchecker import corners_to_bbox, detect_colorchecker, order_corners
from wallpaper_lab.export import create_output_directory, save_analysis_outputs
from wallpaper_lab.io_utils import (
    SUPPORTED_UPLOAD_FILE_TYPES,
    crop_with_padding,
    load_rgb_image,
    resize_to_long_edge,
)
from wallpaper_lab.pipeline import run_analysis
from wallpaper_lab.references import (
    SAMPLE_IMAGE_PATH,
    SAMPLE_MANUAL_COLORCHECKER_CORNERS,
)
from wallpaper_lab.roi import PolygonOperation
from wallpaper_lab.segmentation import PIGMENT_PROFILES, SegmentationParameters, get_pigment_profile
from wallpaper_lab.visualization import (
    create_lab_distribution_figure,
    draw_colorchecker_overlay,
    draw_points,
    draw_polygon_overlays,
    overlay_mask,
)


APP_TITLE = "Pigment and Cloth Colour Lab Extractor"


st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
)


def initialise_state() -> None:
    defaults = {
        "image_signature": None,
        "chart_points": [],
        "chart_last_click": None,
        "cleanup_points": [],
        "cleanup_last_click": None,
        "cleanup_polygons": [],
        "roi_points": [],
        "roi_last_click": None,
        "roi_polygons": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_interaction_state() -> None:
    for key in [
        "chart_points",
        "chart_last_click",
        "cleanup_points",
        "cleanup_last_click",
        "cleanup_polygons",
        "roi_points",
        "roi_last_click",
        "roi_polygons",
    ]:
        st.session_state[key] = [] if key.endswith("points") or key.endswith("polygons") else None


def read_uploaded_image(uploaded_file) -> np.ndarray:
    return load_rgb_image(BytesIO(uploaded_file.getvalue()), source_name=uploaded_file.name)


def append_click(
    event: dict | None,
    points_key: str,
    last_click_key: str,
    scale_to_original: float,
    offset_x: int = 0,
    offset_y: int = 0,
) -> None:
    if not event or "x" not in event or "y" not in event:
        return
    signature = (
        int(event["x"]),
        int(event["y"]),
        float(event.get("time", -1.0)),
    )
    if st.session_state[last_click_key] == signature:
        return

    point = (
        float(event["x"]) * scale_to_original + offset_x,
        float(event["y"]) * scale_to_original + offset_y,
    )
    if len(st.session_state[points_key]) < 4 or points_key != "chart_points":
        st.session_state[points_key].append(point)
    st.session_state[last_click_key] = signature


def build_chart_crop(image_rgb: np.ndarray, initial_corners: np.ndarray | None) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = image_rgb.shape[:2]
    if initial_corners is None:
        xmin = int(width * 0.58)
        xmax = width - 1
        ymin = int(height * 0.58)
        ymax = height - 1
        return crop_with_padding(image_rgb, xmin, ymin, xmax, ymax, padding=0.05)
    bbox = corners_to_bbox(initial_corners)
    return crop_with_padding(image_rgb, *bbox, padding=0.2)


def render_chart_clicker(image_rgb: np.ndarray, initial_corners: np.ndarray | None) -> np.ndarray | None:
    crop_image, origin = build_chart_crop(image_rgb, initial_corners)
    local_points = [
        (point[0] - origin[0], point[1] - origin[1])
        for point in st.session_state["chart_points"]
    ]
    preview = draw_points(crop_image, local_points, color_rgb=(255, 80, 80))
    display_image, scale = resize_to_long_edge(preview, 1100)
    scale_to_original = 1.0 / scale

    click = streamlit_image_coordinates(display_image, key="chart_clicker")
    append_click(click, "chart_points", "chart_last_click", scale_to_original, *origin)

    st.caption(
        "Click the ColorChecker corners in order: top-left, top-right, bottom-right, bottom-left. "
        "These points are used only for geometric chart registration."
    )
    action_cols = st.columns(4)
    if action_cols[0].button("Undo chart point"):
        if st.session_state["chart_points"]:
            st.session_state["chart_points"].pop()
    if action_cols[1].button("Reset chart points"):
        st.session_state["chart_points"] = []
    if action_cols[2].button("Use sample fallback corners"):
        st.session_state["chart_points"] = [tuple(point) for point in SAMPLE_MANUAL_COLORCHECKER_CORNERS.tolist()]
    if action_cols[3].button("Use auto-detected corners") and initial_corners is not None:
        st.session_state["chart_points"] = [tuple(point) for point in initial_corners.tolist()]

    if len(st.session_state["chart_points"]) == 4:
        corners = order_corners(np.array(st.session_state["chart_points"], dtype=np.float32))
        return corners
    return None


def render_polygon_editor(
    image_rgb: np.ndarray,
    state_prefix: str,
    current_operations: list[PolygonOperation],
    allow_operation_toggle: bool,
    default_name: str,
) -> list[PolygonOperation]:
    points_key = f"{state_prefix}_points"
    last_click_key = f"{state_prefix}_last_click"
    polygons_key = f"{state_prefix}_polygons"

    working_ops = list(current_operations)
    preview = draw_polygon_overlays(image_rgb, working_ops)
    preview = draw_points(preview, st.session_state[points_key], color_rgb=(255, 80, 80))
    display_image, scale = resize_to_long_edge(preview, 1200)
    scale_to_original = 1.0 / scale
    click = streamlit_image_coordinates(display_image, key=f"{state_prefix}_clicker")
    append_click(click, points_key, last_click_key, scale_to_original)

    cols = st.columns(4)
    if cols[0].button(f"Undo {state_prefix} point"):
        if st.session_state[points_key]:
            st.session_state[points_key].pop()
    if cols[1].button(f"Reset {state_prefix} points"):
        st.session_state[points_key] = []
    if cols[2].button(f"Remove last {state_prefix} polygon"):
        if st.session_state[polygons_key]:
            st.session_state[polygons_key].pop()
    if cols[3].button(f"Clear all {state_prefix} polygons"):
        st.session_state[polygons_key] = []
        st.session_state[points_key] = []

    operation = "include"
    if allow_operation_toggle:
        operation = st.selectbox(
            "Polygon operation",
            options=["exclude", "include"],
            index=0,
            key=f"{state_prefix}_operation",
        )
    polygon_name = st.text_input(
        "Polygon label",
        value=default_name,
        key=f"{state_prefix}_name",
    )
    if st.button(f"Finalize {state_prefix} polygon"):
        points = st.session_state[points_key]
        if len(points) >= 3:
            st.session_state[polygons_key].append(
                PolygonOperation(
                    name=polygon_name.strip() or default_name,
                    points=[tuple(point) for point in points],
                    operation=operation,
                )
            )
            st.session_state[points_key] = []

    return list(st.session_state[polygons_key])


def build_segmentation_parameters() -> SegmentationParameters:
    pigment_key = st.sidebar.selectbox(
        "Pigment extractor",
        options=list(PIGMENT_PROFILES),
        format_func=lambda value: PIGMENT_PROFILES[value].label,
    )
    profile = get_pigment_profile(pigment_key)
    material_options = ["printed", "woven_cloth"]
    material_mode = st.sidebar.selectbox(
        "Material workflow",
        options=material_options,
        index=material_options.index(profile.material_mode),
        format_func=lambda value: {
            "printed": "Wallpaper / printed pigment",
            "woven_cloth": "Woven cloth / bookbinding",
        }[value],
        key=f"{profile.key}_material_workflow_v2",
    )
    default_params = SegmentationParameters.from_profile(profile, material_mode=material_mode)
    if material_mode == "woven_cloth":
        st.sidebar.caption(
            "Texture-tolerant cloth extraction uses local dyed-thread support to recover thread shadows "
            "and substrate show-through without changing the wallpaper defaults."
        )

    st.sidebar.subheader(f"{profile.label} Mask Tuning")
    st.sidebar.caption(profile.description)
    hue_cols = st.sidebar.columns(2)
    hue_min = hue_cols[0].number_input(
        "Hue min",
        min_value=0.0,
        max_value=360.0,
        value=float(profile.hue_min),
        step=1.0,
        key=f"{profile.key}_hue_min",
    )
    hue_max = hue_cols[1].number_input(
        "Hue max",
        min_value=0.0,
        max_value=360.0,
        value=float(profile.hue_max),
        step=1.0,
        key=f"{profile.key}_hue_max",
    )
    if hue_min > hue_max:
        st.sidebar.caption("Hue range wraps through 0 degrees.")

    saturation_min = st.sidebar.slider(
        "Minimum HSV saturation",
        0.0,
        1.0,
        float(profile.saturation_min),
        0.005,
        key=f"{profile.key}_saturation_min",
    )
    a_range = st.sidebar.slider(
        "a* range",
        -90.0,
        90.0,
        (float(profile.a_min), float(profile.a_max)),
        0.5,
        key=f"{profile.key}_a_range",
    )
    b_range = st.sidebar.slider(
        "b* range",
        -110.0,
        110.0,
        (float(profile.b_min), float(profile.b_max)),
        0.5,
        key=f"{profile.key}_b_range",
    )
    chroma_min = st.sidebar.slider(
        "Minimum chroma",
        0.0,
        120.0,
        float(profile.chroma_min),
        0.5,
        key=f"{profile.key}_chroma_min",
    )
    lightness_range = st.sidebar.slider(
        "L* range",
        0.0,
        100.0,
        (float(profile.lightness_min), float(profile.lightness_max)),
        0.5,
        key=f"{profile.key}_lightness_range",
    )

    st.sidebar.subheader("Exclusion / Grouping")
    workflow_key = f"{profile.key}_{material_mode}"
    chart_padding_px = st.sidebar.slider(
        "ColorChecker exclusion padding (px)",
        0,
        300,
        int(default_params.chart_padding_px),
        5,
        key=f"{workflow_key}_chart_padding_px",
    )
    bottom_exclusion_fraction = st.sidebar.slider(
        "Ignore lower board zone below image fraction",
        0.70,
        0.98,
        float(default_params.bottom_exclusion_fraction),
        0.01,
        key=f"{workflow_key}_bottom_exclusion_fraction",
    )
    edge_exclusion_px = st.sidebar.slider(
        "Exclude pixels this far from support boundary",
        0,
        40,
        int(default_params.edge_exclusion_px),
        1,
        key=f"{workflow_key}_edge_exclusion_px",
    )
    min_area = st.sidebar.slider(
        "Minimum selected component area",
        1,
        500,
        int(default_params.min_area),
        1,
        key=f"{workflow_key}_min_area",
    )
    fragment_group_dilation_radius = st.sidebar.slider(
        "Fragment grouping radius (downscaled)",
        4,
        30,
        int(default_params.fragment_group_dilation_radius),
        1,
        key=f"{workflow_key}_fragment_group_dilation_radius",
    )

    cloth_kwargs = {}
    if material_mode == "woven_cloth":
        st.sidebar.subheader("Woven Cloth Controls")
        sample_mode_options = [
            "dyed_threads",
            "dyed_threads_with_shadows",
            "surface_appearance",
        ]
        cloth_kwargs = {
            "cloth_sample_mode": st.sidebar.selectbox(
                "Lab sampling basis",
                options=sample_mode_options,
                index=sample_mode_options.index(default_params.cloth_sample_mode),
                format_func=lambda value: {
                    "dyed_threads": "Dyed threads only",
                    "dyed_threads_with_shadows": "Dyed threads + thread shadows",
                    "surface_appearance": "Visible cloth surface",
                }[value],
                key=f"{workflow_key}_cloth_sample_mode",
            ),
            "cloth_smoothing_radius_px": st.sidebar.slider(
                "Texture smoothing radius (px)",
                0,
                20,
                int(default_params.cloth_smoothing_radius_px),
                1,
                key=f"{workflow_key}_cloth_smoothing_radius_px",
            ),
            "cloth_support_radius_px": st.sidebar.slider(
                "Local cloth support radius (px)",
                3,
                35,
                int(default_params.cloth_support_radius_px),
                1,
                key=f"{workflow_key}_cloth_support_radius_px",
            ),
            "cloth_min_seed_fraction": st.sidebar.slider(
                "Minimum local dyed-thread fraction",
                0.01,
                0.50,
                float(default_params.cloth_min_seed_fraction),
                0.01,
                key=f"{workflow_key}_cloth_min_seed_fraction",
            ),
            "cloth_shadow_lightness_tolerance": st.sidebar.slider(
                "Thread-shadow L* allowance",
                0.0,
                45.0,
                float(default_params.cloth_shadow_lightness_tolerance),
                1.0,
                key=f"{workflow_key}_cloth_shadow_lightness_tolerance",
            ),
            "cloth_ab_tolerance": st.sidebar.slider(
                "Thread-shadow a*/b* tolerance",
                0.0,
                24.0,
                float(default_params.cloth_ab_tolerance),
                0.5,
                key=f"{workflow_key}_cloth_ab_tolerance",
            ),
            "cloth_substrate_chroma_max": st.sidebar.slider(
                "Substrate show-through maximum chroma",
                0.0,
                30.0,
                float(default_params.cloth_substrate_chroma_max),
                0.5,
                key=f"{workflow_key}_cloth_substrate_chroma_max",
            ),
            "cloth_substrate_seed_fraction": st.sidebar.slider(
                "Substrate local dyed-thread fraction",
                0.01,
                0.75,
                float(default_params.cloth_substrate_seed_fraction),
                0.01,
                key=f"{workflow_key}_cloth_substrate_seed_fraction",
            ),
        }

    return SegmentationParameters(
        pigment_key=profile.key,
        material_mode=material_mode,
        hue_min=float(hue_min),
        hue_max=float(hue_max),
        saturation_min=saturation_min,
        a_min=float(a_range[0]),
        a_max=float(a_range[1]),
        b_min=float(b_range[0]),
        b_max=float(b_range[1]),
        chroma_min=chroma_min,
        lightness_min=float(lightness_range[0]),
        lightness_max=float(lightness_range[1]),
        chart_padding_px=chart_padding_px,
        bottom_exclusion_fraction=bottom_exclusion_fraction,
        edge_exclusion_px=edge_exclusion_px,
        min_area=min_area,
        fragment_group_dilation_radius=fragment_group_dilation_radius,
        **cloth_kwargs,
    )


def main() -> None:
    initialise_state()

    st.title(APP_TITLE)
    st.caption(
        "Calibrated image-derived CIE L*a*b* values for selected printed pigment and bookcloth colours. "
        "These outputs are image-based estimates derived from a ColorChecker-corrected photograph, "
        "not direct spectrophotometer measurements."
    )

    st.sidebar.header("Input")
    source_mode = st.sidebar.radio(
        "Image source",
        ["Use provided sample image", "Upload another image"],
    )

    image_rgb: np.ndarray | None = None
    image_name = "sample"
    if source_mode == "Use provided sample image":
        if SAMPLE_IMAGE_PATH.exists():
            try:
                image_rgb = load_rgb_image(SAMPLE_IMAGE_PATH)
                image_name = SAMPLE_IMAGE_PATH.stem
            except Exception as exc:
                st.error(f"Could not load sample image {SAMPLE_IMAGE_PATH.name}: {exc}")
        else:
            st.error(f"Sample image not found at {SAMPLE_IMAGE_PATH}")
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload sample image",
            type=list(SUPPORTED_UPLOAD_FILE_TYPES),
        )
        if uploaded_file is not None:
            try:
                image_rgb = read_uploaded_image(uploaded_file)
                image_name = Path(uploaded_file.name).stem
            except Exception as exc:
                st.error(f"Could not load {uploaded_file.name}: {exc}")
        elif SAMPLE_IMAGE_PATH.exists():
            st.sidebar.info("No upload selected. Falling back to the provided sample image.")
            try:
                image_rgb = load_rgb_image(SAMPLE_IMAGE_PATH)
                image_name = SAMPLE_IMAGE_PATH.stem
            except Exception as exc:
                st.error(f"Could not load sample image {SAMPLE_IMAGE_PATH.name}: {exc}")

    if image_rgb is None:
        st.stop()

    image_signature = f"{image_name}:{image_rgb.shape}"
    if st.session_state["image_signature"] != image_signature:
        reset_interaction_state()
        st.session_state["image_signature"] = image_signature

    segmentation_params = build_segmentation_parameters()
    pigment_profile = get_pigment_profile(segmentation_params.pigment_key)
    roi_mode = st.sidebar.selectbox(
        "ROI mode",
        options=["whole_pigment_mask", "per_fragment", "manual_polygon"],
        format_func=lambda value: {
            "whole_pigment_mask": f"Whole {pigment_profile.label} mask",
            "per_fragment": "Per fragment",
            "manual_polygon": "Manual polygon ROIs",
        }[value],
    )

    calibration_mode = st.sidebar.radio(
        "ColorChecker registration",
        options=["Automatic", "Manual 4-corner registration"],
    )

    with st.spinner("Detecting ColorChecker..."):
        automatic_detection = detect_colorchecker(image_rgb)

    chart_corners: np.ndarray | None = None
    registration_status = ""
    if calibration_mode == "Automatic":
        if automatic_detection is not None:
            chart_corners = automatic_detection.corners
            registration_status = f"Automatic detection succeeded ({automatic_detection.method})."
        elif source_mode == "Use provided sample image":
            chart_corners = SAMPLE_MANUAL_COLORCHECKER_CORNERS.copy()
            registration_status = "Automatic detection failed; using the sample's fallback ColorChecker corners."
        else:
            registration_status = "Automatic detection did not find a chart. Switch to manual registration."
    else:
        guess_corners = automatic_detection.corners if automatic_detection is not None else SAMPLE_MANUAL_COLORCHECKER_CORNERS
        with st.expander("Manual ColorChecker registration", expanded=True):
            chart_corners = render_chart_clicker(image_rgb, guess_corners)
        if chart_corners is not None:
            registration_status = "Manual registration ready."
        else:
            registration_status = "Manual registration requires four clicked corners."

    st.info(registration_status)

    original_display, _ = resize_to_long_edge(image_rgb, 1400)
    original_caption = "Original image"
    if chart_corners is not None:
        registration_overlay = draw_colorchecker_overlay(image_rgb, chart_corners)
        registration_display, _ = resize_to_long_edge(registration_overlay, 1400)
    else:
        registration_display = original_display

    top_cols = st.columns(2)
    top_cols[0].image(original_display, caption=original_caption, use_container_width=True)
    top_cols[1].image(registration_display, caption="ColorChecker registration", use_container_width=True)

    if chart_corners is None:
        st.warning("Calibration cannot run until ColorChecker registration is available.")
        st.stop()

    cleanup_polygons: list[PolygonOperation] = list(st.session_state["cleanup_polygons"])
    manual_roi_polygons: list[PolygonOperation] = list(st.session_state["roi_polygons"])

    with st.expander("Manual mask cleanup", expanded=False):
        st.caption(
            f"Use exclusion polygons to remove false positives such as torn edges or residual non-{pigment_profile.label.lower()} artefacts. "
            f"Use inclusion polygons only when the automatic mask misses valid {pigment_profile.label.lower()} pigment."
        )
        cleanup_overlay = overlay_mask(
            image_rgb,
            np.zeros(image_rgb.shape[:2], dtype=bool),
            color_rgb=pigment_profile.overlay_color_rgb,
        )
        cleanup_polygons = render_polygon_editor(
            cleanup_overlay,
            "cleanup",
            cleanup_polygons,
            allow_operation_toggle=True,
            default_name="mask_cleanup",
        )

    if roi_mode == "manual_polygon":
        with st.expander("Manual ROI polygons", expanded=True):
            st.caption(
                f"Define one or more polygon ROIs. Each ROI is intersected with the final {pigment_profile.label.lower()} mask before Lab statistics are calculated."
            )
            manual_roi_polygons = render_polygon_editor(
                image_rgb,
                "roi",
                manual_roi_polygons,
                allow_operation_toggle=False,
                default_name=f"roi_{len(st.session_state['roi_polygons']) + 1}",
            )

    with st.spinner("Running calibration, segmentation, and Lab analysis..."):
        analysis = run_analysis(
            image_rgb=image_rgb,
            chart_corners=chart_corners,
            segmentation_params=segmentation_params,
            cleanup_operations=cleanup_polygons,
            roi_mode=roi_mode,
            manual_roi_polygons=manual_roi_polygons,
        )

    middle_cols = st.columns(3)
    corrected_display, _ = resize_to_long_edge(analysis.calibration.corrected_rgb, 1100)
    support_caption = (
        "Cloth material support"
        if segmentation_params.material_mode == "woven_cloth"
        else "Fragment grouping support"
    )
    wallpaper_display, _ = resize_to_long_edge(
        overlay_mask(
            analysis.calibration.corrected_rgb,
            analysis.wallpaper_mask,
            color_rgb=analysis.pigment_profile.overlay_color_rgb,
        ),
        1100,
    )
    mask_display, _ = resize_to_long_edge(
        overlay_mask(
            analysis.calibration.corrected_rgb,
            analysis.pigment_mask,
            color_rgb=analysis.pigment_profile.overlay_color_rgb,
        ),
        1100,
    )
    middle_cols[0].image(corrected_display, caption="Calibrated image", use_container_width=True)
    middle_cols[1].image(wallpaper_display, caption=support_caption, use_container_width=True)
    middle_cols[2].image(mask_display, caption=f"Sampled {analysis.pigment_profile.label} mask", use_container_width=True)

    if analysis.material_masks:
        with st.expander("Woven cloth component masks", expanded=False):
            component_cols = st.columns(2)
            for index, (mask_name, mask) in enumerate(analysis.material_masks.items()):
                component_display, _ = resize_to_long_edge(
                    overlay_mask(
                        analysis.calibration.corrected_rgb,
                        mask,
                        color_rgb=analysis.pigment_profile.overlay_color_rgb,
                    ),
                    900,
                )
                component_cols[index % 2].image(
                    component_display,
                    caption=mask_name.replace("_", " "),
                    use_container_width=True,
                )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Mean chart Delta E00 before", f"{analysis.calibration.mean_delta_e_00_before:.2f}")
    metric_cols[1].metric("Mean chart Delta E00 after", f"{analysis.calibration.mean_delta_e_00_after:.2f}")
    metric_cols[2].metric("Mean chart Delta E76 after", f"{analysis.calibration.mean_delta_e_76_after:.2f}")
    metric_cols[3].metric(f"Sampled {analysis.pigment_profile.label} pixels", int(analysis.pigment_mask.sum()))

    st.subheader("Lab Results")
    if analysis.summary_df.empty:
        st.warning(
            f"No {analysis.pigment_profile.label.lower()} pixels matched the current criteria. "
            "Relax the thresholds or add inclusion polygons."
        )
    else:
        st.dataframe(
            analysis.summary_df.style.format(
                {
                    "mean_L": "{:.2f}",
                    "mean_a": "{:.2f}",
                    "mean_b": "{:.2f}",
                    "mean_C": "{:.2f}",
                    "mean_h": "{:.2f}",
                    "median_L": "{:.2f}",
                    "median_a": "{:.2f}",
                    "median_b": "{:.2f}",
                    "median_C": "{:.2f}",
                    "std_L": "{:.2f}",
                    "std_a": "{:.2f}",
                    "std_b": "{:.2f}",
                    "std_C": "{:.2f}",
                    "std_h": "{:.2f}",
                }
            ),
            use_container_width=True,
        )

    st.subheader("Distribution Inspection")
    distribution_figure = create_lab_distribution_figure(
        analysis.lab_image,
        analysis.pigment_mask,
        point_color=analysis.pigment_profile.overlay_color_rgb,
    )
    st.pyplot(distribution_figure, clear_figure=True, use_container_width=True)

    with st.expander("Calibration diagnostics", expanded=False):
        st.caption("Patch-by-patch ColorChecker fit diagnostics for the current registration.")
        st.dataframe(
            analysis.calibration_df.style.format(
                {
                    "delta_e_00_before": "{:.2f}",
                    "delta_e_00_after": "{:.2f}",
                    "delta_e_76_before": "{:.2f}",
                    "delta_e_76_after": "{:.2f}",
                    "observed_r": "{:.3f}",
                    "observed_g": "{:.3f}",
                    "observed_b": "{:.3f}",
                    "corrected_r": "{:.3f}",
                    "corrected_g": "{:.3f}",
                    "corrected_b": "{:.3f}",
                }
            ),
            use_container_width=True,
        )

    if not analysis.segmentation_df.empty:
        with st.expander("Segmentation diagnostics", expanded=False):
            st.dataframe(analysis.segmentation_df, use_container_width=True)

    st.subheader("Export")
    output_root = st.text_input(
        "Output directory",
        value=str(Path.cwd() / "outputs"),
    )
    if st.button("Save corrected image, mask, overlay, and CSV"):
        output_dir = create_output_directory(output_root, image_name)
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
        st.success(f"Saved analysis outputs to {output_dir}")
        for label, file_path in files.items():
            st.write(f"{label}: {file_path}")

    csv_bytes = analysis.summary_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Lab summary CSV",
        data=csv_bytes,
        file_name=f"{image_name}_{analysis.pigment_profile.key}_lab_summary.csv",
        mime="text/csv",
    )

    st.markdown(
        f"""
        **Method note:** The app registers the in-frame ColorChecker, fits an image-derived RGB correction,
        converts the corrected image to CIE L*a*b* with a reproducible sRGB-to-Lab path, derives C*ab chroma
        and h_ab hue angle for ROI summaries, isolates the selected
        {analysis.pigment_profile.label.lower()} colour band
        directly from the corrected image, groups those selections into fragment-level supports when needed, and
        reports statistics only for the selected pixels. The output is suitable for documented image-based
        comparison, not as a substitute for direct instrument measurement.
        """
    )


if __name__ == "__main__":
    main()
