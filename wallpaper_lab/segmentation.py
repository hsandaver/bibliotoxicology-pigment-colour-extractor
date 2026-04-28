from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import color, measure, morphology

from wallpaper_lab.colorchecker import corners_to_bbox


@dataclass
class PigmentProfile:
    key: str
    label: str
    roi_label: str
    mask_file_stem: str
    overlay_color_rgb: tuple[int, int, int]
    hue_min: float
    hue_max: float
    saturation_min: float
    a_min: float
    a_max: float
    b_min: float
    b_max: float
    chroma_min: float
    lightness_min: float
    lightness_max: float
    description: str
    material_mode: str = "printed"


PIGMENT_PROFILES: dict[str, PigmentProfile] = {
    "green": PigmentProfile(
        key="green",
        label="Green",
        roi_label="green",
        mask_file_stem="green_mask",
        overlay_color_rgb=(0, 210, 80),
        hue_min=40.0,
        hue_max=160.0,
        saturation_min=0.03,
        a_min=-80.0,
        a_max=-8.0,
        b_min=7.0,
        b_max=100.0,
        chroma_min=10.0,
        lightness_min=20.0,
        lightness_max=65.0,
        description="Olive and green printed pigment selections.",
    ),
    "vermilion": PigmentProfile(
        key="vermilion",
        label="Vermilion",
        roi_label="vermilion",
        mask_file_stem="vermilion_mask",
        overlay_color_rgb=(235, 70, 35),
        hue_min=345.0,
        hue_max=38.0,
        saturation_min=0.10,
        a_min=20.0,
        a_max=85.0,
        b_min=4.0,
        b_max=78.0,
        chroma_min=20.0,
        lightness_min=18.0,
        lightness_max=82.0,
        description="Warm red-orange selections around vermilion/cinnabar-like pigment colours.",
    ),
    "chrome_yellow": PigmentProfile(
        key="chrome_yellow",
        label="Chrome yellow",
        roi_label="chrome_yellow",
        mask_file_stem="chrome_yellow_mask",
        overlay_color_rgb=(255, 205, 0),
        hue_min=35.0,
        hue_max=70.0,
        saturation_min=0.16,
        a_min=-18.0,
        a_max=38.0,
        b_min=45.0,
        b_max=105.0,
        chroma_min=45.0,
        lightness_min=30.0,
        lightness_max=96.0,
        description="Bright warm-yellow selections around chrome-yellow lead chromate pigment colours.",
    ),
    "altered_chrome_yellow": PigmentProfile(
        key="altered_chrome_yellow",
        label="Altered chrome yellow",
        roi_label="altered_chrome_yellow",
        mask_file_stem="altered_chrome_yellow_mask",
        overlay_color_rgb=(185, 135, 20),
        hue_min=20.0,
        hue_max=78.0,
        saturation_min=0.08,
        a_min=-8.0,
        a_max=46.0,
        b_min=12.0,
        b_max=85.0,
        chroma_min=18.0,
        lightness_min=12.0,
        lightness_max=82.0,
        description=(
            "Darker yellow-orange to brown candidate selections consistent with reported chrome-yellow "
            "darkening trends; this is a colour mask, not chemical identification."
        ),
    ),
    "chrome_green": PigmentProfile(
        key="chrome_green",
        label="Chrome green mixture",
        roi_label="chrome_green",
        mask_file_stem="chrome_green_mask",
        overlay_color_rgb=(45, 170, 120),
        hue_min=70.0,
        hue_max=190.0,
        saturation_min=0.04,
        a_min=-70.0,
        a_max=-3.0,
        b_min=-35.0,
        b_max=60.0,
        chroma_min=8.0,
        lightness_min=12.0,
        lightness_max=76.0,
        description=(
            "Candidate green selections for chrome-yellow/Prussian-blue mixture colours; this is a "
            "visible-colour mask and should be checked against analytical evidence."
        ),
    ),
    "bookcloth_blue_green": PigmentProfile(
        key="bookcloth_blue_green",
        label="Bookcloth blue-green",
        roi_label="bookcloth_blue_green",
        mask_file_stem="bookcloth_blue_green_mask",
        overlay_color_rgb=(0, 190, 165),
        hue_min=60.0,
        hue_max=245.0,
        saturation_min=0.035,
        a_min=-22.0,
        a_max=6.0,
        b_min=-35.0,
        b_max=18.0,
        chroma_min=2.0,
        lightness_min=18.0,
        lightness_max=78.0,
        description=(
            "Muted blue-green and green-grey cloth-cover selections using the strict colour-mask workflow."
        ),
    ),
    "bookcloth_blue_green_woven": PigmentProfile(
        key="bookcloth_blue_green_woven",
        label="Bookcloth blue-green woven",
        roi_label="bookcloth_blue_green_woven",
        mask_file_stem="bookcloth_blue_green_woven_mask",
        overlay_color_rgb=(0, 190, 165),
        hue_min=60.0,
        hue_max=245.0,
        saturation_min=0.035,
        a_min=-22.0,
        a_max=6.0,
        b_min=-35.0,
        b_max=18.0,
        chroma_min=2.0,
        lightness_min=18.0,
        lightness_max=78.0,
        description=(
            "Muted blue-green and green-grey cloth-cover selections with woven-cloth diagnostics for "
            "thread texture, thread shadows, and substrate show-through."
        ),
        material_mode="woven_cloth",
    ),
}

PIGMENT_ALIASES = {
    "vermillion": "vermilion",
    "chrome-yellow": "chrome_yellow",
    "chrome yellow": "chrome_yellow",
    "altered chrome yellow": "altered_chrome_yellow",
    "darkened_chrome_yellow": "altered_chrome_yellow",
    "darkened chrome yellow": "altered_chrome_yellow",
    "chrome-green": "chrome_green",
    "chrome green": "chrome_green",
    "brunswick_green": "chrome_green",
    "brunswick green": "chrome_green",
    "bookcloth": "bookcloth_blue_green",
    "book cloth": "bookcloth_blue_green",
    "cloth": "bookcloth_blue_green",
    "cloth green": "bookcloth_blue_green",
    "blue green cloth": "bookcloth_blue_green",
    "blue-green cloth": "bookcloth_blue_green",
    "woven bookcloth": "bookcloth_blue_green_woven",
    "bookcloth woven": "bookcloth_blue_green_woven",
    "book cloth woven": "bookcloth_blue_green_woven",
    "woven blue green cloth": "bookcloth_blue_green_woven",
    "blue green woven cloth": "bookcloth_blue_green_woven",
}


def get_pigment_profile(pigment_key: str | PigmentProfile) -> PigmentProfile:
    if isinstance(pigment_key, PigmentProfile):
        return pigment_key
    normalized = pigment_key.strip().lower().replace("-", "_")
    normalized = PIGMENT_ALIASES.get(normalized, normalized)
    if normalized not in PIGMENT_PROFILES:
        choices = ", ".join(PIGMENT_PROFILES)
        raise ValueError(f"Unknown pigment extractor '{pigment_key}'. Expected one of: {choices}")
    return PIGMENT_PROFILES[normalized]


DEFAULT_PIGMENT_PROFILE = PIGMENT_PROFILES["green"]


@dataclass
class PigmentMaskResult:
    pigment_mask: np.ndarray
    material_masks: dict[str, np.ndarray]


@dataclass
class SegmentationParameters:
    pigment_key: str = DEFAULT_PIGMENT_PROFILE.key
    hue_min: float = DEFAULT_PIGMENT_PROFILE.hue_min
    hue_max: float = DEFAULT_PIGMENT_PROFILE.hue_max
    saturation_min: float = DEFAULT_PIGMENT_PROFILE.saturation_min
    a_min: float = DEFAULT_PIGMENT_PROFILE.a_min
    a_max: float = DEFAULT_PIGMENT_PROFILE.a_max
    b_min: float = DEFAULT_PIGMENT_PROFILE.b_min
    b_max: float = DEFAULT_PIGMENT_PROFILE.b_max
    chroma_min: float = DEFAULT_PIGMENT_PROFILE.chroma_min
    lightness_min: float = DEFAULT_PIGMENT_PROFILE.lightness_min
    lightness_max: float = DEFAULT_PIGMENT_PROFILE.lightness_max
    chart_padding_px: int = 100
    bottom_exclusion_fraction: float = 0.88
    edge_exclusion_px: int = 4
    min_area: int = 20
    fragment_group_scale: float = 0.2
    fragment_group_dilation_radius: int = 15
    material_mode: str = DEFAULT_PIGMENT_PROFILE.material_mode
    cloth_smoothing_radius_px: int = 4
    cloth_support_radius_px: int = 9
    cloth_min_seed_fraction: float = 0.08
    cloth_shadow_lightness_tolerance: float = 22.0
    cloth_shadow_chroma_factor: float = 0.35
    cloth_ab_tolerance: float = 8.0
    cloth_substrate_chroma_max: float = 8.0
    cloth_substrate_saturation_max: float = 0.16
    cloth_substrate_seed_fraction: float = 0.16
    cloth_sample_mode: str = "dyed_threads"

    @classmethod
    def from_profile(
        cls,
        pigment_key: str | PigmentProfile = DEFAULT_PIGMENT_PROFILE.key,
        **overrides,
    ) -> "SegmentationParameters":
        overrides = dict(overrides)
        profile = get_pigment_profile(pigment_key)
        material_mode = overrides.pop("material_mode", profile.material_mode)
        values = {
            "pigment_key": profile.key,
            "hue_min": profile.hue_min,
            "hue_max": profile.hue_max,
            "saturation_min": profile.saturation_min,
            "a_min": profile.a_min,
            "a_max": profile.a_max,
            "b_min": profile.b_min,
            "b_max": profile.b_max,
            "chroma_min": profile.chroma_min,
            "lightness_min": profile.lightness_min,
            "lightness_max": profile.lightness_max,
            "material_mode": material_mode,
        }
        if material_mode == "woven_cloth":
            values.update(
                {
                    "bottom_exclusion_fraction": 0.98,
                    "edge_exclusion_px": 0,
                    "min_area": 8,
                    "fragment_group_dilation_radius": 10,
                }
            )
        values.update(overrides)
        return cls(**values)


def _hue_in_range(hue_degrees: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    if minimum <= maximum:
        return (hue_degrees >= minimum) & (hue_degrees <= maximum)
    return (hue_degrees >= minimum) | (hue_degrees <= maximum)


def _profile_threshold_mask(rgb: np.ndarray, params: SegmentationParameters) -> np.ndarray:
    rgb = np.clip(rgb, 0.0, 1.0)
    lab = color.rgb2lab(rgb)
    hsv = color.rgb2hsv(rgb)
    return _profile_threshold_mask_from_channels(lab, hsv, params)


def _profile_threshold_mask_from_channels(
    lab: np.ndarray,
    hsv: np.ndarray,
    params: SegmentationParameters,
) -> np.ndarray:
    hue = hsv[:, :, 0] * 360.0
    saturation = hsv[:, :, 1]
    lightness = lab[:, :, 0]
    a_star = lab[:, :, 1]
    b_star = lab[:, :, 2]
    chroma = np.sqrt(np.square(a_star) + np.square(b_star))

    pigment_mask = _hue_in_range(hue, params.hue_min, params.hue_max)
    pigment_mask &= saturation >= params.saturation_min
    pigment_mask &= a_star >= params.a_min
    pigment_mask &= a_star <= params.a_max
    pigment_mask &= b_star >= params.b_min
    pigment_mask &= b_star <= params.b_max
    pigment_mask &= chroma >= params.chroma_min
    pigment_mask &= lightness >= params.lightness_min
    pigment_mask &= lightness <= params.lightness_max
    return pigment_mask


def _smooth_rgb_for_cloth(rgb: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return rgb
    kernel_size = max(3, int(radius_px) * 2 + 1)
    return cv2.GaussianBlur(rgb, (kernel_size, kernel_size), sigmaX=max(float(radius_px), 0.1))


def build_exclusion_mask(
    image_shape: tuple[int, int],
    chart_corners: np.ndarray,
    params: SegmentationParameters,
) -> np.ndarray:
    exclusion = np.zeros(image_shape, dtype=bool)
    xmin, ymin, xmax, ymax = corners_to_bbox(chart_corners)
    pad = params.chart_padding_px
    height, width = image_shape
    exclusion[max(0, ymin - pad):min(height, ymax + pad), max(0, xmin - pad):min(width, xmax + pad)] = True
    exclusion[int(height * params.bottom_exclusion_fraction):, :] = True
    return exclusion


def build_pigment_mask(
    corrected_rgb: np.ndarray,
    chart_corners: np.ndarray,
    params: SegmentationParameters,
) -> np.ndarray:
    """Directly isolate the selected pigment or material band."""
    return build_pigment_mask_result(corrected_rgb, chart_corners, params).pigment_mask


def build_pigment_mask_result(
    corrected_rgb: np.ndarray,
    chart_corners: np.ndarray,
    params: SegmentationParameters,
) -> PigmentMaskResult:
    """Build the selected mask and any material-specific component masks."""
    if params.material_mode == "woven_cloth":
        return _build_woven_cloth_mask_result(corrected_rgb, chart_corners, params)
    return PigmentMaskResult(
        pigment_mask=_build_printed_pigment_mask(corrected_rgb, chart_corners, params),
        material_masks={},
    )


def _build_printed_pigment_mask(
    corrected_rgb: np.ndarray,
    chart_corners: np.ndarray,
    params: SegmentationParameters,
) -> np.ndarray:
    """Strict printed-pigment path used by the wallpaper workflow."""
    pigment_mask = _profile_threshold_mask(corrected_rgb, params)
    pigment_mask &= ~build_exclusion_mask(pigment_mask.shape, chart_corners, params)
    pigment_mask = morphology.remove_small_objects(pigment_mask, params.min_area)

    support_mask, _ = build_fragment_support_mask(pigment_mask, params)
    if params.edge_exclusion_px > 0 and np.any(support_mask):
        distance = ndi.distance_transform_edt(support_mask)
        pigment_mask &= distance >= params.edge_exclusion_px

    return morphology.remove_small_objects(pigment_mask, params.min_area)


def _build_woven_cloth_mask_result(
    corrected_rgb: np.ndarray,
    chart_corners: np.ndarray,
    params: SegmentationParameters,
) -> PigmentMaskResult:
    """Texture-tolerant path for dyed or coloured woven bookcloth surfaces.

    The seed remains colour-profile based, but the final mask can recover local
    thread shadows and neutral substrate show-through only when they sit inside
    a neighbourhood already supported by dyed cloth-colour pixels.
    """
    rgb = np.clip(corrected_rgb, 0.0, 1.0)
    lab = color.rgb2lab(rgb)
    hsv = color.rgb2hsv(rgb)
    smoothed_rgb = _smooth_rgb_for_cloth(rgb, params.cloth_smoothing_radius_px)
    smoothed_seed = _profile_threshold_mask(smoothed_rgb, params)
    raw_seed = _profile_threshold_mask_from_channels(lab, hsv, params)

    exclusion = build_exclusion_mask(raw_seed.shape, chart_corners, params)
    strict_colour_seed = raw_seed & ~exclusion
    texture_smoothed_seed = smoothed_seed & ~exclusion
    dyed_thread_seed = strict_colour_seed.copy()
    dyed_thread_seed = morphology.remove_small_objects(dyed_thread_seed, params.min_area)
    support_seed = morphology.remove_small_objects(
        dyed_thread_seed | texture_smoothed_seed,
        params.min_area,
    )

    support_size = max(3, int(params.cloth_support_radius_px) * 2 + 1)
    local_seed_fraction = ndi.uniform_filter(
        support_seed.astype(np.float32),
        size=support_size,
        mode="constant",
        cval=0.0,
    )
    cloth_support = (local_seed_fraction >= params.cloth_min_seed_fraction) & ~exclusion
    if params.cloth_support_radius_px > 1:
        cloth_support = morphology.binary_closing(
            cloth_support,
            morphology.disk(max(1, params.cloth_support_radius_px // 2)),
        )
    cloth_support = ndi.binary_fill_holes(cloth_support).astype(bool)
    cloth_support &= ~exclusion
    cloth_support = morphology.remove_small_objects(cloth_support, max(params.min_area, 10))

    lightness = lab[:, :, 0]
    a_star = lab[:, :, 1]
    b_star = lab[:, :, 2]
    chroma = np.sqrt(np.square(a_star) + np.square(b_star))
    saturation = hsv[:, :, 1]

    shadow_mask = cloth_support.copy()
    shadow_mask &= ~dyed_thread_seed
    shadow_mask &= lightness >= max(0.0, params.lightness_min - params.cloth_shadow_lightness_tolerance)
    shadow_mask &= lightness <= params.lightness_max
    shadow_mask &= a_star >= params.a_min - params.cloth_ab_tolerance
    shadow_mask &= a_star <= params.a_max + params.cloth_ab_tolerance
    shadow_mask &= b_star >= params.b_min - params.cloth_ab_tolerance
    shadow_mask &= b_star <= params.b_max + params.cloth_ab_tolerance
    shadow_mask &= chroma >= params.chroma_min * params.cloth_shadow_chroma_factor
    shadow_mask &= saturation >= params.saturation_min * 0.25

    substrate_mask = cloth_support.copy()
    substrate_mask &= ~dyed_thread_seed
    substrate_mask &= local_seed_fraction >= params.cloth_substrate_seed_fraction
    substrate_mask &= chroma <= params.cloth_substrate_chroma_max
    substrate_mask &= saturation <= params.cloth_substrate_saturation_max
    substrate_mask &= lightness >= max(0.0, params.lightness_min - params.cloth_shadow_lightness_tolerance)
    substrate_mask &= lightness <= min(100.0, params.lightness_max + params.cloth_shadow_lightness_tolerance)
    shadow_mask &= ~substrate_mask

    woven_surface_mask = dyed_thread_seed | shadow_mask | substrate_mask
    if params.cloth_sample_mode == "surface_appearance":
        pigment_mask = woven_surface_mask
    elif params.cloth_sample_mode == "dyed_threads_with_shadows":
        pigment_mask = dyed_thread_seed | shadow_mask
    else:
        pigment_mask = dyed_thread_seed
    pigment_mask = morphology.remove_small_objects(pigment_mask & ~exclusion, params.min_area)

    profile = get_pigment_profile(params.pigment_key)
    mask_prefix = profile.mask_file_stem
    return PigmentMaskResult(
        pigment_mask=pigment_mask,
        material_masks={
            f"{mask_prefix}_strict_colour_seed": strict_colour_seed,
            f"{mask_prefix}_texture_smoothed_seed": texture_smoothed_seed,
            f"{mask_prefix}_dyed_thread_seed": dyed_thread_seed,
            f"{mask_prefix}_thread_shadow": shadow_mask,
            f"{mask_prefix}_substrate_showthrough": substrate_mask,
            f"{mask_prefix}_woven_surface": woven_surface_mask,
            f"{mask_prefix}_cloth_support": cloth_support,
        },
    )


def build_green_mask(
    corrected_rgb: np.ndarray,
    chart_corners: np.ndarray,
    params: SegmentationParameters,
) -> np.ndarray:
    """Backward-compatible alias for the selected pigment mask builder."""
    return build_pigment_mask(corrected_rgb, chart_corners, params)


def build_fragment_support_mask(
    pigment_mask: np.ndarray,
    params: SegmentationParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Group separated selected pigment motifs into fragment-level support regions at low resolution."""
    height, width = pigment_mask.shape
    scale = params.fragment_group_scale
    small_width = max(1, int(round(width * scale)))
    small_height = max(1, int(round(height * scale)))
    small_mask = cv2.resize(
        pigment_mask.astype(np.uint8) * 255,
        (small_width, small_height),
        interpolation=cv2.INTER_NEAREST,
    ) > 0
    small_mask = morphology.binary_dilation(
        small_mask,
        morphology.disk(params.fragment_group_dilation_radius),
    )
    small_mask = morphology.binary_closing(
        small_mask,
        morphology.disk(max(1, params.fragment_group_dilation_radius // 2)),
    )
    small_mask = morphology.remove_small_objects(small_mask, 10)
    small_labels = measure.label(small_mask)

    fragment_labels = np.zeros(pigment_mask.shape, dtype=np.int32)
    for label_id in np.unique(small_labels):
        if label_id == 0:
            continue
        label_mask = small_labels == label_id
        upsampled = cv2.resize(
            label_mask.astype(np.uint8) * 255,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
        fragment_labels[upsampled] = label_id

    return fragment_labels > 0, fragment_labels
