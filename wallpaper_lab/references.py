from __future__ import annotations

from pathlib import Path

import numpy as np

SAMPLE_IMAGE_PATH = Path("/Users/hsandaver/Downloads/CCMC628-15.png")

# Approximate outer corners for the supplied sample image, ordered TL, TR, BR, BL.
# These are derived from the actual sample frame and used as a practical fallback
# when automatic ColorChecker detection is not available or needs manual correction.
SAMPLE_MANUAL_COLORCHECKER_CORNERS = np.array(
    [
        [3170.28, 2578.68],
        [4600.56, 2578.68],
        [4600.56, 3420.84],
        [3170.28, 3420.84],
    ],
    dtype=np.float32,
)

# Macbeth / X-Rite ColorChecker Classic sRGB targets in row-major patch order.
# Source basis: BabelColor ColorChecker Classic reference datasets and the widely
# used 8-bit sRGB equivalents published for D65-adapted display workflows. The
# app fits a reproducible image-derived correction into sRGB before converting to
# CIE Lab; it does not claim raw-camera or spectral calibration equivalence.
COLORCHECKER_PATCH_NAMES = [
    "Dark Skin",
    "Light Skin",
    "Blue Sky",
    "Foliage",
    "Blue Flower",
    "Bluish Green",
    "Orange",
    "Purplish Blue",
    "Moderate Red",
    "Purple",
    "Yellow Green",
    "Orange Yellow",
    "Blue",
    "Green",
    "Red",
    "Yellow",
    "Magenta",
    "Cyan",
    "White 9.5",
    "Neutral 8",
    "Neutral 6.5",
    "Neutral 5",
    "Neutral 3.5",
    "Black 2",
]

COLORCHECKER_CLASSIC_SRGB = np.array(
    [
        [115, 82, 68],
        [194, 150, 130],
        [98, 122, 157],
        [87, 108, 67],
        [133, 128, 177],
        [103, 189, 170],
        [214, 126, 44],
        [80, 91, 166],
        [193, 90, 99],
        [94, 60, 108],
        [157, 188, 64],
        [224, 163, 46],
        [56, 61, 150],
        [70, 148, 73],
        [175, 54, 60],
        [231, 199, 31],
        [187, 86, 149],
        [8, 133, 161],
        [243, 243, 242],
        [200, 200, 200],
        [160, 160, 160],
        [122, 122, 121],
        [85, 85, 85],
        [52, 52, 52],
    ],
    dtype=np.float32,
) / 255.0

# The Calibrite chart used in the supplied sample is laid out with the grayscale
# row across the top. This reordered table maps the same 24 reference colors into
# the observed display order of that chart.
CALIBRITE_DISPLAY_ORDER_INDICES = np.array(
    [
        23, 22, 21, 20, 19, 18,
        17, 16, 15, 14, 13, 12,
        11, 10, 9, 8, 7, 6,
        5, 4, 3, 2, 1, 0,
    ],
    dtype=np.int32,
)

CALIBRITE_DISPLAY_PATCH_NAMES = [
    COLORCHECKER_PATCH_NAMES[index] for index in CALIBRITE_DISPLAY_ORDER_INDICES
]
CALIBRITE_DISPLAY_SRGB = COLORCHECKER_CLASSIC_SRGB[CALIBRITE_DISPLAY_ORDER_INDICES]
