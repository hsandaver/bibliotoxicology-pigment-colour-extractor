from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import numpy as np
from PIL import Image

try:
    import rawpy
except ImportError:  # pragma: no cover - exercised via loader error path in tests.
    rawpy = None


RASTER_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
RAW_IMAGE_EXTENSIONS = frozenset(
    {
        ".nef",
        ".nrw",
        ".dng",
        ".cr2",
        ".cr3",
        ".arw",
        ".orf",
        ".raf",
        ".rw2",
        ".pef",
        ".srw",
    }
)
SUPPORTED_UPLOAD_FILE_TYPES = tuple(
    extension.lstrip(".") for extension in (*RASTER_IMAGE_EXTENSIONS, *sorted(RAW_IMAGE_EXTENSIONS))
)


def _source_label(source: Any, source_name: str | None = None) -> str | None:
    if source_name:
        return source_name
    if isinstance(source, (str, Path)):
        return str(source)
    name = getattr(source, "name", None)
    return str(name) if name is not None else None


def _source_suffix(source: Any, source_name: str | None = None) -> str:
    label = _source_label(source, source_name)
    return Path(label).suffix.lower() if label is not None else ""


def _rewind_source(source: Any) -> None:
    seek = getattr(source, "seek", None)
    if callable(seek):
        seek(0)


def _normalize_to_float_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if np.issubdtype(array.dtype, np.integer):
        scale = float(np.iinfo(array.dtype).max)
    else:
        scale = 1.0
    return array.astype(np.float32) / scale


def _load_raw_rgb_image(source: Any, source_name: str | None = None) -> np.ndarray:
    if rawpy is None:
        raise ImportError(
            f"RAW image support for {_source_label(source, source_name) or 'this file'} requires the "
            "'rawpy' package. Install the project requirements and restart the app."
        )

    _rewind_source(source)
    raw_source = str(source) if isinstance(source, (str, Path)) else source
    postprocess_kwargs = {
        "use_camera_wb": True,
        "no_auto_bright": True,
        "output_bps": 16,
    }
    color_space = getattr(getattr(rawpy, "ColorSpace", None), "sRGB", None)
    if color_space is not None:
        postprocess_kwargs["output_color"] = color_space

    with rawpy.imread(raw_source) as raw:
        rgb = raw.postprocess(**postprocess_kwargs)
    return _normalize_to_float_rgb(rgb)


def load_rgb_image(source: str | Path | Any, source_name: str | None = None) -> np.ndarray:
    """Load a raster or RAW image as float32 RGB in the range [0, 1]."""
    if _source_suffix(source, source_name) in RAW_IMAGE_EXTENSIONS:
        return _load_raw_rgb_image(source, source_name=source_name)

    _rewind_source(source)
    with Image.open(source) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def as_uint8(image: np.ndarray) -> np.ndarray:
    """Convert an image in [0, 1] or uint8 into uint8 RGB."""
    if image.dtype == np.uint8:
        return image
    clipped = np.clip(image, 0.0, 1.0)
    return np.round(clipped * 255.0).astype(np.uint8)


def save_rgb_image(path: str | Path, image: np.ndarray) -> None:
    Image.fromarray(as_uint8(image)).save(path)


def resize_to_long_edge(image: np.ndarray, long_edge: int) -> Tuple[np.ndarray, float]:
    """Resize while preserving aspect ratio. Returns resized image and scale."""
    height, width = image.shape[:2]
    current_long_edge = max(height, width)
    if current_long_edge <= long_edge:
        return image.copy(), 1.0

    scale = long_edge / float(current_long_edge)
    new_size = (int(round(width * scale)), int(round(height * scale)))
    pil_image = Image.fromarray(as_uint8(image))
    resized = np.asarray(pil_image.resize(new_size, Image.Resampling.LANCZOS))
    return resized, scale


def crop_with_padding(
    image: np.ndarray,
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    padding: float = 0.1,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop a box with proportional padding. Returns crop and top-left origin."""
    height, width = image.shape[:2]
    box_width = xmax - xmin
    box_height = ymax - ymin
    pad_x = int(round(box_width * padding))
    pad_y = int(round(box_height * padding))
    x0 = max(0, xmin - pad_x)
    y0 = max(0, ymin - pad_y)
    x1 = min(width, xmax + pad_x)
    y1 = min(height, ymax + pad_y)
    return image[y0:y1, x0:x1].copy(), (x0, y0)
