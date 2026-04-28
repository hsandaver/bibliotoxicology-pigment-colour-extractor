import io
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from wallpaper_lab.io_utils import SUPPORTED_UPLOAD_FILE_TYPES, load_rgb_image


class DummyRawContext:
    def __init__(self) -> None:
        self.postprocess_kwargs = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def postprocess(self, **kwargs):
        self.postprocess_kwargs = kwargs
        return np.full((3, 4, 3), 65535, dtype=np.uint16)


class DummyRawPy:
    class ColorSpace:
        sRGB = "srgb"

    def __init__(self) -> None:
        self.last_source = None
        self.last_source_pos = None
        self.last_context: DummyRawContext | None = None

    def imread(self, source):
        self.last_source = source
        self.last_source_pos = source.tell() if hasattr(source, "tell") else None
        self.last_context = DummyRawContext()
        return self.last_context


class IoUtilsTests(unittest.TestCase):
    def test_supported_upload_types_include_nef(self) -> None:
        self.assertIn("nef", SUPPORTED_UPLOAD_FILE_TYPES)

    def test_load_rgb_image_reads_standard_raster_upload(self) -> None:
        buffer = io.BytesIO()
        Image.fromarray(np.array([[[0, 128, 255]]], dtype=np.uint8)).save(buffer, format="PNG")
        buffer.seek(0)

        image = load_rgb_image(buffer, source_name="sample.png")

        self.assertEqual(image.shape, (1, 1, 3))
        np.testing.assert_allclose(image[0, 0], np.array([0.0, 128.0 / 255.0, 1.0], dtype=np.float32))

    def test_load_rgb_image_uses_rawpy_for_raw_uploads(self) -> None:
        raw_source = io.BytesIO(b"pretend-nef-data")
        raw_source.seek(5)
        dummy_rawpy = DummyRawPy()

        with mock.patch("wallpaper_lab.io_utils.rawpy", dummy_rawpy):
            image = load_rgb_image(raw_source, source_name="sample.NEF")

        self.assertEqual(image.shape, (3, 4, 3))
        self.assertEqual(dummy_rawpy.last_source_pos, 0)
        self.assertIsNotNone(dummy_rawpy.last_context)
        assert dummy_rawpy.last_context is not None
        self.assertTrue(dummy_rawpy.last_context.postprocess_kwargs["use_camera_wb"])
        self.assertTrue(dummy_rawpy.last_context.postprocess_kwargs["no_auto_bright"])
        self.assertEqual(dummy_rawpy.last_context.postprocess_kwargs["output_bps"], 16)
        self.assertEqual(dummy_rawpy.last_context.postprocess_kwargs["output_color"], DummyRawPy.ColorSpace.sRGB)
        self.assertAlmostEqual(float(image.max()), 1.0, places=6)

    def test_load_rgb_image_raises_helpful_error_without_rawpy(self) -> None:
        with mock.patch("wallpaper_lab.io_utils.rawpy", None):
            with self.assertRaisesRegex(ImportError, "rawpy"):
                load_rgb_image(io.BytesIO(b"pretend-nef-data"), source_name="sample.nef")


if __name__ == "__main__":
    unittest.main()
