import unittest

import cv2
import numpy as np

from wallpaper_lab.colorchecker import detect_colorchecker
from wallpaper_lab.references import CALIBRITE_DISPLAY_SRGB


class ColorCheckerDetectionTests(unittest.TestCase):
    def _build_chart_scene(self) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        height, width = 900, 1400
        image = np.full((height, width, 3), np.array([0.98, 0.75, 0.35], dtype=np.float32), dtype=np.float32)

        chart_x, chart_y, chart_w, chart_h = 150, 90, 460, 300
        image[chart_y:chart_y + chart_h, chart_x:chart_x + chart_w] = np.array([0.05, 0.05, 0.05], dtype=np.float32)

        x_centers = np.linspace(chart_x + 55, chart_x + chart_w - 55, 6, dtype=np.float32)
        y_centers = np.linspace(chart_y + 45, chart_y + chart_h - 45, 4, dtype=np.float32)
        patch_half_w = 28
        patch_half_h = 24
        patch_index = 0
        for center_y in y_centers:
            for center_x in x_centers:
                x0 = int(round(center_x - patch_half_w))
                x1 = int(round(center_x + patch_half_w))
                y0 = int(round(center_y - patch_half_h))
                y1 = int(round(center_y + patch_half_h))
                image[y0:y1, x0:x1] = CALIBRITE_DISPLAY_SRGB[patch_index]
                patch_index += 1

        fragment_points = np.array(
            [
                [760, 430],
                [1080, 360],
                [1280, 500],
                [1220, 760],
                [900, 820],
                [700, 680],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(image, [fragment_points], color=(0.22, 0.08, 0.04))

        return image, (chart_x, chart_y, chart_x + chart_w, chart_y + chart_h)

    def _build_dark_bookcloth_chart_scene(self) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        height, width = 900, 1200
        image = np.full((height, width, 3), np.array([0.09, 0.12, 0.13], dtype=np.float32), dtype=np.float32)

        chart_x, chart_y, chart_w, chart_h = 120, 80, 360, 620
        image[chart_y:chart_y + chart_h, chart_x:chart_x + chart_w] = np.array(
            [0.025, 0.025, 0.025],
            dtype=np.float32,
        )

        rotated_patches = np.rot90(CALIBRITE_DISPLAY_SRGB.reshape(4, 6, 3), k=1)
        x_centers = chart_x + np.linspace(0.17, 0.83, 4, dtype=np.float32) * chart_w
        y_centers = chart_y + np.linspace(0.15, 0.85, 6, dtype=np.float32) * chart_h
        patch_half = 28
        for row_index, center_y in enumerate(y_centers):
            for column_index, center_x in enumerate(x_centers):
                x0 = int(round(center_x - patch_half))
                x1 = int(round(center_x + patch_half))
                y0 = int(round(center_y - patch_half))
                y1 = int(round(center_y + patch_half))
                image[y0:y1, x0:x1] = rotated_patches[row_index, column_index]

        return image, (chart_x, chart_y, chart_x + chart_w, chart_y + chart_h)

    def _build_border_spanning_dark_scene(self) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        height, width = 900, 1400
        image = np.full((height, width, 3), np.array([0.72, 0.69, 0.62], dtype=np.float32), dtype=np.float32)
        image[:, :660] = np.array([0.025, 0.025, 0.025], dtype=np.float32)

        chart_x, chart_y, chart_w, chart_h = 120, 80, 360, 620
        image[chart_y:chart_y + chart_h, chart_x:chart_x + chart_w] = np.array(
            [0.025, 0.025, 0.025],
            dtype=np.float32,
        )

        rotated_patches = np.rot90(CALIBRITE_DISPLAY_SRGB.reshape(4, 6, 3), k=1)
        x_centers = chart_x + np.linspace(0.17, 0.83, 4, dtype=np.float32) * chart_w
        y_centers = chart_y + np.linspace(0.15, 0.85, 6, dtype=np.float32) * chart_h
        patch_half = 28
        for row_index, center_y in enumerate(y_centers):
            for column_index, center_x in enumerate(x_centers):
                x0 = int(round(center_x - patch_half))
                x1 = int(round(center_x + patch_half))
                y0 = int(round(center_y - patch_half))
                y1 = int(round(center_y + patch_half))
                image[y0:y1, x0:x1] = rotated_patches[row_index, column_index]

        return image, (chart_x, chart_y, chart_x + chart_w, chart_y + chart_h)

    def test_prefers_rectangular_chart_over_larger_fragment(self) -> None:
        image, expected_bbox = self._build_chart_scene()

        detection = detect_colorchecker(image)

        self.assertIsNotNone(detection)
        xmin, ymin = np.min(detection.corners, axis=0)
        xmax, ymax = np.max(detection.corners, axis=0)
        detected_bbox = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
        expected = np.array(expected_bbox, dtype=np.float32)

        np.testing.assert_allclose(detected_bbox, expected, atol=35.0)

    def test_detects_colorchecker_by_patch_grid_on_dark_bookcloth(self) -> None:
        image, expected_bbox = self._build_dark_bookcloth_chart_scene()

        detection = detect_colorchecker(image)

        self.assertIsNotNone(detection)
        self.assertIn("patch-grid", detection.method)
        xmin, ymin = np.min(detection.corners, axis=0)
        xmax, ymax = np.max(detection.corners, axis=0)
        detected_bbox = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
        expected = np.array(expected_bbox, dtype=np.float32)

        np.testing.assert_allclose(detected_bbox, expected, atol=45.0)

    def test_rejects_border_spanning_dark_region_before_patch_grid_detection(self) -> None:
        image, expected_bbox = self._build_border_spanning_dark_scene()

        detection = detect_colorchecker(image)

        self.assertIsNotNone(detection)
        self.assertIn("patch-grid", detection.method)
        xmin, ymin = np.min(detection.corners, axis=0)
        xmax, ymax = np.max(detection.corners, axis=0)
        detected_bbox = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
        expected = np.array(expected_bbox, dtype=np.float32)

        np.testing.assert_allclose(detected_bbox, expected, atol=45.0)


if __name__ == "__main__":
    unittest.main()
