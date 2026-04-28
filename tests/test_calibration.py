import unittest

import numpy as np
from skimage import color

from wallpaper_lab.calibration import (
    calibrate_image_from_colorchecker,
    calibration_diagnostics_dataframe,
    fit_calibration_model,
    linear_to_srgb,
    orient_colorchecker_to_reference,
    srgb_to_linear,
)
from wallpaper_lab.references import CALIBRITE_DISPLAY_SRGB


class CalibrationOrientationTests(unittest.TestCase):
    def _build_synthetic_chart(self) -> np.ndarray:
        height, width = 800, 1200
        chart = np.zeros((height, width, 3), dtype=np.float32)

        x_centers = np.linspace(width * 0.13, width * 0.87, 6, dtype=np.float32)
        y_centers = np.linspace(height * 0.12, height * 0.88, 4, dtype=np.float32)
        half_size = 50

        patch_index = 0
        for center_y in y_centers:
            for center_x in x_centers:
                x0 = int(round(center_x - half_size))
                x1 = int(round(center_x + half_size))
                y0 = int(round(center_y - half_size))
                y1 = int(round(center_y + half_size))
                chart[y0:y1, x0:x1] = CALIBRITE_DISPLAY_SRGB[patch_index]
                patch_index += 1

        return chart

    def test_orients_upside_down_chart(self) -> None:
        upright_chart = self._build_synthetic_chart()
        upside_down_chart = np.rot90(upright_chart, k=2).copy()

        oriented_chart, observed_patches, rotation_k = orient_colorchecker_to_reference(
            upside_down_chart,
            CALIBRITE_DISPLAY_SRGB,
        )

        self.assertEqual(rotation_k, 2)
        np.testing.assert_allclose(oriented_chart, upright_chart, atol=1e-6)
        np.testing.assert_allclose(observed_patches, CALIBRITE_DISPLAY_SRGB, atol=1e-6)

    def test_calibrates_portrait_rotated_chart_layout(self) -> None:
        height, width = 1200, 800
        chart = np.zeros((height, width, 3), dtype=np.float32)
        rotated_patches = np.rot90(CALIBRITE_DISPLAY_SRGB.reshape(4, 6, 3), k=3)
        x_centers = np.linspace(width * 0.17, width * 0.83, 4, dtype=np.float32)
        y_centers = np.linspace(height * 0.15, height * 0.85, 6, dtype=np.float32)
        half_size = 50

        for row_index, center_y in enumerate(y_centers):
            for column_index, center_x in enumerate(x_centers):
                x0 = int(round(center_x - half_size))
                x1 = int(round(center_x + half_size))
                y0 = int(round(center_y - half_size))
                y1 = int(round(center_y + half_size))
                chart[y0:y1, x0:x1] = rotated_patches[row_index, column_index]

        corners = np.array(
            [
                [0.0, 0.0],
                [width - 1.0, 0.0],
                [width - 1.0, height - 1.0],
                [0.0, height - 1.0],
            ],
            dtype=np.float32,
        )

        calibration = calibrate_image_from_colorchecker(chart, corners)

        self.assertEqual(calibration.patch_grid_shape, (6, 4))
        self.assertLess(calibration.mean_delta_e_00_before, 0.1)
        self.assertLess(calibration.mean_delta_e_00_after, 0.5)


class CalibrationModelTests(unittest.TestCase):
    def _build_shadow_cast_patches(self) -> np.ndarray:
        reference_linear = srgb_to_linear(CALIBRITE_DISPLAY_SRGB)
        reference_black = reference_linear[0]

        camera_matrix = np.array(
            [
                [1.08, 0.05, 0.01],
                [0.03, 0.87, 0.05],
                [0.02, 0.04, 0.79],
            ],
            dtype=np.float32,
        )
        observed_black = np.array([0.08, 0.03, 0.01], dtype=np.float32)

        observed_linear = np.clip((reference_linear - reference_black) @ camera_matrix + observed_black, 0.0, 1.0)

        shadow_strength = np.linspace(1.0, 0.15, 6, dtype=np.float32)
        observed_linear[:6, 0] += 0.010 * shadow_strength
        observed_linear[:6, 1] += 0.055 * shadow_strength
        observed_linear[:6, 2] += 0.028 * shadow_strength
        observed_linear = np.clip(observed_linear, 0.0, 1.0)
        return linear_to_srgb(observed_linear)

    def test_black_patch_is_anchored_to_reference(self) -> None:
        observed_patches = self._build_shadow_cast_patches()

        model = fit_calibration_model(observed_patches, CALIBRITE_DISPLAY_SRGB)
        corrected_patches = model.apply(observed_patches)

        np.testing.assert_allclose(corrected_patches[0], CALIBRITE_DISPLAY_SRGB[0], atol=3e-3)

    def test_shadow_neutralization_reduces_dark_neutral_cast(self) -> None:
        observed_patches = self._build_shadow_cast_patches()

        model = fit_calibration_model(observed_patches, CALIBRITE_DISPLAY_SRGB)
        corrected_patches = model.apply(observed_patches)

        observed_neutral_spread = np.ptp(observed_patches[1:4], axis=1).mean()
        corrected_neutral_spread = np.ptp(corrected_patches[1:4], axis=1).mean()
        self.assertLess(corrected_neutral_spread, observed_neutral_spread * 0.2)

        reference_lab = color.rgb2lab(CALIBRITE_DISPLAY_SRGB.reshape(1, -1, 3)).reshape(-1, 3)
        observed_lab = color.rgb2lab(observed_patches.reshape(1, -1, 3)).reshape(-1, 3)
        corrected_lab = color.rgb2lab(corrected_patches.reshape(1, -1, 3)).reshape(-1, 3)

        mean_delta_e_before = float(np.mean(color.deltaE_ciede2000(observed_lab, reference_lab)))
        mean_delta_e_after = float(np.mean(color.deltaE_ciede2000(corrected_lab, reference_lab)))
        self.assertLess(mean_delta_e_after, mean_delta_e_before * 0.45)

        blue_patch = corrected_patches[11]
        self.assertGreater(blue_patch[2], blue_patch[1] + 0.18)
        self.assertGreater(blue_patch[2], blue_patch[0] + 0.22)

    def test_calibration_diagnostics_names_delta_e_conventions(self) -> None:
        chart = CalibrationOrientationTests()._build_synthetic_chart()
        corners = np.array(
            [
                [0.0, 0.0],
                [1199.0, 0.0],
                [1199.0, 799.0],
                [0.0, 799.0],
            ],
            dtype=np.float32,
        )

        calibration = calibrate_image_from_colorchecker(chart, corners)
        diagnostics = calibration_diagnostics_dataframe(calibration)

        self.assertIn("delta_e_00_after", diagnostics.columns)
        self.assertIn("delta_e_76_after", diagnostics.columns)


if __name__ == "__main__":
    unittest.main()
