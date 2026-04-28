import unittest

import numpy as np

from wallpaper_lab.color_metrics import (
    delta_e_cie1976,
    lab_chroma,
    lab_hue_degrees,
    weighted_circular_mean_degrees,
)


class ColorMetricsTests(unittest.TestCase):
    def test_lab_chroma_and_hue_follow_cie_ab_polar_axes(self) -> None:
        lab = np.array(
            [
                [50.0, 10.0, 0.0],
                [50.0, 0.0, 10.0],
                [50.0, -10.0, 0.0],
                [50.0, 0.0, -10.0],
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(lab_chroma(lab), np.full(4, 10.0), atol=1e-6)
        np.testing.assert_allclose(lab_hue_degrees(lab), [0.0, 90.0, 180.0, 270.0], atol=1e-6)

    def test_delta_e_cie1976_is_euclidean_lab_distance(self) -> None:
        first = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        second = np.array([[3.0, 4.0, 12.0]], dtype=np.float32)

        np.testing.assert_allclose(delta_e_cie1976(first, second), [13.0], atol=1e-6)

    def test_weighted_circular_mean_wraps_cleanly(self) -> None:
        angle = weighted_circular_mean_degrees(
            np.array([350.0, 10.0], dtype=np.float32),
            np.array([1.0, 1.0], dtype=np.float32),
        )

        self.assertTrue(angle < 1.0 or angle > 359.0)


if __name__ == "__main__":
    unittest.main()
