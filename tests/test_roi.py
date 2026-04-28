import unittest

import numpy as np

from wallpaper_lab.roi import compute_lab_statistics


class RoiStatisticsTests(unittest.TestCase):
    def test_compute_lab_statistics_reports_chroma_and_hue(self) -> None:
        lab_image = np.zeros((2, 2, 3), dtype=np.float32)
        lab_image[0, 0] = (50.0, 0.0, 10.0)
        lab_image[0, 1] = (70.0, 0.0, 20.0)
        mask = np.zeros((2, 2), dtype=bool)
        mask[0, 0] = True
        mask[0, 1] = True

        stats = compute_lab_statistics(lab_image, {"yellow_axis": mask})

        self.assertEqual(list(stats["roi"]), ["yellow_axis"])
        self.assertAlmostEqual(float(stats.loc[0, "mean_C"]), 15.0, places=5)
        self.assertAlmostEqual(float(stats.loc[0, "median_C"]), 15.0, places=5)
        self.assertAlmostEqual(float(stats.loc[0, "mean_h"]), 90.0, places=5)
        self.assertAlmostEqual(float(stats.loc[0, "std_h"]), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
