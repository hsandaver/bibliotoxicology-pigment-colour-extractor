import unittest

import numpy as np

from wallpaper_lab.pipeline import _apply_cleanup_exclusions_to_material_masks
from wallpaper_lab.roi import PolygonOperation


class PipelineMaterialMaskTests(unittest.TestCase):
    def test_material_diagnostics_keep_non_sampled_components_after_cleanup(self):
        shadow = np.zeros((5, 5), dtype=bool)
        shadow[1, 1] = True
        substrate = np.zeros((5, 5), dtype=bool)
        substrate[3, 3] = True

        masks = _apply_cleanup_exclusions_to_material_masks(
            {
                "shadow": shadow,
                "substrate": substrate,
            },
            [],
        )

        self.assertTrue(masks["shadow"][1, 1])
        self.assertTrue(masks["substrate"][3, 3])

    def test_material_diagnostics_respect_cleanup_exclusions(self):
        shadow = np.ones((5, 5), dtype=bool)
        masks = _apply_cleanup_exclusions_to_material_masks(
            {"shadow": shadow},
            [
                PolygonOperation(
                    name="remove_corner",
                    points=[(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)],
                    operation="exclude",
                )
            ],
        )

        self.assertFalse(masks["shadow"][0, 0])
        self.assertTrue(masks["shadow"][4, 4])


if __name__ == "__main__":
    unittest.main()
