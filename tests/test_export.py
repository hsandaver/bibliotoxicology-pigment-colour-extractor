import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from wallpaper_lab.export import save_analysis_outputs
from wallpaper_lab.segmentation import get_pigment_profile


class ExportTests(unittest.TestCase):
    def test_save_analysis_outputs_writes_cloth_component_masks_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            image = np.zeros((3, 3, 3), dtype=np.float32)
            pigment_mask = np.zeros((3, 3), dtype=bool)
            pigment_mask[1, 1] = True
            shadow_mask = np.zeros((3, 3), dtype=bool)
            shadow_mask[0, 1] = True
            segmentation_df = pd.DataFrame(
                [
                    {
                        "mask": "bookcloth_blue_green_woven_mask",
                        "role": "final_sample_mask",
                        "material_mode": "woven_cloth",
                        "pixel_count": 1,
                    }
                ]
            )

            files = save_analysis_outputs(
                output_dir=output_dir,
                corrected_rgb=image,
                pigment_mask=pigment_mask,
                overlay_rgb=image,
                summary_df=pd.DataFrame([{"roi": "combined_bookcloth_blue_green_woven", "pixel_count": 1}]),
                calibration_df=pd.DataFrame([{"patch": 1, "delta_e_00_after": 0.0}]),
                segmentation_df=segmentation_df,
                extra_masks={"bookcloth blue/green shadow mask": shadow_mask},
                pigment_profile=get_pigment_profile("bookcloth_blue_green_woven"),
            )

            self.assertTrue(files["bookcloth_blue_green_woven_mask"].exists())
            self.assertTrue(files["bookcloth_blue_green_shadow_mask"].exists())
            self.assertTrue(files["segmentation_diagnostics_csv"].exists())


if __name__ == "__main__":
    unittest.main()
