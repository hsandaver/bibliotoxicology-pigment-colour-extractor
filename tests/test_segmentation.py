import unittest

import numpy as np

from wallpaper_lab.segmentation import (
    SegmentationParameters,
    build_pigment_mask,
    build_pigment_mask_result,
    get_pigment_profile,
)


class TestPigmentSegmentation(unittest.TestCase):
    def test_get_pigment_profile_accepts_vermillion_alias(self):
        profile = get_pigment_profile("vermillion")

        self.assertEqual(profile.key, "vermilion")

    def test_get_pigment_profile_accepts_paper_derived_aliases(self):
        self.assertEqual(get_pigment_profile("darkened chrome yellow").key, "altered_chrome_yellow")
        self.assertEqual(get_pigment_profile("Brunswick green").key, "chrome_green")
        self.assertEqual(get_pigment_profile("bookcloth").key, "bookcloth_blue_green")
        self.assertEqual(get_pigment_profile("woven bookcloth").key, "bookcloth_blue_green_woven")

    def test_vermilion_profile_selects_red_orange_pixels(self):
        image = np.zeros((4, 4, 3), dtype=np.float32)
        image[:] = (0.45, 0.45, 0.45)
        image[0, 0] = (0.89, 0.26, 0.20)
        image[0, 1] = (1.0, 0.65, 0.0)

        mask = build_pigment_mask(
            image,
            self._empty_chart_corners(),
            self._params_for("vermilion"),
        )

        self.assertTrue(mask[0, 0])
        self.assertFalse(mask[0, 1])

    def test_chrome_yellow_profile_selects_warm_yellow_pixels(self):
        image = np.zeros((4, 4, 3), dtype=np.float32)
        image[:] = (0.45, 0.45, 0.45)
        image[0, 0] = (1.0, 0.65, 0.0)
        image[0, 1] = (0.89, 0.26, 0.20)

        mask = build_pigment_mask(
            image,
            self._empty_chart_corners(),
            self._params_for("chrome_yellow"),
        )

        self.assertTrue(mask[0, 0])
        self.assertFalse(mask[0, 1])

    def test_altered_chrome_yellow_profile_selects_darker_yellow_brown_pixels(self):
        image = np.zeros((4, 4, 3), dtype=np.float32)
        image[:] = (0.45, 0.45, 0.45)
        image[0, 0] = (0.65, 0.42, 0.12)
        image[0, 1] = (0.20, 0.35, 0.70)

        mask = build_pigment_mask(
            image,
            self._empty_chart_corners(),
            self._params_for("altered_chrome_yellow"),
        )

        self.assertTrue(mask[0, 0])
        self.assertFalse(mask[0, 1])

    def test_chrome_green_profile_selects_mixed_green_pixels(self):
        image = np.zeros((4, 4, 3), dtype=np.float32)
        image[:] = (0.45, 0.45, 0.45)
        image[0, 0] = (0.35, 0.55, 0.30)
        image[0, 1] = (1.0, 0.65, 0.0)

        mask = build_pigment_mask(
            image,
            self._empty_chart_corners(),
            self._params_for("chrome_green"),
        )

        self.assertTrue(mask[0, 0])
        self.assertFalse(mask[0, 1])

    def test_bookcloth_profile_selects_muted_blue_green_pixels(self):
        image = np.zeros((4, 4, 3), dtype=np.float32)
        image[:] = (0.45, 0.45, 0.45)
        image[0, 0] = (0.22, 0.30, 0.32)
        image[0, 1] = (0.30, 0.30, 0.30)

        mask = build_pigment_mask(
            image,
            self._empty_chart_corners(),
            self._params_for("bookcloth_blue_green"),
        )

        self.assertTrue(mask[0, 0])
        self.assertFalse(mask[0, 1])

    def test_printed_profiles_keep_strict_mask_workflow_by_default(self):
        params = SegmentationParameters.from_profile(
            "green",
            chart_padding_px=0,
            bottom_exclusion_fraction=1.0,
            edge_exclusion_px=0,
            min_area=1,
        )
        image = np.zeros((4, 4, 3), dtype=np.float32)
        image[:] = (0.45, 0.45, 0.45)
        image[0, 0] = (0.35, 0.55, 0.30)

        result = build_pigment_mask_result(image, self._empty_chart_corners(), params)

        self.assertEqual(params.material_mode, "printed")
        self.assertTrue(result.pigment_mask[0, 0])
        self.assertEqual(result.material_masks, {})

    def test_existing_bookcloth_profile_keeps_printed_workflow_by_default(self):
        params = SegmentationParameters.from_profile("bookcloth_blue_green")

        self.assertEqual(params.material_mode, "printed")
        self.assertEqual(params.edge_exclusion_px, 4)
        self.assertEqual(params.bottom_exclusion_fraction, 0.88)

    def test_woven_bookcloth_profile_uses_woven_cloth_workflow_by_default(self):
        params = SegmentationParameters.from_profile("bookcloth_blue_green_woven")

        self.assertEqual(params.material_mode, "woven_cloth")
        self.assertEqual(params.cloth_sample_mode, "dyed_threads")
        self.assertEqual(params.edge_exclusion_px, 0)
        self.assertGreaterEqual(params.bottom_exclusion_fraction, 0.98)

    def test_woven_cloth_workflow_recovers_thread_shadow_and_substrate_showthrough(self):
        image = self._woven_bookcloth_scene()
        chart_corners = self._empty_chart_corners()
        strict_params = SegmentationParameters.from_profile(
            "bookcloth_blue_green",
            material_mode="printed",
            chart_padding_px=0,
            bottom_exclusion_fraction=1.0,
            edge_exclusion_px=0,
            min_area=1,
        )
        cloth_params = SegmentationParameters.from_profile(
            "bookcloth_blue_green_woven",
            chart_padding_px=0,
            bottom_exclusion_fraction=1.0,
            edge_exclusion_px=0,
            min_area=1,
            cloth_sample_mode="surface_appearance",
        )

        strict_mask = build_pigment_mask(image, chart_corners, strict_params)
        cloth_result = build_pigment_mask_result(image, chart_corners, cloth_params)
        cloth_mask = cloth_result.pigment_mask
        cloth_region = np.zeros(cloth_mask.shape, dtype=bool)
        cloth_region[20:60, 20:60] = True

        self.assertLess(float(strict_mask[cloth_region].mean()), 0.80)
        self.assertGreater(float(cloth_mask[cloth_region].mean()), 0.95)
        self.assertFalse(cloth_mask[10, 10])
        self.assertTrue(cloth_mask[26, 27])
        self.assertTrue(cloth_mask[34, 25])
        self.assertTrue(
            cloth_result.material_masks["bookcloth_blue_green_woven_mask_substrate_showthrough"][26, 27]
        )
        self.assertTrue(
            cloth_result.material_masks["bookcloth_blue_green_woven_mask_thread_shadow"][34, 25]
        )

    def test_woven_cloth_default_samples_dyed_threads_without_shadow_or_substrate_fill(self):
        image = self._woven_bookcloth_scene()
        chart_corners = self._empty_chart_corners()
        dyed_thread_params = SegmentationParameters.from_profile(
            "bookcloth_blue_green_woven",
            chart_padding_px=0,
            bottom_exclusion_fraction=1.0,
            edge_exclusion_px=0,
            min_area=1,
        )
        surface_params = SegmentationParameters.from_profile(
            "bookcloth_blue_green_woven",
            chart_padding_px=0,
            bottom_exclusion_fraction=1.0,
            edge_exclusion_px=0,
            min_area=1,
            cloth_sample_mode="surface_appearance",
        )

        dyed_thread_result = build_pigment_mask_result(image, chart_corners, dyed_thread_params)
        surface_result = build_pigment_mask_result(image, chart_corners, surface_params)

        self.assertFalse(dyed_thread_result.pigment_mask[34, 25])
        self.assertFalse(dyed_thread_result.pigment_mask[26, 27])
        self.assertTrue(dyed_thread_result.material_masks["bookcloth_blue_green_woven_mask_thread_shadow"][34, 25])
        self.assertTrue(
            dyed_thread_result.material_masks["bookcloth_blue_green_woven_mask_substrate_showthrough"][26, 27]
        )
        self.assertGreater(int(surface_result.pigment_mask.sum()), int(dyed_thread_result.pigment_mask.sum()))

    @staticmethod
    def _empty_chart_corners() -> np.ndarray:
        return np.zeros((4, 2), dtype=np.float32)

    @staticmethod
    def _params_for(pigment_key: str) -> SegmentationParameters:
        return SegmentationParameters.from_profile(
            pigment_key,
            chart_padding_px=0,
            bottom_exclusion_fraction=1.0,
            edge_exclusion_px=0,
            min_area=1,
            fragment_group_dilation_radius=1,
        )

    @staticmethod
    def _woven_bookcloth_scene() -> np.ndarray:
        image = np.zeros((80, 80, 3), dtype=np.float32)
        image[:] = (0.62, 0.62, 0.60)
        image[20:60, 20:60] = (0.22, 0.30, 0.32)
        image[20:60, 20:60:5] = (0.11, 0.16, 0.18)
        image[20:60:6, 20:60] = (0.10, 0.15, 0.17)

        for y in range(26, 56, 10):
            for x in range(27, 57, 10):
                image[y:y + 2, x:x + 2] = (0.50, 0.50, 0.495)
        return image


if __name__ == "__main__":
    unittest.main()
