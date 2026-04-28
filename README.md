# Pigment and Cloth Colour Lab Extractor

This project provides a local Streamlit app for extracting calibrated, image-derived CIE L\*a\*b\*, C\*ab chroma, and h_ab hue-angle values from selected printed pigment-colour or woven bookcloth-colour areas in a sample photograph that includes a ColorChecker Classic target.

Important scope note: the reported values are calibrated image-derived Lab estimates from a photograph. They are not direct spectrophotometer measurements and should be described that way in conservation documentation.

The app includes profile-based extractors for green, vermilion, chrome yellow, altered chrome yellow, chrome-green mixture, muted blue-green bookcloth, and a woven bookcloth preset. Existing wallpaper and bookcloth profiles use the original strict colour-mask workflow by default. The woven bookcloth preset adds material diagnostics that smooth local thread texture, locate compatible thread-shadow pixels, and mark neutral substrate show-through inside locally supported cloth regions. These profiles are heuristic image-colour masks around visible colour families; they do not identify pigment chemistry or replace spectral analysis.

## What the app does

1. Loads the source image.
2. Detects the in-frame ColorChecker automatically or allows manual four-corner registration.
3. Perspective-warps the chart and samples the 24 patches from the chart interior in a repeatable grid order, supporting both landscape 6x4 and portrait 4x6 ColorChecker Classic orientations.
4. Fits a reproducible affine RGB correction from observed chart patch RGB values to ColorChecker Classic sRGB reference values.
5. Applies the correction to the full image.
6. Converts the corrected image to CIE L\*a\*b\* using a reproducible sRGB-to-Lab path and derives C\*ab chroma plus h_ab hue angle in the CIE a\*b\* plane.
7. Isolates selected pigment-colour pixels directly from the corrected image using combined HSV and Lab constraints plus exclusion masks for the chart and lower date board.
8. For woven bookcloth, expands the strict colour seed through material-aware local support so thread shadows and substrate show-through are represented without changing the wallpaper defaults.
9. Supports manual cleanup with inclusion/exclusion polygons and manual ROI polygons.
10. Groups selected pigment or cloth pixels into fragment-level support regions when fragment summaries are requested.
11. Exports the corrected image, pigment/material mask, sampled-pixel overlay, CSV summaries, explicit Delta E00 and Delta E76 ColorChecker diagnostics, and cloth component masks when the woven workflow is used.

## Files

- `app.py`: Streamlit interface.
- `run_sample_analysis.py`: headless verifier / CLI runner for the same pipeline.
- `wallpaper_lab/colorchecker.py`: ColorChecker detection, ordering, and warp helpers.
- `wallpaper_lab/calibration.py`: chart sampling, RGB correction fit, and Lab conversion.
- `wallpaper_lab/segmentation.py`: pigment-colour masking profiles, material-aware woven-cloth masking, exclusions, and fragment grouping support.
- `wallpaper_lab/roi.py`: polygon cleanup, ROI masks, and statistics.
- `wallpaper_lab/export.py`: export helpers for corrected images, masks, component masks, overlays, and diagnostics.
- `wallpaper_lab/references.py`: sample defaults and ColorChecker reference data.

## Run instructions

The app is designed to run locally with Python 3.9+.

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

This installs `rawpy`, which the app uses for full-resolution camera RAW imports such as `.nef`.

### 3. Launch the Streamlit app

```bash
streamlit run app.py
```

The app will load `/Users/hsandaver/Downloads/CCMC628-15.png` automatically if that file is present. You can also upload another image from the sidebar.
Supported uploads include `png`, `jpg`, `jpeg`, `tif`, `tiff`, and common camera RAW files such as `.nef`, `.dng`, `.cr2`, `.cr3`, and `.arw`.

### Streamlit Community Cloud

Deploy from `app.py` at the repository root. The project uses `opencv-python-headless` because Community Cloud runs without desktop GUI libraries. If deployment uses a very new Python runtime and a native imaging dependency fails to import, redeploy the app with a stable Python version such as 3.11 or 3.12 from Community Cloud's Advanced settings.

### 4. Optional: run the pipeline without the UI

```bash
python3 run_sample_analysis.py --image /Users/hsandaver/Downloads/CCMC628-15.png --pigment chrome_yellow
python3 run_sample_analysis.py --image /path/to/bookcloth.png --pigment "woven bookcloth" --cloth-sample-mode dyed_threads
```

That command writes a timestamped export directory under `outputs/`.

## UI workflow

1. Choose the sample image or upload another file.
2. Pick `Automatic` ColorChecker registration or `Manual 4-corner registration`.
3. If manual mode is needed, click the chart corners in this order:
   - top-left
   - top-right
   - bottom-right
   - bottom-left
4. Choose a pigment extractor such as `Green`, `Vermilion`, `Chrome yellow`, `Bookcloth blue-green`, or `Bookcloth blue-green woven`.
5. Choose the material workflow. Existing wallpaper and bookcloth profiles default to `Wallpaper / printed pigment`; `Bookcloth blue-green woven` defaults to `Woven cloth / bookbinding`.
6. Tune the profile thresholds in the sidebar if the initial mask is too broad or too narrow.
7. For woven cloth, tune the cloth controls if needed:
   - `Lab sampling basis` controls whether Lab statistics use the dyed-thread sample only, dyed threads plus shadows, or the whole visible cloth surface
   - `Texture smoothing radius` bridges alternating thread highlights and shadows
   - `Local cloth support radius` and `Minimum local dyed-thread fraction` define the local neighbourhood required before shadow or substrate pixels can be added
   - `Thread-shadow L* allowance` and `a*/b* tolerance` control dark thread recovery
   - `Substrate show-through` controls limit low-chroma substrate fill to locally supported cloth fields
8. Use `Manual mask cleanup` to exclude torn edges, abraded areas, or residual false positives.
9. Choose ROI mode:
   - `Whole pigment mask` for one aggregate result
   - `Per fragment` for separate fragment-level values
   - `Manual polygon ROIs` for custom sampling regions
10. Export the corrected image, mask, overlay, CSV summary, diagnostics, and any woven-cloth component masks.

## Methodology notes

- The calibration step uses the in-frame ColorChecker Classic rather than global white balance alone.
- The chart is sampled after a perspective warp so the patch grid is measured in a stable, repeatable way.
- The RGB correction is an image-derived affine fit into a display-referred sRGB space. This is a practical local correction from the available file, not a raw-camera spectral characterization.
- Lab conversion uses a reproducible sRGB-to-Lab path via `skimage.color.rgb2lab`.
- ROI summaries report CIE L\*a\*b\*, C\*ab chroma, and h_ab hue angle. Hue is calculated with `atan2(b*, a*)`, where red is 0 degrees, yellow is 90 degrees, green is 180 degrees, and blue is 270 degrees.
- Calibration diagnostics report both CIEDE2000 (`delta_e_00_*`) and CIE 1976 (`delta_e_76_*`) chart differences because both conventions appear in conservation and colorimetry literature.
- Pigment segmentation combines:
  - HSV hue and saturation constraints
  - Lab `a*`, `b*`, chroma, and `L*` range constraints
  - exclusion masks for the ColorChecker and the lower date-board region
  - optional edge exclusion to reduce mixed pixels near torn borders
  - morphology cleanup
  - fragment grouping derived from the selected pigment motifs for per-fragment summaries
  - manual polygon refinement when needed
- Woven cloth segmentation adds:
  - a texture-smoothed colour seed to stabilize alternating warp/weft highlights and shadows
  - local cloth support from nearby dyed-thread pixels before any tolerant fill is allowed
  - thread-shadow recovery for darker pixels that remain colour-compatible with the selected cloth profile
  - substrate show-through recovery for low-chroma, low-saturation pixels embedded inside the supported cloth field
  - a `Lab sampling basis` default of dyed threads only, so shadow and substrate diagnostics are saved without shifting colour-match values unless explicitly included
  - saved component masks and `segmentation_diagnostics.csv` so the material-aware additions are auditable
- Built-in pigment defaults:
  - `green`: olive and green printed pigment selections
  - `vermilion`: warm red-orange selections around vermilion/cinnabar-like colour
  - `chrome_yellow`: bright warm-yellow selections around chrome-yellow lead chromate colour
  - `altered_chrome_yellow`: darker yellow-orange to brown candidate selections consistent with reported chrome-yellow darkening trends
  - `chrome_green`: candidate green selections for chrome-yellow/Prussian-blue mixture colours
  - `bookcloth_blue_green`: muted blue-green and green-grey cloth-cover selections using the strict colour-mask workflow
  - `bookcloth_blue_green_woven`: muted blue-green and green-grey cloth-cover selections with woven-cloth diagnostics

## Practical defaults for the supplied sample

For `/Users/hsandaver/Downloads/CCMC628-15.png`, the default settings are tuned to prioritize the green leaf and stem motifs while excluding:

- the white background board
- the date board and label text
- the ColorChecker itself
- the blue border printing
- the pink floral motif
- most torn-edge mixed pixels via edge exclusion

The exact sampled result still depends on the selected extractor, threshold choices, and any manual cleanup polygons you apply.

## Limitations

- The result is only as defensible as the source photograph and chart visibility allow.
- The app assumes a ColorChecker Classic target, and scores the visible 6x4 or rotated 4x6 patch layout before calibration.
- Automatic ColorChecker detection is heuristic. Manual four-corner registration is provided specifically as a fallback.
- The chart reference values are display-referred ColorChecker Classic targets, which is appropriate for a JPEG-style local workflow but not equivalent to direct instrument measurement.
- Severe glare, blur, clipping, metamerism, or uneven illumination can still bias the result.
- Mixed pixels at torn edges and abraded regions are reduced, not perfectly eliminated, by the automated mask.
- The pigment extractors select colour-space neighbourhoods. They should be treated as candidate sampling masks, not pigment identification.
- The altered chrome yellow and chrome-green profiles encode visible-colour behaviour discussed in the reviewed literature, but chemical claims still require reflectance, XRF, Raman, FTIR, XANES, or equivalent analytical evidence.
