# Dataset card: SEVIRI night-time IR, Central Europe

An ML-ready stack of Meteosat SEVIRI infrared brightness temperatures for night-time fog and low
stratus detection. Built with `fogseg fetch` (download + crop) and `fogseg build` (stack into Zarr).
Weak fog / low-stratus labels for every scene are made with `fogseg label` and described in
[Labels](#labels).

## Summary

| | |
|---|---|
| Format | Zarr v3 store, `seviri_ceu.zarr` (803 MB on disk) |
| Main variable | `bt` (time, channel, y, x) = (608, 8, 241, 462), float32, Kelvin |
| Other variables | `sza` (time, y, x) solar zenith angle in degrees, at the time the region was scanned; `platform`, `source_file` per scene |
| Coordinates | `time` (UTC, nominal start of the 15-minute repeat cycle), `channel` + `wavelength` (µm), `y`/`x` (metres, geostationary projection), `lat`/`lon` (y, x) |
| Chunks | 8 scenes per chunk (~28 MB of `bt` uncompressed) |
| Source | EUMETSAT Data Store, `EO:EUM:DAT:MSG:HRSEVIRI` (SEVIRI Level 1.5, 0° service) |
| Satellite | Meteosat-10 (MSG-3) for all 608 scenes |
| Built with | `fogseg` 0.1.0, satpy reader `seviri_l1b_native`, `calibration="brightness_temperature"` |

Open it with:

```python
import xarray as xr

ds = xr.open_zarr("seviri_ceu.zarr", consolidated=False)
```

The numeric arrays use core Zarr v3 data types. The two text variables (`platform`,
`source_file`) use zarr-python's fixed-length UTF-32 string type, an extension that non-Python
Zarr readers may not support. The store has no consolidated metadata (not part of the v3 standard).

## Spatial coverage

- Region of interest: 2–20°E, 43–55°N (Benelux, France, Germany, Switzerland, Austria, Czechia,
  Poland, northern Italy): fog hotspots such as the Po Valley, the Swiss Plateau and the North German Plain.
- Grid: SEVIRI's **native geostationary grid**, cropped, **not resampled**, so there are no
  interpolation artefacts. Every scene has exactly the same pixels (checked during the build).
- The crop is the smallest rectangle *in the satellite projection* that contains the region, so its
  corners reach a little further: latitude 42.6–56.1°N, longitude 1.6–27.4°E.
- Pixel size is ~3 km at the sub-satellite point, but over Central Europe the slanted view makes
  pixels ~3.3 km east–west × ~5.8 km north–south (medians; up to 4.1 × 8.6 km in the far
  north-east corner).
- Row 0 is the northern edge, column 0 the western edge (`upper_right_corner="NE"`).

## Temporal coverage

| | |
|---|---|
| Seasons | 2023-24 (1 Oct 2023 – 29 Feb 2024) and 2024-25 (1 Oct 2024 – 28 Feb 2025): 304 scenes each |
| Nights | every second night |
| Slots (UTC) | 21:00, 00:00, 03:00, 05:00: 152 scenes each |
| Range | 2023-10-01 21:00 → 2025-03-01 05:00 |

**Intended split (by time, never random):** train on 2023-24 (with a held-out validation block,
e.g. February 2024), test on 2024-25. Neighbouring scenes of one night are nearly identical, so a
random split would leak test information into training.

## Channels

| channel | λ (µm) | mean (K) | std (K) | min (K) | max (K) |
|---|---|---|---|---|---|
| IR_039 | 3.9 | 259.78 | 13.09 | 204.8 | 290.3 |
| WV_062 | 6.2 | 230.17 | 5.58 | 200.7 | 250.4 |
| WV_073 | 7.3 | 244.81 | 9.48 | 201.3 | 267.9 |
| IR_087 | 8.7 | 259.24 | 14.81 | 201.3 | 290.8 |
| IR_097 | 9.7 | 238.48 | 8.89 | 208.3 | 264.9 |
| IR_108 | 10.8 | 260.76 | 15.80 | 200.7 | 293.7 |
| IR_120 | 12.0 | 259.78 | 16.21 | 199.0 | 292.9 |
| IR_134 | 13.4 | 245.11 | 10.84 | 200.6 | 268.1 |

Statistics over all 608 scenes and all pixels. They are descriptive only; the model's
normalisation must be computed on the **training season only**.

The four solar channels (VIS006, VIS008, IR_016, HRV) are left out: they carry no signal at night.

## Quality notes

- **Invalid values:** 1,238 values (0.0002 % of `bt`), all in IR_039, were ~0 K and are set to
  NaN. They occur under very cold cloud tops (IR_108 ≈ 210 K), where the 3.9 µm signal is below
  the instrument noise. These pixels are deep convective cloud, never fog. Threshold: < 150 K.
- **Scan time vs. `time`:** `time` is the nominal start of the repeat cycle (e.g. 05:00:00).
  SEVIRI scans from south to north, so Central Europe is observed 10.7–11.5 minutes later. `sza`
  is computed at `time` plus the scan delay of the middle of the region (11.1 minutes, derived from
  the grid), which keeps its error below ~0.1°. Using the nominal time instead would
  **overestimate** it by up to ~2° in the 05:00 scenes (the sun is rising), letting twilight
  pixels pass the night mask.
- **Twilight at 05:00 UTC:** 13.3 % of pixels in the 05:00 scenes have a solar zenith angle of
  95° or less (minimum 83.1°, early October and late February, eastern part of the region).
  Reflected sunlight contaminates IR_039 there; use `sza > 95` as a night mask. All 21:00, 00:00
  and 03:00 pixels are night; 96.7 % of all pixels are night.
- **BTD convention:** the fog signal used in this project is the **uncorrected** difference
  `IR_108 − IR_039`. satpy's `night_fog` composite applies a CO₂ correction to IR_039 that shifts
  this difference by ~5 K over low cloud; we follow the EUMETSAT Night Microphysics recipe instead.

## Labels

Weak labels for fog and low stratus, made by `fogseg label` from the dataset above. There is one
Zarr store per mode, next to the dataset: `labels_fixed.zarr` and `labels_dynamic.zarr` (folder
`labels.output_dir` in [`configs/default.yaml`](../configs/default.yaml)). They have the same
`time`, `y` and `x` as the dataset, so `xr.merge([dataset, labels])` lines them up. The default mode
is `fixed`.

| Value | Meaning |
|---|---|
| `1` | fog or low stratus, visible from space |
| `0` | not fog / low stratus |
| `255` | ignore: data not trustworthy (twilight, missing values) or too uncertain to call; the loss skips it |

| Variable | Dimensions | Content |
|---|---|---|
| `label` | time, y, x | `uint8`: `0`, `1` or `255` |
| `btd_threshold` | time | BTD threshold used for the scene (K) |
| `threshold_source` | time | `fixed`, `dynamic` or `fallback` |
| `fog_fraction` | time | share of the **decided** pixels (label 0 or 1, not 255) that are `1`; NaN if a scene has none |
| `ignore_fraction` | time | share of **all** pixels that are `255` |

The store attributes hold the mode, every threshold used, the `fogseg` version and the source
dataset, so a store explains itself.

### The rule

```mermaid
flowchart TD
    px["one pixel"] --> trust{"night and valid?<br/>sza above 95°, no missing value"}
    trust -- no --> ignore1["255 ignore"]
    trust -- yes --> other{"warm top, no cirrus, smooth?"}
    other -- no --> zero["0 not fog"]
    other -- yes --> band{"BTD within 0.5 K of the threshold,<br/>above or below?"}
    band -- yes --> ignore2["255 ignore"]
    band -- no --> high{"BTD above the threshold?"}
    high -- no --> zero
    high -- yes --> tiny{"in a patch of fewer<br/>than 5 pixels?"}
    tiny -- yes --> ignore3["255 ignore"]
    tiny -- no --> one["1 fog / low stratus"]
```

1. **Trust.** A pixel is `255` if the solar zenith angle is 95° or less (reflected sunlight in
   IR_039) or if IR_039, IR_108 or IR_120 is missing.
2. **Three tests besides the BTD.** A trusted pixel can only be fog if it passes all of them; a pixel
   that fails one becomes `0`:

   | Test | Rule |
   |---|---|
   | Low cloud | `IR_108` at least 255 K |
   | Not thin cirrus | `IR_108 − IR_120` below 1.5 K |
   | Smooth top | standard deviation of `IR_108` in a 3 × 3 window below 1.5 K |

3. **BTD against the threshold** (water droplets), for the pixels that passed step 2. If `IR_108 − IR_039`
   is within 0.5 K of the BTD threshold, **on either side**, the pixel is too uncertain and becomes `255`.
   Above the threshold plus 0.5 K it is a fog candidate; below the threshold minus 0.5 K it is `0`.
4. **Small patches.** A fog candidate in a group of fewer than 5 connected pixels (about 100 km² here)
   becomes `255`, since a small patch can be real valley fog. The groups are counted on the candidates
   that are left after step 3.
5. **Everything else** that is trusted becomes `0`.

All values are in the `labels` section of [`configs/default.yaml`](../configs/default.yaml): `night_min_sza`
95.0, `warm_min_bt108` 255.0, `cirrus_max_split` 1.5, `smooth_max_std` 1.5, `uncertainty_band` 0.5,
`min_blob_pixels` 5, `btd_fixed` 4.0. They were measured on the 2023-24 training season only
([`docs/notes/btd_exploration.md`](notes/btd_exploration.md)); `btd_fixed` was then tuned against station
observations of the same season ([`docs/notes/station_validation.md`](notes/station_validation.md)). The 2024-25
season was not used to set any of them.

### The BTD threshold

- **`fixed` mode:** 4.0 K (`btd_fixed`) in every scene. Tuned against 1175 stations in the training season; it
  replaced 5.0 K, which missed most of the fog.
- **`dynamic` mode:** clear ground has a BTD peak of its own in each scene, and it moves from night to
  night (1.9 to 4.6 K, 1st to 99th percentile in the training season). Per scene, the BTD of
  the trusted, warm, non-cirrus pixels is put in a histogram (bins of 0.25 K from −2 to 8 K, smoothed over
  3 bins). The clear-ground peak is its highest bin inside the window 1.5 to 4.75 K, and the
  threshold is that peak plus 1.5 K, the distance at which the valley between clear ground and low
  cloud lay in the training season. The result is limited to 3.5 to 6.0 K; with the current window the upper limit is never reached, because the
  highest accepted peak is 4.375 K, so the highest threshold is 5.875 K.
  The scene falls back to `btd_fixed`, 4.0 K (`threshold_source = fallback`), if the histogram has fewer
  than 2,000 pixels inside its range of -2 to 8 K or the highest bin is at the edge of the window, so no real peak was found.

### Label statistics

Computed from the stores on the 304 scenes of each season (a night belongs to the season in which it
starts, so the morning slots after midnight count with the night before). "Mean fog fraction" is the
mean over scenes of `fog_fraction`, "mean ignore fraction" the mean of `ignore_fraction`.

| Mode | Season | Mean fog fraction | Mean ignore fraction | Dynamic threshold | Fallback to 4.0 K |
|---|---|---|---|---|---|
| fixed | 2023-24 (training) | 0.1250 | 0.1464 | – | – |
| fixed | 2024-25 (test) | 0.2369 | 0.1619 | – | – |
| dynamic | 2023-24 (training) | 0.0614 | 0.0904 | 293 scenes | 11 scenes |
| dynamic | 2024-25 (test) | 0.1470 | 0.1194 | 288 scenes | 16 scenes |

In the training season the dynamic threshold has a median of 5.1 K (middle half 4.6 to 5.4 K,
range 3.5 to 5.9 K; [btd_exploration](notes/btd_exploration.md)), so the dynamic labels mark much
less fog than the fixed 4.0 K. Fog is a minority class: the mean fog fraction is about 13 % of the
decided pixels in the training season and about 24 % in the test season (fixed labels), so the class
balance differs between the two. The ignore fraction is higher with 4.0 K than it was with 5.0 K
(0.146 against 0.096 in the training season), because the 0.5 K band on either side of the threshold now lies where many
pixels are. The test-season numbers describe the labels only; nothing was tuned on them.

### How the labels were checked

15 training-season scenes were looked at by eye, in both modes, before any model was trained
([`docs/notes/label_check.md`](notes/label_check.md); the scenes are listed in
[`configs/check_scenes.yaml`](../configs/check_scenes.yaml)). That check chose `dynamic` at first.

The labels were then scored against 1175 airport (METAR) and weather stations (SYNOP)
([`docs/notes/station_validation.md`](notes/station_validation.md)). A report counts as fog or low stratus when
visibility is below 1 km or a broken or overcast deck has its base below 1 km. For the cases the satellite can see:

| Labels | Season | POD | FAR | CSI (95 % interval) |
|---|---|---|---|---|
| `dynamic`, peak + 1.5 K, fallback 5.0 K (before) | 2023-24 (training) | 0.27 | 0.19 | 0.26 (0.20–0.32) |
| `fixed`, 4.0 K (now the default) | 2023-24 (training) | 0.50 | 0.28 | 0.42 (0.37–0.47) |
| `fixed`, 4.0 K | 2024-25 (test, scored once) | 0.70 | 0.19 | 0.60 (0.55–0.64) |

The threshold was chosen on the Heidke and Peirce skill scores. CSI alone peaks at 3.5 K (0.44), but at 3.5 K far more
clear ground is labelled fog. The test season scores higher because it had more widespread low stratus and less fog
under high cloud, not because the labels changed; the details are in the note.

### Limitations

- **Weak labels are not ground truth.** They come from a rule on satellite data. Against stations they
  find half of the visible fog and low stratus (POD 0.50), and 28 % of what they call fog is not (FAR).
- **Fog under higher cloud gets `0`**, not `255`. 51 % of the fog or low-stratus reports in the training
  season lie under a pixel that fails the high-cloud tests. The satellite sees only the top, so a model
  trained on these labels learns that fog is absent under thick cloud; that is a property of the
  data, not of the weather.
- **Fog and low stratus cannot be told apart** from space. `1` means a low water-cloud deck that may or may not
  touch the ground.
- **Snow-covered ground** can look like low cloud in some of the tests. The check scenes were
  not chosen to include snow, so this has not been examined.
- **A low-cloud deck inside the clear-ground window.** If a deck covers most of the region and its BTD
  is only slightly above the ground, the dynamic method may take the deck for clear ground and set the
  threshold too high (for example 21 November 2023, 00:00). There is no cloud mask to decide this.
- **Fallback nights (dynamic mode).** When the histogram peak lies at the upper edge of the window,
  that hump is usually a low-cloud deck, not clear ground. The stations confirm it: on those scenes
  the old fallback of 5.0 K had CSI 0.30, the new 4.0 K has 0.59. This affects 11 of 304 training scenes
  and 16 of 304 test scenes.
- **The viewing-angle drift is not corrected.** The clear-ground BTD rises by about 1 K from 54° to
  65.5° viewing angle, which in this region means from south to north (the two are almost the
  same thing, so the effect is mixed with surface type and air temperature). The stations confirm
  it: at cloud-free stations the median BTD is 3.0 K in 43–47° N and 3.8 to 3.9 K in 51–56° N. With
  one threshold for the whole region, 26 % of the cloud-free reports in 51–56° N are labelled fog,
  against 5 % in 43–47° N.
- **Many reported fog cases look like clear ground** in the BTD. At the station pixels, fog becomes
  more likely than clear ground only about 0.5 K above the threshold. 60 % of the missed visible fog
  fails the smoothness test (broken or patchy decks). No threshold separates these.

## What this dataset is not

- **Not ground truth.** The labels are a physical rule, not observations; see the limitations above.
- **Fog vs. low stratus:** the satellite sees cloud tops. It cannot tell whether a low cloud touches
  the ground: against visibility below 1 km alone, 92 % of the labelled fog is not ground fog.
- **Not redistributed.** The data are EUMETSAT data used under the EUMETSAT data licence; this
  repository contains the code to rebuild the dataset, not the data itself.

## Reproduce

```bash
fogseg fetch --workers 2     # 608 scenes, ~4-8 h, resumable (needs a EUMETSAT API key)
fogseg build                 # ~1-3 min
fogseg label --mode fixed    # labels_fixed.zarr
fogseg label --mode dynamic  # labels_dynamic.zarr
fogseg stations              # station reports for the check; SYNOP takes many hours (Ogimet rate limit)
fogseg validate --season 2023-24
```
