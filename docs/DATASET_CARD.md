# Dataset card: SEVIRI night-time IR, Central Europe

An ML-ready stack of Meteosat SEVIRI infrared brightness temperatures for night-time fog and low
stratus detection. Built with `fogseg fetch` (download + crop) and `fogseg build` (stack into Zarr).

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
- **Class balance (preview):** a rough rule (IR_108 ≥ 265 K and BTD > 2.5 K) flags 38.6 % of
  night pixels as fog / low stratus candidates; in 384 of the 608 scenes, candidates are more than
  30 % of that scene's night pixels. The proper labels (step 5) will report the final balance.

## What this dataset is not

- **Not labelled yet.** Fog / low-stratus labels come in a later step.
- **Fog vs. low stratus:** the satellite sees cloud tops. It cannot tell whether a low cloud touches
  the ground; that is checked separately against airport visibility (METAR).
- **Not redistributed.** The data are EUMETSAT data used under the EUMETSAT data licence; this
  repository contains the code to rebuild the dataset, not the data itself.

## Reproduce

```bash
fogseg fetch --workers 2   # 608 scenes, ~4-8 h, resumable (needs a EUMETSAT API key)
fogseg build               # ~1-3 min
```
