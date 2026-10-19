# Fog / Low-Stratus Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce per-pixel `0/1/255` fog / low-stratus labels for all 608 scenes, with a fixed and a dynamic BTD threshold mode, plus the tools to check them visually.

**Architecture:** Pure rule functions (`labels.py`, `dynamic_threshold.py`) that work on numpy arrays of one scene; one I/O module (`label_store.py`) that runs them over `seviri_ceu.zarr` in batches and writes `labels_<mode>.zarr` with the same safety pattern as `build.py`; `inspect.py` and `check_scenes.py` for figures and the check-scene list; new `fogseg` sub-commands wire them up. Thresholds live in a new `labels:` section of `configs/default.yaml`, measured in an exploration notebook before they are fixed.

**Tech Stack:** Python ≥ 3.11, numpy, xarray, zarr v3, scipy (`ndimage`), matplotlib, pytest, ruff 0.17.

**Spec:** `docs/superpowers/specs/2026-10-18-fog-labels-design.md`

## Global Constraints

- Label values: `0` not fog, `1` fog / low stratus, `255` ignore; stored as `uint8`.
- One label store per mode: `labels_fixed.zarr`, `labels_dynamic.zarr`, next to `seviri_ceu.zarr`; same `time`, `y`, `x` coordinates as the dataset.
- Night means `sza > 95°`; `sza ≤ 95°` → `255`. Missing IR_039, IR_108 or IR_120 → `255`.
- BTD is the **uncorrected** `IR_108 − IR_039` (no CO₂ correction).
- Thresholds are tuned on the **training season 2023-24 only**; never on 2024-25 scenes.
- Writes are atomic: `<name>.zarr.part`, renamed on success, deleted on any error; no overwrite without `--overwrite`.
- Tests use synthetic arrays only (no data, no credentials); they run in CI on Python 3.11–3.13.
- Code style (project owner's rules): comments only explain a non-obvious *why*, never restate code; prefer named steps over dense one-liners; `ruff check` and `ruff format --check` clean with ruff 0.17.
- Provisional starting values (spec Sections 4–5), replaced by measured values in Task 2: `btd_fixed` 2.5 K, `warm_min_bt108` 260 K, `cirrus_max_split` 2.0 K, `smooth_max_std` 1.5 K, `uncertainty_band` 0.5 K, `min_blob_pixels` 5; dynamic: bins 0.25 K over −2…+8 K, `min_pixels` 2000, `min_hump_share` 0.05, `max_valley_ratio` 0.5.

## Review Focus

1. **A scene with no night pixels** (05:00 twilight) or no histogram pixels: labels are all `255`, the dynamic mode falls back, and `fog_fraction` is `NaN`, not a crash or a division-by-zero warning. → test in Task 3 and Task 5.
2. **NaN next to valid pixels in the 3×3 smoothness window** (the 1,238 invalid IR_039 values): the standard deviation ignores missing neighbours instead of turning valid neighbours into NaN. → test in Task 3.
3. **Fog blobs touching each other only diagonally or touching the image edge** count as one connected blob (8-connectivity) and edge blobs are not dropped for being at the edge. → test in Task 3.
4. **Rerunning `fogseg label` after changing thresholds**: refused without `--overwrite`; with it, the old store survives until the new one is complete. → test in Task 5.
5. **`fogseg inspect` with a time that is not in the dataset** (e.g. a skipped night): a clear error that lists the nearest available times. → test in Task 6.

---

## File Structure

| File | Responsibility |
|---|---|
| `configs/default.yaml` (modify) | new `labels:` section |
| `pyproject.toml` (modify) | declare `scipy`, `matplotlib` |
| `src/fogseg/labels.py` (create) | `LabelParams`, label constants, trust mask, fog tests, cleanup, `label_scene` |
| `src/fogseg/dynamic_threshold.py` (create) | `DynamicParams`, `ThresholdChoice`, histogram valley search |
| `src/fogseg/label_store.py` (create) | `label_dataset`: run the rule over the Zarr dataset, write the label store |
| `src/fogseg/inspect.py` (create) | Night Microphysics RGB, four-panel check figure, nearest-time lookup |
| `src/fogseg/check_scenes.py` (create) | per-scene statistics and the reproducible check-scene selection |
| `src/fogseg/explore.py` (create) | satellite viewing zenith angle, helpers for the exploration notebook |
| `src/fogseg/cli.py` (modify) | `label`, `inspect`, `select-scenes` sub-commands |
| `notebooks/02_explore_btd.ipynb` (create) | Step 0 exploration |
| `docs/notes/btd_exploration.md` (create) | findings and the measured threshold values, with figures |
| `configs/check_scenes.yaml` (create, generated) | the check-scene list |
| `tests/test_labels.py`, `tests/test_dynamic_threshold.py`, `tests/test_label_store.py`, `tests/test_inspect.py`, `tests/test_check_scenes.py`, `tests/test_explore.py` (create) | unit tests |
| `tests/helpers.py` (modify) | `make_label_dataset(...)`: tiny synthetic Zarr dataset with `bt` and `sza` |

---

### Task 1: Label parameters, trust mask and histogram pixels

**Files:**
- Create: `src/fogseg/labels.py`, `tests/test_labels.py`
- Modify: `configs/default.yaml`, `pyproject.toml`

**Interfaces:**
- Produces:
  - constants `NO_FOG = 0`, `FOG = 1`, `IGNORE = 255`
  - `@dataclass(frozen=True) class LabelParams` with fields `night_min_sza: float`, `btd_fixed: float`, `warm_min_bt108: float`, `cirrus_max_split: float`, `smooth_max_std: float`, `uncertainty_band: float`, `min_blob_pixels: int`, and `@classmethod from_config(cls, labels_cfg: dict) -> LabelParams`
  - `trusted_mask(ir039, ir108, ir120, sza, night_min_sza: float) -> np.ndarray` (bool, same shape)
  - `histogram_pixels(ir039, ir108, ir120, sza, params: LabelParams) -> np.ndarray` (1-D float array of BTD values of trusted, warm, non-cirrus pixels)

- [ ] **Step 1: Add the `labels:` section to `configs/default.yaml`** with the provisional values from Global Constraints, plus `mode: fixed`, `output_dir: "/mnt/d/fogseg-data/processed"`, `figures_dir: "/mnt/d/fogseg-data/figures/inspect"`, and a `dynamic:` block with `bin_width: 0.25`, `bin_range: [-2.0, 8.0]`, `smoothing_bins: 3`, `clear_window: [0.5, 3.0]`, `cloud_window: [3.0, 8.0]`, `min_pixels: 2000`, `min_hump_share: 0.05`, `max_valley_ratio: 0.5`, `clamp: [1.5, 5.0]`. One comment above the block: values are provisional until `docs/notes/btd_exploration.md`.

- [ ] **Step 2: Add `scipy` and `matplotlib` to `dependencies` in `pyproject.toml`**, then `pip install -e ".[dev]"`.

- [ ] **Step 3: Write the failing tests** in `tests/test_labels.py`

```python
def test_params_from_config_reads_every_threshold():
    params = LabelParams.from_config(load_config("configs/default.yaml")["labels"])
    assert params.night_min_sza == 95.0
    assert params.min_blob_pixels == 5


def test_daytime_and_twilight_pixels_are_not_trusted():
    sza = np.array([[80.0, 95.0, 95.1]])
    bt = np.full((1, 3), 270.0)
    assert trusted_mask(bt, bt, bt, sza, 95.0).tolist() == [[False, False, True]]


def test_missing_channel_value_is_not_trusted():
    ir039 = np.array([[np.nan, 270.0]])
    bt = np.full((1, 2), 270.0)
    sza = np.full((1, 2), 120.0)
    assert trusted_mask(ir039, bt, bt, sza, 95.0).tolist() == [[False, True]]


def test_histogram_pixels_keep_only_warm_non_cirrus_night_pixels():
    # columns: valid fog-like, cold top, thin cirrus, daytime
    ir108 = np.array([[270.0, 230.0, 270.0, 270.0]])
    ir039 = ir108 - 3.0
    ir120 = np.array([[269.5, 229.5, 266.0, 269.5]])
    sza = np.array([[120.0, 120.0, 120.0, 80.0]])
    btd = histogram_pixels(ir039, ir108, ir120, sza, PROVISIONAL)
    assert btd.tolist() == [3.0]
```

`PROVISIONAL` is a module-level `LabelParams` in the test file with the Global Constraints values.

- [ ] **Step 4: Run** `pytest tests/test_labels.py -v` → FAIL (`ModuleNotFoundError: fogseg.labels`).

- [ ] **Step 5: Implement** `LabelParams`, the constants, `trusted_mask` and `histogram_pixels` in `src/fogseg/labels.py`. "Thin cirrus" is `ir108 - ir120 >= cirrus_max_split`; "warm" is `ir108 >= warm_min_bt108`.

- [ ] **Step 6: Run** `pytest -q && ruff check . && ruff format --check .` → all pass.

- [ ] **Step 7: Commit** `git add configs/default.yaml pyproject.toml src/fogseg/labels.py tests/test_labels.py && git commit -m "Add label parameters, trust mask and histogram pixel selection"`

---

### Task 2: Step 0, explore the BTD distributions and measure the thresholds

**Files:**
- Create: `src/fogseg/explore.py`, `tests/test_explore.py`, `notebooks/02_explore_btd.ipynb`, `docs/notes/btd_exploration.md`, `docs/notes/figures/` (PNGs)
- Modify: `configs/default.yaml` (measured values)

**Interfaces:**
- Consumes: `LabelParams`, `histogram_pixels` (Task 1)
- Produces: `satellite_zenith_angle(lons: np.ndarray, lats: np.ndarray, satellite_lon: float = 0.0) -> np.ndarray` (degrees); measured values in `configs/default.yaml`

- [ ] **Step 1: Write the failing test** in `tests/test_explore.py`

```python
def test_satellite_zenith_is_zero_below_the_satellite():
    assert satellite_zenith_angle(np.array([0.0]), np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-6)


def test_satellite_zenith_at_50_north_on_the_meridian():
    # geostationary at 0°E, Earth radius 6378.169 km, orbit radius 42164.0 km
    assert satellite_zenith_angle(np.array([0.0]), np.array([50.0]))[0] == pytest.approx(57.3, abs=0.2)
```

- [ ] **Step 2: Run** `pytest tests/test_explore.py -v` → FAIL.

- [ ] **Step 3: Implement `satellite_zenith_angle`** in `src/fogseg/explore.py` (spherical Earth; central angle `γ` from `cos γ = cos(lat)·cos(lon − satellite_lon)`, zenith `= atan2(sin γ, cos γ − R_earth / R_orbit)`).

- [ ] **Step 4: Run** `pytest -q` → pass. Commit `git commit -m "Add satellite viewing zenith angle for the BTD exploration"`.

- [ ] **Step 5: Build `notebooks/02_explore_btd.ipynb`** on **2023-24 scenes only**, with these sections:
  1. Pixel BTD histograms (`histogram_pixels`) of ~12 varied scenes, each next to its Night Microphysics RGB (EUMETSAT recipe ranges: R `IR_120−IR_108` −4…+2 K, G `IR_108−IR_039` 0…+10 K, B `IR_108` 243…293 K).
  2. For each scene, mark which hump is clear ground and which is low cloud, judged from the RGB.
  3. Median BTD of the clear-ground hump against satellite viewing zenith angle (10 bins over the region), month, and slot (21/00/03/05 UTC).
  4. IR_108 of low-cloud pixels by month (for `warm_min_bt108`); `IR_108 − IR_120` of low cloud vs thin cirrus (for `cirrus_max_split`).

- [ ] **Step 6: Human checkpoint.** Show the figures to the project owner; agree on: `btd_fixed`, `clear_window`, `cloud_window`, `clamp`, `warm_min_bt108`, `cirrus_max_split`, and whether viewing-angle bands are needed (spec Section 5: only if the clear-ground BTD clearly drifts with viewing angle).

- [ ] **Step 7: Write `docs/notes/btd_exploration.md`** (findings, figures saved to `docs/notes/figures/`, the agreed values and why) and update `configs/default.yaml` with the agreed values; replace the "provisional" comment with a link to the note.

- [ ] **Step 8: Commit** `git commit -m "Explore BTD distributions and set measured label thresholds"`

---

### Task 3: Fog tests, cleanup and the scene label

**Files:**
- Modify: `src/fogseg/labels.py`, `tests/test_labels.py`

**Interfaces:**
- Consumes: Task 1
- Produces:
  - `local_std(field: np.ndarray, size: int = 3) -> np.ndarray` (NaN-aware standard deviation in a `size × size` window)
  - `fog_candidates(ir039, ir108, ir120, btd_threshold: float, params: LabelParams) -> np.ndarray` (bool: all four tests)
  - `small_blobs(mask: np.ndarray, min_pixels: int) -> np.ndarray` (bool: pixels of 8-connected groups smaller than `min_pixels`)
  - `label_scene(ir039, ir108, ir120, sza, btd_threshold: float, params: LabelParams) -> np.ndarray` (`uint8`)

- [ ] **Step 1: Write the failing tests**

```python
def test_warm_high_btd_pixel_is_fog():
    assert scene(btd=4.0)[2, 2] == FOG


def test_cold_top_is_not_fog():
    assert scene(btd=4.0, ir108=230.0)[2, 2] == NO_FOG


def test_thin_cirrus_is_not_fog():
    assert scene(btd=4.0, split=3.0)[2, 2] == NO_FOG


def test_borderline_btd_is_ignored():
    assert scene(btd=2.5 + 0.3, btd_threshold=2.5)[2, 2] == IGNORE


def test_rough_top_is_not_fog():
    labels = scene(btd=4.0, checkerboard_amplitude=3.0)
    assert labels[2, 2] == NO_FOG


def test_local_std_ignores_missing_neighbours():
    field = np.full((3, 3), 270.0)
    field[0, 0] = np.nan
    assert local_std(field)[1, 1] == pytest.approx(0.0)


def test_three_pixel_island_is_ignored_but_ten_pixel_deck_stays_fog():
    mask = np.zeros((8, 12), bool)
    mask[1, 1:4] = True  # 3 pixels
    mask[5:7, 2:7] = True  # 10 pixels
    assert small_blobs(mask, 5)[1, 2] and not small_blobs(mask, 5)[5, 3]


def test_diagonal_pixels_form_one_blob():
    mask = np.eye(6, dtype=bool)  # 6 pixels touching only at corners
    assert not small_blobs(mask, 5).any()


def test_deck_touching_the_image_edge_is_kept():
    mask = np.zeros((8, 12), bool)
    mask[0:2, 7:12] = True  # 10 pixels along the top edge
    assert not small_blobs(mask, 5).any()


def test_scene_without_night_pixels_is_all_ignored():
    assert (scene(btd=4.0, sza=80.0) == IGNORE).all()
```

`scene(...)` is a test helper building 5×5 arrays from the named values (defaults: `ir108` 270 K, `split` 0.5 K, `sza` 120°, `btd_threshold` 2.5 K) and calling `label_scene` with `PROVISIONAL`; `checkerboard_amplitude` adds ±amplitude to IR_108 in a checkerboard pattern; IR_039 is derived as `ir108 − btd` *after* that, so only the smoothness test changes.

- [ ] **Step 2: Run** `pytest tests/test_labels.py -v` → new tests FAIL.

- [ ] **Step 3: Implement** in `src/fogseg/labels.py`:
  - `local_std`: `scipy.ndimage.uniform_filter` on values (NaN → 0) and on a validity mask, mean and mean of squares divided by the valid count; windows with no valid value → NaN.
  - `fog_candidates`: BTD `> btd_threshold`, warm, `split < cirrus_max_split`, `local_std(ir108) < smooth_max_std`.
  - `small_blobs`: `scipy.ndimage.label` with a 3×3 structure of ones, then `np.bincount` of the labels.
  - `label_scene`: start with `IGNORE`; trusted pixels → `NO_FOG`; trusted fog candidates → `FOG`; then candidates with `|BTD − btd_threshold| < uncertainty_band` → `IGNORE`; then `FOG` pixels in `small_blobs(..., min_blob_pixels)` → `IGNORE`.

- [ ] **Step 4: Run** `pytest -q && ruff check . && ruff format --check .` → pass.

- [ ] **Step 5: Commit** `git commit -m "Add fog tests, cleanup and per-scene labelling"`

---

### Task 4: Dynamic BTD threshold

> **Amended 2026-10-19 (after Task 2, Step 0).** The valley method below found a valley in 0 of 304
> training-season scenes. With the project owner's agreement it is replaced by **clear-ground peak +
> 1.5 K** (spec Section 5). A follow-up Task 4b reworks `dynamic_threshold.py`: `DynamicParams` loses
> `cloud_window`, `min_hump_share` and `max_valley_ratio` and gains `offset`; values `clear_window`
> [1.5, 4.75], `clamp` [3.5, 6.0]. The task text below is kept as the record of the first design.

**Files:**
- Create: `src/fogseg/dynamic_threshold.py`, `tests/test_dynamic_threshold.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure function on BTD samples)
- Produces:
  - `@dataclass(frozen=True) class DynamicParams` with `bin_width`, `bin_range: tuple[float, float]`, `smoothing_bins: int`, `clear_window`, `cloud_window`, `min_pixels: int`, `min_hump_share: float`, `max_valley_ratio: float`, `clamp: tuple[float, float]`, and `from_config(cls, dynamic_cfg: dict)`
  - `@dataclass(frozen=True) class ThresholdChoice` with `value: float`, `source: str` (`"fixed"`, `"dynamic"` or `"fallback"`)
  - `dynamic_threshold(btd: np.ndarray, fixed: float, params: DynamicParams) -> ThresholdChoice`

- [ ] **Step 1: Write the failing tests**

```python
def test_two_humps_give_the_valley():
    rng = np.random.default_rng(0)
    btd = np.concatenate([rng.normal(1.8, 0.4, 6000), rng.normal(5.0, 0.6, 6000)])
    choice = dynamic_threshold(btd, fixed=2.5, params=PARAMS)
    assert choice.source == "dynamic" and 2.8 < choice.value < 4.0


def test_one_hump_falls_back_to_fixed():
    btd = np.random.default_rng(1).normal(2.0, 0.5, 10000)
    assert dynamic_threshold(btd, 2.5, PARAMS) == ThresholdChoice(2.5, "fallback")


def test_too_few_pixels_fall_back():
    assert dynamic_threshold(np.array([1.0, 5.0]), 2.5, PARAMS).source == "fallback"


def test_tiny_second_hump_falls_back():
    rng = np.random.default_rng(2)
    btd = np.concatenate([rng.normal(1.8, 0.4, 9900), rng.normal(5.0, 0.3, 100)])
    assert dynamic_threshold(btd, 2.5, PARAMS).source == "fallback"


def test_valley_outside_the_plausible_range_is_clamped():
    rng = np.random.default_rng(3)
    btd = np.concatenate([rng.normal(1.8, 0.4, 6000), rng.normal(5.0, 0.6, 6000)])
    narrow = dataclasses.replace(PARAMS, clamp=(1.5, 3.0))
    assert dynamic_threshold(btd, 2.5, narrow) == ThresholdChoice(3.0, "dynamic")


def test_flat_valley_bottom_picks_its_middle():
    rng = np.random.default_rng(4)
    btd = np.concatenate([rng.uniform(1.0, 2.0, 6000), rng.uniform(5.0, 6.0, 6000)])
    assert dynamic_threshold(btd, 2.5, PARAMS).value == pytest.approx(3.5, abs=0.25)
```

`PARAMS` uses the Task 1 provisional dynamic values (explicit in the test file, not read from the config, so tuning never breaks the tests).

- [ ] **Step 2: Run** `pytest tests/test_dynamic_threshold.py -v` → FAIL.

- [ ] **Step 3: Implement `dynamic_threshold`** with named steps: `np.histogram` over `bin_range` with `bin_width`; moving average of `smoothing_bins`; clear peak = highest bin with centre in `clear_window`, cloud peak = highest bin with centre in `cloud_window`; hump share = raw counts inside each window ÷ total; valley = lowest smoothed bin strictly between the two peaks; if several bins share that minimum (a flat bottom), take the middle one. Fallback (`ThresholdChoice(fixed, "fallback")`) when: fewer than `min_pixels` samples, either share below `min_hump_share`, peaks not separated by at least one bin, or valley ÷ smaller peak above `max_valley_ratio`. Otherwise the valley bin centre clamped to `clamp`, source `"dynamic"`.

- [ ] **Step 4: Run** `pytest -q && ruff check . && ruff format --check .` → pass.

- [ ] **Step 5: Commit** `git commit -m "Add dynamic per-scene BTD threshold with fallback"`

---

### Task 5: Label store and `fogseg label`

**Files:**
- Create: `src/fogseg/label_store.py`, `tests/test_label_store.py`
- Modify: `src/fogseg/cli.py`, `tests/helpers.py`

**Interfaces:**
- Consumes: `LabelParams`, `label_scene`, `histogram_pixels` (Tasks 1, 3); `DynamicParams`, `dynamic_threshold` (Task 4)
- Produces:
  - `label_store_path(output_dir: Path, mode: str) -> Path` → `output_dir / f"labels_{mode}.zarr"`
  - `label_dataset(dataset_path: Path, out_path: Path, mode: str, params: LabelParams, dynamic: DynamicParams | None = None, overwrite: bool = False, batch: int = 32) -> int` (returns number of scenes)
  - store variables exactly as spec Section 3.4: `label` (`uint8`, chunks `(8, ny, nx)`), `btd_threshold` (`float32`), `threshold_source` (str), `fog_fraction`, `ignore_fraction` (`float32`); attrs `mode`, every `LabelParams` field, the dynamic fields when `mode == "dynamic"`, `fogseg_version`, `source_dataset`, `source_created`
  - CLI: `fogseg label [--mode fixed|dynamic] [--overwrite]` (mode default from config)
  - test helper `make_label_dataset(path: Path, n_scenes: int = 3) -> Path`: a small Zarr store with `bt` (channels IR_039, IR_108, IR_120) and `sza`, one fog patch per scene

- [ ] **Step 1: Write the failing tests**

```python
def test_store_has_spec_variables_and_aligns_with_the_dataset(tmp_path):
    data = make_label_dataset(tmp_path / "data.zarr")
    out = tmp_path / "labels_fixed.zarr"
    assert label_dataset(data, out, "fixed", PROVISIONAL) == 3
    labels = xr.open_zarr(out, consolidated=False)
    assert labels["label"].dtype == np.uint8
    assert set(np.unique(labels["label"])) <= {0, 1, 255}
    assert list(labels["threshold_source"].values) == ["fixed"] * 3
    merged = xr.merge([xr.open_zarr(data, consolidated=False), labels])
    assert merged.sizes["time"] == 3
    assert labels.attrs["btd_fixed"] == PROVISIONAL.btd_fixed


def test_dynamic_mode_records_threshold_source(tmp_path):
    ...
    # assert every source is "dynamic" or "fallback"


def test_scene_without_night_gets_nan_fog_fraction(tmp_path):
    ...
    # one scene with sza 80° everywhere: label all 255, fog_fraction NaN, no RuntimeWarning (pytest.warns not raised)


def test_missing_dataset_names_the_expected_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="nowhere.zarr"):
        label_dataset(tmp_path / "nowhere.zarr", tmp_path / "labels.zarr", "fixed", PROVISIONAL)


def test_refuses_to_overwrite_and_keeps_old_store_until_success(tmp_path):
    ...
    # second call without overwrite → FileExistsError; with overwrite=True and a dataset
    # that fails mid-way (missing sza in the 2nd batch, batch=1) → error, old store unchanged, no .part left
```

- [ ] **Step 2: Run** `pytest tests/test_label_store.py -v` → FAIL.

- [ ] **Step 3: Implement `label_dataset`** following `build.py`'s structure (temporary `.part`, `try/except BaseException` cleanup, overwrite check first): per batch, per scene: in dynamic mode `dynamic_threshold(histogram_pixels(...), params.btd_fixed, dynamic)`, in fixed mode `ThresholdChoice(params.btd_fixed, "fixed")`; `label_scene(...)`; fractions computed as counts ÷ counts with `np.nan` when the denominator is 0. Write `time`, `y`, `x` coordinates from the dataset; metadata written once at the end.

- [ ] **Step 4: Add `cmd_label`** and the `label` sub-parser in `cli.py`: dataset path from `data.processed_path`, output from `label_store_path(labels.output_dir, mode)`; log scenes, the dynamic/fallback counts and the overall fog fraction.

- [ ] **Step 5: Run** `pytest -q && ruff check . && ruff format --check .` → pass.

- [ ] **Step 6: Commit** `git commit -m "Add label store and fogseg label command"`

---

### Task 6: Check figures and `fogseg inspect`

**Files:**
- Create: `src/fogseg/inspect.py`, `tests/test_inspect.py`
- Modify: `src/fogseg/cli.py`

**Interfaces:**
- Consumes: label store from Task 5; `DynamicParams` (bins for the histogram); `histogram_pixels`
- Produces:
  - `night_microphysics_rgb(ir039, ir108, ir120) -> np.ndarray` (shape `(y, x, 3)`, values 0–1; ranges as in Task 2 Step 5)
  - `find_scene(times: pd.DatetimeIndex, wanted: pd.Timestamp) -> int` (index; raises `ValueError` listing the 3 nearest times when absent)
  - `inspect_figure(dataset: xr.Dataset, labels: xr.Dataset, wanted: pd.Timestamp, params: LabelParams, dynamic: DynamicParams, out_dir: Path) -> Path` (PNG with the four panels of spec Section 6.1; file name `inspect_<mode>_<YYYYmmddTHHMM>.png`)
  - CLI: `fogseg inspect --time <ISO> [--labels fixed|dynamic]`

- [ ] **Step 1: Write the failing tests**

```python
def test_rgb_maps_recipe_ranges_to_zero_and_one():
    rgb = night_microphysics_rgb(np.array([[293.0]]) - 10, np.array([[293.0]]), np.array([[295.0]]))
    assert rgb[0, 0].tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_missing_time_lists_nearest_available_times():
    times = pd.DatetimeIndex(["2023-11-20 21:00", "2023-11-21 00:00", "2023-11-21 03:00"])
    with pytest.raises(ValueError, match="2023-11-21 00:00"):
        find_scene(times, pd.Timestamp("2023-11-22 00:00"))


def test_inspect_writes_a_png(tmp_path):
    ...
    # make_label_dataset + label_dataset, then inspect_figure → file exists, size > 0
```

- [ ] **Step 2: Run** `pytest tests/test_inspect.py -v` → FAIL.

- [ ] **Step 3: Implement** `inspect.py` with matplotlib (`Agg` backend set inside the module), panels: RGB; RGB with `FOG` in cyan and `IGNORE` hatched grey; BTD histogram with the threshold line and its source in the title; `sza` with the 95° contour. `cmd_inspect` in `cli.py`.

- [ ] **Step 4: Run** `pytest -q && ruff check . && ruff format --check .` → pass.

- [ ] **Step 5: Commit** `git commit -m "Add fogseg inspect check figures"`

---

### Task 7: Reproducible check-scene selection

**Files:**
- Create: `src/fogseg/check_scenes.py`, `tests/test_check_scenes.py`
- Modify: `src/fogseg/cli.py`

**Interfaces:**
- Consumes: dataset and a label store (Task 5)
- Produces:
  - `scene_statistics(dataset: xr.Dataset, labels: xr.Dataset) -> pd.DataFrame` with columns `time`, `season` (`"2023-24"`/`"2024-25"`), `fog_fraction`, `high_cloud_fraction` (night pixels with `IR_108 < 240 K`), `twilight_fraction` (`sza ≤ 95`), `threshold_source`
  - `select_check_scenes(stats: pd.DataFrame) -> pd.DataFrame`: training season only; 3 highest `fog_fraction`, 3 lowest, 3 highest `high_cloud_fraction`, 2 highest `twilight_fraction`, 2 `fallback` scenes, then fill to 15 with the scenes spread evenly over October–February; no duplicates; deterministic (ties broken by time)
  - CLI: `fogseg select-scenes --labels fixed|dynamic` → writes `configs/check_scenes.yaml` (list of ISO times, each with its group)

- [ ] **Step 1: Write the failing tests**

```python
def test_selection_uses_only_the_training_season():
    ...
    # stats for 40 scenes over both seasons → every selected time in 2023-24


def test_selection_is_deterministic_and_has_no_duplicates():
    ...
    # two calls → identical frames; time column unique; len == 15


def test_selection_contains_each_group():
    ...
    # groups "foggy", "clear", "high_cloud", "twilight", "fallback", "spread" present
```

- [ ] **Step 2: Run** `pytest tests/test_check_scenes.py -v` → FAIL.

- [ ] **Step 3: Implement** `scene_statistics`, `select_check_scenes` and `cmd_select_scenes`.

- [ ] **Step 4: Run** `pytest -q && ruff check . && ruff format --check .` → pass.

- [ ] **Step 5: Commit** `git commit -m "Add reproducible check-scene selection"`

---

### Task 8: Run, check, tune and compare (real data)

**Files:**
- Modify: `configs/default.yaml` (tuned values), `configs/check_scenes.yaml` (generated)
- Create: `docs/notes/label_check.md`, `docs/notes/figures/` (selected PNGs)

**Interfaces:**
- Consumes: everything above; no new code unless the check reveals a bug (then a test first)

- [ ] **Step 1:** `fogseg label --mode fixed` → log shows 608 scenes; `fogseg select-scenes --labels fixed` → `configs/check_scenes.yaml` with 15 training-season scenes.
- [ ] **Step 2:** `fogseg inspect` for each check scene (fixed labels).
- [ ] **Step 3: Human checkpoint.** Review the figures with the project owner; adjust thresholds in `configs/default.yaml` (training-season evidence only); rerun Steps 1–2 with `--overwrite` until agreed.
- [ ] **Step 4:** `fogseg label --mode dynamic`; inspect the same check scenes with `--labels dynamic`; count dynamic vs fallback scenes.
- [ ] **Step 5: Hero night.** Search DWD / news for a documented fog event on one of the dataset's nights (prefer 2023-24; 10 Nov 2024 only as showcase, never for tuning); verify in the RGB; inspect it in both modes.
- [ ] **Step 6:** Write `docs/notes/label_check.md`: per check scene the fixed vs dynamic figures, what changed, the tuning steps, the hero night, and a recommendation for the default mode.
- [ ] **Step 7: Commit** `git commit -m "Check and tune fog labels on training-season scenes"`

---

### Task 9: Dataset card and README

**Files:**
- Modify: `docs/DATASET_CARD.md`, `README.md`

- [ ] **Step 1:** Dataset card: new "Labels" section with the label values, the rule (Mermaid flowchart from the spec), per mode the fog and ignore fractions and the dynamic/fallback counts, link to `docs/notes/btd_exploration.md` and `docs/notes/label_check.md`, and the limitations (high cloud over fog, fog vs low stratus, snow).
- [ ] **Step 2:** README: `fogseg label` / `fogseg inspect` / `fogseg select-scenes` in the quickstart and the "What exists" table; the spec's data-flow diagram (Mermaid); the hero-night figure with "Contains modified EUMETSAT data".
- [ ] **Step 3:** `pytest -q && ruff check . && ruff format --check .` → pass.
- [ ] **Step 4: Commit** `git commit -m "Document the fog labels"`; push the branch; open the PR into `develop`.

---

## Out of scope (separate plans)

- `docs/explainers/` with the conversation visuals (spec Section 10, separate step).
- Tuning against training-season METAR (spec Section 7, after the METAR step).
- Viewing-angle bands for the dynamic threshold, unless Task 2 shows they are needed (then a new task is added before Task 4).
