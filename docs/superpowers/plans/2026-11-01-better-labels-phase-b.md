# Better labels, phase B: rule repair and time — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** New options for the label rule (texture test, 3.9 µm emissivity index, KMA split window, temporal
hysteresis), a `fogseg experiment --phase B` command that cross-validates ten variants at matched FAR on the selection
cases, and one held-out comparison of the winner against the fixed 4.0 K rule with the keep rule of the spec.

**Architecture:** `labels.py` gains the options as `LabelParams` fields whose defaults reproduce today's labels bit for
bit, plus `label_night` for the night-level hysteresis; `label_store.py` labels whole nights. A new `experiment.py`
holds the matched-FAR choice, the blocked cross-validation and the variant runs; it reads the case table that
`fogseg validate` writes and relabels only the scenes of the case nights.

**Tech Stack:** Python 3.11+, numpy, pandas, xarray, zarr 3, scipy (`ndimage`). No new dependency.

**Spec:** `docs/superpowers/specs/2026-10-31-better-labels-design.md` (Sections 2, 3.3, 5, 7, 8, 9, 10). Phase A
results: `docs/notes/better_labels.md`.

**Decisions this plan takes where the spec is silent** (each is in the task that owns it):

- The paired bootstrap resamples night blocks, as spec Section 3.3 says (phase A's `paired_difference` is given block
  counts instead of night counts; no new function).
- The emissivity threshold is turned into a per-pixel BTD threshold, so the uncertainty band and the hysteresis stay in
  K for both indices.
- Every texture option uses the same limit, `smooth_max_std` 1.5 K; only the fog threshold is chosen per variant.
- The scored case set is fixed by the baseline case table; a variant that calls one of those cases obscured gets "no".
- The pooled cross-validation score sums the four test folds; a case whose night and station sit in different folds is
  tested twice, the same for every variant.
- Variant 9 may keep the baseline texture if none of variants 2–5 beats it.
- The held-out comparison is against the rule in the config (fixed 4.0 K); "FAR is not higher" compares the two
  held-out point estimates.
- `stations.temporal_relax_grid: [0.5, 1.0, 1.5]` K is the small grid of variant 10.

**Branch:** `feature/better-labels-rule`, cut from `feature/better-labels-evaluation` (PR #6); its PR goes into
`feature/better-labels-evaluation`.

## Global Constraints

- Owner's code style: "we do not comment dumb stuff. Linting is always there the code should look absolutely readable. Even if you can make things much smaller if you see that the readability shrinks just select a simpler approach." Match the docstring style of the module you edit.
- `ruff check .` and `ruff format --check .` clean (ruff pinned `0.17.*`, line length 110).
- Tests use synthetic data only; no network call and no file under `/mnt/d` in any test.
- The 2024-25 season is never used for a decision: `experiment` refuses any season but `stations.tune_season`.
- Nothing is chosen on held-out cases (night fold 5 or station fold 5). `experiment` without `--held-out` reads only selection cases; `--held-out` runs once per phase and refuses a second run.
- Default `LabelParams` reproduce the current labels bit for bit: `texture: ir108_std3`, `fog_index: btd`, `emissivity_max: null`, `split_window: fixed`, `temporal_relax: 0.0`, `temporal_max_dt108: 2.0`.
- Label values: `0` no fog, `1` fog, `2` obscured, `255` ignore. Full count: label `1` is "yes"; `0`, `2`, `255` are "no"; truth `-1` is skipped.
- Scored cases: deduplicated (`validation.deduplicate`), `hidden` false in the baseline case table, truth defined. A variant that marks one of these cases obscured gets "no" for it; the case set never changes between variants.
- Matched FAR: `stations.matched_far: 0.28`; the chosen grid value has the highest full-count FAR not above 0.28.
- Bootstrap unit: night blocks of `stations.night_block_days: 14` (`splits.night_block_start`), `stations.bootstrap_samples` draws, `stations.seed`.
- Keep rule: the winner replaces the fixed 4.0 K rule only if the 95 % interval of its held-out full-count POD gain lies above 0 and its held-out full-count FAR is not higher than the baseline's.
- Run Python in WSL: `source ~/code/seviri-fog-segmentation/.venv/bin/activate; export PYTHONPATH=<worktree>/src`. Run git with `GIT_DIR=~/code/seviri-fog-segmentation/.git/worktrees/better-labels-4df4fc GIT_WORK_TREE=.`; never `git worktree prune`.
- Commit messages: imperative sentence like the existing history, ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. A night with a missing slot (only 21:00 and 03:00 stored): the hysteresis neighbours are the stored scenes before and after, not the clock slots — test in Task 3.
2. A grid value at which a variant labels no case fog on a training part (FAR undefined) must be skipped by the matched-FAR choice, not crash it — test in Task 5.
3. No grid value reaches FAR ≤ 0.28 on a fold: take the value with the lowest FAR and log a warning naming the variant and fold — test in Task 5.
4. A label store written before this phase (no `texture`, `fog_index`, … attributes) must still be read by `validate`, `inspect` and `diagnose`, with the defaults — test in Task 4.
5. A pixel with a NaN channel near a texture window or in the emissivity index must stay `255` without NumPy warnings in the test output — test in Task 2.

---

### Task 1: Split-window and fog-index options

**Files:**
- Modify: `src/fogseg/labels.py`, `configs/default.yaml` (`labels:` and `stations:`)
- Test: `tests/test_labels.py`

**Interfaces:**
- Produces:
  - Constants `TEXTURES = ("ir108_std3", "btd_std3", "ir108_mad5", "blob", "none")`, `FOG_INDICES = ("btd", "emissivity")`, `SPLIT_WINDOWS = ("fixed", "kma")`, `KMA_INTERCEPT = -37.48`, `KMA_SLOPE = 0.1329`, `KMA_HALFWIDTH_K = 1.0`, `WAVELENGTH_39_UM = 3.92`.
  - `LabelParams` gains, after the existing fields and with these defaults: `texture: str = "ir108_std3"`, `fog_index: str = "btd"`, `emissivity_max: float | None = None`, `split_window: str = "fixed"`, `temporal_relax: float = 0.0`, `temporal_max_dt108: float = 2.0`. `__post_init__` raises `ValueError` naming the allowed values for an unknown `texture`, `fog_index` or `split_window`. `from_config` takes a field's default when its key is missing (old configs and old store attributes).
  - `warm_and_not_cirrus(ir108, ir120, params)`: with `split_window: kma` a pixel is not cirrus when `|(ir108 - ir120) - (KMA_INTERCEPT + KMA_SLOPE * ir108)| <= KMA_HALFWIDTH_K`.
  - `planck_39(bt: np.ndarray) -> np.ndarray`, `emissivity_39(ir039, ir108) -> np.ndarray` (= `planck_39(ir039) / planck_39(ir108)`), `emissivity_btd_threshold(ir108, emissivity_max: float) -> np.ndarray` (the BTD at which `emissivity_39` equals `emissivity_max`, per pixel: `ir108 - T` with `planck_39(T) = emissivity_max * planck_39(ir108)`).
  - `btd_threshold_image(ir108, threshold: float, params) -> float | np.ndarray`: `threshold` itself for `fog_index: btd`, `emissivity_btd_threshold(ir108, threshold)` for `emissivity`.
  - `label_scene(ir039, ir108, ir120, sza, threshold, params)`: the fifth parameter is renamed `threshold`, in the units of `params.fog_index` (K, or emissivity); the BTD margin is `btd - btd_threshold_image(...)`, so the uncertainty band and the hysteresis stay in K for both indices.
- Config keys (defaults keep today's labels): `labels.texture: ir108_std3`, `labels.fog_index: btd`, `labels.emissivity_max: null`, `labels.split_window: fixed`, `labels.temporal_relax: 0.0`, `labels.temporal_max_dt108: 2.0`, `stations.temporal_relax_grid: [0.5, 1.0, 1.5]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_defaults_reproduce_the_labels():
    ir039, ir108, ir120, sza = scene_arrays()  # a small synthetic scene with every label value
    old_style = LabelParams(**{name: getattr(PARAMS, name) for name in BASE_FIELDS})
    assert np.array_equal(label_scene(ir039, ir108, ir120, sza, 4.0, old_style), EXPECTED_LABELS)


def test_from_config_without_the_new_keys_takes_the_defaults():
    params = LabelParams.from_config({name: getattr(PARAMS, name) for name in BASE_FIELDS})
    assert (params.texture, params.fog_index, params.emissivity_max) == ("ir108_std3", "btd", None)
    assert (params.split_window, params.temporal_relax, params.temporal_max_dt108) == ("fixed", 0.0, 2.0)


def test_unknown_option_names_the_allowed_values():
    with pytest.raises(ValueError, match="ir108_mad5"):
        dataclasses.replace(PARAMS, texture="sobel")


def test_kma_split_window_follows_the_temperature():
    ir108 = np.array([280.0, 280.0])
    ir120 = ir108 - np.array([0.9, -0.2])  # expected split at 280 K is -0.268 K
    fixed = warm_and_not_cirrus(ir108, ir120, PARAMS)
    kma = warm_and_not_cirrus(ir108, ir120, dataclasses.replace(PARAMS, split_window="kma"))
    assert fixed.tolist() == [True, True] and kma.tolist() == [False, True]


def test_emissivity_threshold_is_the_btd_with_that_emissivity():
    ir108 = np.array([270.0, 285.0])
    emissivity = emissivity_39(ir108 - 4.0, ir108)
    assert emissivity_btd_threshold(ir108, float(emissivity[0]))[0] == pytest.approx(4.0, abs=1e-6)
    assert emissivity[1] > emissivity[0]  # the same BTD is a weaker signal on a warmer top


def test_emissivity_index_labels_like_the_btd_on_a_uniform_top():
    ir039, ir108, ir120, sza = uniform_top_scene(ir108=280.0)
    threshold = float(emissivity_39(np.array([276.0]), np.array([280.0]))[0])
    params = dataclasses.replace(PARAMS, fog_index="emissivity")
    assert np.array_equal(
        label_scene(ir039, ir108, ir120, sza, threshold, params),
        label_scene(ir039, ir108, ir120, sza, 4.0, PARAMS),
    )
```

`BASE_FIELDS` are the seven current field names. `EXPECTED_LABELS` is a literal array: run the current
`label_scene` on `scene_arrays()` before changing `labels.py` and write its output into the test, so the test pins
today's labels. `uniform_top_scene` is a small local helper with one IR10.8 temperature everywhere and BTDs on both
sides of 4.0 K.

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_labels.py -v`
Expected: FAIL (`emissivity_39` not defined, `LabelParams` has no `texture`)

- [ ] **Step 3: Implement the fields, the two options and the config keys**

`planck_39` may drop the constant factor of Planck's law (it cancels in every ratio):
`1 / (np.exp(14387.77 / (WAVELENGTH_39_UM * bt)) - 1)`; its inverse gives `T` in `emissivity_btd_threshold`.

- [ ] **Step 4: Run the tests and ruff**

Run: `pytest -q && ruff check . && ruff format --check .`
Expected: all pass, ruff clean

- [ ] **Step 5: Commit**

```bash
git add src/fogseg/labels.py configs/default.yaml tests/test_labels.py
git commit -m "Add the KMA split window and the 3.9 µm emissivity index as label options"
```

---

### Task 2: Texture options

**Files:**
- Modify: `src/fogseg/labels.py`
- Test: `tests/test_labels.py`

**Interfaces:**
- Consumes: Task 1 `LabelParams.texture`, `btd_threshold_image`.
- Produces:
  - `local_mad(field: np.ndarray, size: int = 5) -> np.ndarray`: `ndimage.median_filter(|field - ndimage.median_filter(field, size)|, size)` (mode `nearest`), NaN wherever the window touches a NaN.
  - `texture_ok(ir108, btd, region, params) -> np.ndarray` (bool), all against `params.smooth_max_std`:
    - `ir108_std3`: `local_std(ir108) < smooth_max_std`
    - `btd_std3`: `local_std(btd) < smooth_max_std`
    - `ir108_mad5`: `local_mad(ir108) < smooth_max_std`
    - `blob`: the median of `local_std(ir108)` over each 8-connected group of `region` is below `smooth_max_std`; pixels outside `region` fail
    - `none`: every pixel passes
  - `label_scene` uses `texture_ok` with `region = trusted & warm_and_not_cirrus & (margin > -uncertainty_band)`; `passes_other_tests` is removed (its only callers are in `labels.py`).

- [ ] **Step 1: Write the failing tests**

```python
def test_mad5_ignores_one_hot_pixel_that_fails_std3():
    ir108 = np.full((7, 7), 270.0)
    ir108[3, 3] = 276.0
    btd = np.full((7, 7), 5.0)
    region = np.ones((7, 7), dtype=bool)
    std3 = texture_ok(ir108, btd, region, PARAMS)
    mad5 = texture_ok(ir108, btd, region, dataclasses.replace(PARAMS, texture="ir108_mad5"))
    assert not std3[3, 3] and mad5.all()


def test_btd_std3_looks_at_the_btd_not_ir108():
    ir108 = rough_field()  # std of IR10.8 above 1.5 K everywhere
    btd = np.full(ir108.shape, 5.0)
    params = dataclasses.replace(PARAMS, texture="btd_std3")
    assert texture_ok(ir108, btd, np.ones(ir108.shape, dtype=bool), params).all()


def test_blob_judges_each_group_by_its_median_spread():
    ir108, region = two_blobs()  # left blob smooth with one rough edge pixel, right blob rough
    passed = texture_ok(ir108, np.zeros_like(ir108), region, dataclasses.replace(PARAMS, texture="blob"))
    assert passed[LEFT_BLOB].all() and not passed[RIGHT_BLOB].any() and not passed[~region].any()


def test_no_texture_test_passes_everything():
    params = dataclasses.replace(PARAMS, texture="none")
    assert texture_ok(rough_field(), np.zeros((6, 6)), np.zeros((6, 6), dtype=bool), params).all()


def test_nan_channel_stays_ignored_under_every_texture(recwarn):
    ir039, ir108, ir120, sza = scene_arrays()
    ir108[2, 2] = np.nan
    for texture in TEXTURES:
        labels = label_scene(ir039, ir108, ir120, sza, 4.0, dataclasses.replace(PARAMS, texture=texture))
        assert labels[2, 2] == IGNORE
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_labels.py -v`
Expected: FAIL (`texture_ok` not defined)

- [ ] **Step 3: Implement `local_mad`, `texture_ok` and use it in `label_scene`**

`blob`: `ndimage.label(region, structure=np.ones((3, 3)))`, then `ndimage.median(spread, labels, index)` per group
and index the result back by group id.

- [ ] **Step 4: Run the tests and ruff**

Run: `pytest -q && ruff check . && ruff format --check .`
Expected: all pass (including Task 1's `test_defaults_reproduce_the_labels`), ruff clean

- [ ] **Step 5: Commit**

```bash
git add src/fogseg/labels.py tests/test_labels.py
git commit -m "Add the BTD, median-deviation, per-blob and no-texture options to the smoothness test"
```

---

### Task 3: Temporal hysteresis over a night

**Files:**
- Modify: `src/fogseg/labels.py`
- Test: `tests/test_labels.py`

**Interfaces:**
- Consumes: Task 1 `btd_threshold_image`, Task 2 `texture_ok`.
- Produces: `label_night(ir039, ir108, ir120, sza, thresholds: np.ndarray, params) -> np.ndarray` — inputs of shape
  `(n_scenes, y, x)` for the stored scenes of one night in time order, `thresholds` of shape `(n_scenes,)`. Each scene
  is labelled with `label_scene`. If `params.temporal_relax > 0`, a pixel of scene *s* becomes `FOG` once, without
  iterating, when all of these hold:
  - its label is `NO_FOG` or `IGNORE` and it is trusted (an untrusted or obscured pixel never changes);
  - it is warm, not cirrus, and passes `texture_ok` with `region = trusted & warm_and_not_cirrus & (margin > -max(uncertainty_band, temporal_relax))`;
  - its margin `btd - btd_threshold_image(...)` is above `-temporal_relax`;
  - in scene *s − 1* or *s + 1* (the stored neighbours) the same pixel has the strict label `FOG` and
    `|ir108[s] - ir108[neighbour]| < temporal_max_dt108`.

- [ ] **Step 1: Write the failing tests**

```python
def test_hysteresis_turns_a_near_miss_next_to_fog_into_fog():
    night = night_arrays(btd=[5.0, 3.7, 3.7], ir108=[270.0, 270.5, 270.5])  # one pixel of interest per scene
    labels = label_night(*night, np.full(3, 4.0), dataclasses.replace(PARAMS, temporal_relax=0.5))
    assert labels[:, 4, 4].tolist() == [FOG, FOG, NO_FOG]  # scene 2 only sees the strict label of scene 1


def test_hysteresis_needs_a_steady_top():
    night = night_arrays(btd=[5.0, 3.7], ir108=[270.0, 273.0])
    labels = label_night(*night, np.full(2, 4.0), dataclasses.replace(PARAMS, temporal_relax=0.5))
    assert labels[1, 4, 4] == NO_FOG


def test_no_relax_gives_the_scene_labels():
    night = night_arrays(btd=[5.0, 3.7, 4.2], ir108=[270.0, 270.0, 270.0])
    labels = label_night(*night, np.full(3, 4.0), PARAMS)
    for s in range(3):
        assert np.array_equal(labels[s], label_scene(*(a[s] for a in night), 4.0, PARAMS))


def test_obscured_pixel_next_to_fog_stays_obscured():
    night = night_arrays(btd=[5.0, 3.9], ir108=[270.0, 250.0])
    labels = label_night(*night, np.full(2, 4.0), dataclasses.replace(PARAMS, temporal_relax=0.5))
    assert labels[1, 4, 4] == OBSCURED


def test_neighbours_are_the_stored_scenes():
    night = night_arrays(btd=[5.0, 3.7], ir108=[270.0, 270.0])  # e.g. 21:00 and 03:00, 00:00 missing
    labels = label_night(*night, np.full(2, 4.0), dataclasses.replace(PARAMS, temporal_relax=0.5))
    assert labels[1, 4, 4] == FOG
```

`night_arrays` is a local helper: a smooth 9 × 9 warm field per scene whose central pixel block (large enough to pass
the small-blob test) has the given BTD and IR10.8; it returns `(ir039, ir108, ir120, sza)` stacked over scenes.

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_labels.py -v`
Expected: FAIL (`label_night` not defined)

- [ ] **Step 3: Implement `label_night`**

- [ ] **Step 4: Run the tests and ruff**

Run: `pytest -q && ruff check . && ruff format --check .`
Expected: all pass, ruff clean

- [ ] **Step 5: Commit**

```bash
git add src/fogseg/labels.py tests/test_labels.py
git commit -m "Relax the fog threshold next to fog in an adjacent slot of the same night"
```

---

### Task 4: The label store honours the options

**Files:**
- Modify: `src/fogseg/label_store.py`, `src/fogseg/cli.py` (`cmd_label`)
- Test: `tests/test_label_store.py`

**Interfaces:**
- Consumes: Task 1 `LabelParams` fields, Task 3 `label_night`.
- Produces:
  - `check_mode(mode, dynamic, params)`: `ValueError` for `dynamic` with `fog_index: emissivity` (the dynamic peak is a BTD), and for `fog_index: emissivity` with `emissivity_max` None (message names `labels.emissivity_max`). `cmd_label` passes `params`.
  - `choose_threshold`: fixed mode returns `ThresholdChoice(params.emissivity_max, "fixed")` for the emissivity index.
  - `night_batches(times: np.ndarray, batch: int) -> list[tuple[int, int]]`: contiguous `[start, stop)` spans of whole nights (a night = scenes with the same date of `time - 12 h`), each at most `batch` scenes unless one night alone is longer.
  - `write_batches` writes `night_batches` spans; `label_batch` labels each night of its span with `label_night`.
  - `btd_threshold` keeps its name; its `long_name` becomes `"threshold of the fog index: K for btd, emissivity for emissivity"`.
  - Store attributes gain the six new fields through `dataclasses.asdict` (`emissivity_max` None is stored as null); `stored_params` reads old stores with the defaults (Task 1's `from_config`).

- [ ] **Step 1: Write the failing tests**

```python
def test_night_batches_never_split_a_night():
    times = np.array(scene_times(["2023-10-01T21", "2023-10-02T00", "2023-10-02T03", "2023-10-03T21"]))
    assert night_batches(times, 2) == [(0, 3), (3, 4)]


def test_store_with_temporal_relax_matches_label_night(tmp_path):
    dataset = write_dataset(tmp_path, n_nights=2, slots=3)
    params = dataclasses.replace(PARAMS, temporal_relax=0.5)
    label_dataset(dataset, tmp_path / "labels.zarr", "fixed", params, batch=2)
    stored = xr.open_zarr(tmp_path / "labels.zarr", consolidated=False)["label"].values
    assert np.array_equal(stored[:3], label_night(*night_inputs(dataset, 0), np.full(3, 4.0), params))


def test_emissivity_store_records_its_threshold(tmp_path):
    params = dataclasses.replace(PARAMS, fog_index="emissivity", emissivity_max=0.9)
    label_dataset(write_dataset(tmp_path), tmp_path / "labels.zarr", "fixed", params)
    store = xr.open_zarr(tmp_path / "labels.zarr", consolidated=False)
    assert np.allclose(store["btd_threshold"].values, 0.9) and store.attrs["fog_index"] == "emissivity"


def test_dynamic_mode_refuses_the_emissivity_index():
    params = dataclasses.replace(PARAMS, fog_index="emissivity", emissivity_max=0.9)
    with pytest.raises(ValueError, match="dynamic"):
        check_mode("dynamic", DYNAMIC, params)


def test_old_store_attributes_give_the_default_options():
    attrs = {"mode": "fixed", **{name: getattr(PARAMS, name) for name in BASE_FIELDS}}
    params, _ = stored_params(attrs, DYNAMIC)
    assert params == LabelParams(**{name: getattr(PARAMS, name) for name in BASE_FIELDS})
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_label_store.py -v`
Expected: FAIL (`night_batches` not defined, `check_mode` takes two arguments)

- [ ] **Step 3: Implement**

`label_store.py` must not import `validation.py` (it is imported by it); compute the night in `night_batches` itself.

- [ ] **Step 4: Run the tests and ruff**

Run: `pytest -q && ruff check . && ruff format --check .`
Expected: all pass, ruff clean

- [ ] **Step 5: Commit**

```bash
git add src/fogseg/label_store.py src/fogseg/cli.py tests/test_label_store.py
git commit -m "Label whole nights and record the label options in the store"
```

---

### Task 5: Matched-FAR choice and blocked cross-validation

**Files:**
- Create: `src/fogseg/experiment.py`, `tests/test_experiment.py`
- Modify: `src/fogseg/validation.py`, `src/fogseg/diagnosis.py`
- Test: `tests/test_diagnosis.py` (only the moved reader)

**Interfaces:**
- Consumes: `scores.contingency(..., full_count=True)`, `scores.skill`, `splits.cv_folds`, `splits.night_block_start`.
- Produces:
  - `validation.read_case_table(cases_path: Path) -> pd.DataFrame`: the table `fogseg validate` wrote, deduplicated, with the dtypes `diagnosis.read_selection_cases` uses today (`FileNotFoundError` naming `fogseg validate`); `diagnosis.read_selection_cases` calls it.
  - `experiment.scored_cases(cases) -> pd.DataFrame`: `hidden` false and `truth_fls` in (0, 1), index reset, with a `case` column `0..n-1`.
  - `experiment.GRID_COLUMNS = ["value", "case", "label"]` — a grid is the label of every case at every grid value.
  - `experiment.value_scores(grid, cases, mask: pd.Series) -> pd.DataFrame`: full-count `hits`, `misses`, `false_alarms`, `correct_negatives`, `pod`, `far`, `csi`, `n_cases` per value, over the cases where `mask` is true.
  - `experiment.matched_value(scores: pd.DataFrame, far_target: float) -> tuple[float, bool]`: the value with the highest `far` not above `far_target` (ties: higher `pod`, then lower value), and `True`; rows with undefined `far` are skipped; if no row qualifies, the value with the lowest `far` and `False`.
  - `experiment.cross_validate(grid, cases, n_folds: int, far_target: float) -> pd.DataFrame`: one row per fold *k* of `splits.cv_folds` (value matched on the training part, scores of the test part at that value) and a row `fold = "pooled"` with the summed test counts of the folds and their scores. Columns: `fold`, `value`, `matched`, then the `value_scores` columns.
  - `experiment.block_counts(grid, cases, value: float, mask, season_start, block_days) -> np.ndarray`: full-count counts per night block, shape `(n_blocks, 4)`, rows in block order — the input of `scores.paired_difference`.

- [ ] **Step 1: Write the failing tests**

```python
def test_matched_value_is_the_highest_far_not_above_the_target():
    scores = pd.DataFrame({"value": [3.0, 4.0, 5.0], "far": [0.35, 0.27, 0.20], "pod": [0.7, 0.6, 0.5]})
    assert matched_value(scores, 0.28) == (4.0, True)


def test_matched_value_skips_values_without_any_fog():
    scores = pd.DataFrame({"value": [4.0, 9.0], "far": [0.25, np.nan], "pod": [0.5, 0.0]})
    assert matched_value(scores, 0.28) == (4.0, True)


def test_matched_value_falls_back_to_the_lowest_far():
    scores = pd.DataFrame({"value": [3.0, 4.0], "far": [0.5, 0.4], "pod": [0.8, 0.7]})
    assert matched_value(scores, 0.28) == (4.0, False)


def test_each_fold_chooses_on_its_training_part():
    cases, grid = two_value_grid()  # value 3.0 reaches FAR <= 0.28 only on fold 1's training part
    table = cross_validate(grid, cases, n_folds=5, far_target=0.28).set_index("fold")
    assert table.loc[1, "value"] == 3.0 and table.loc[2, "value"] == 4.0
    folds = table.drop(index="pooled")
    assert table.loc["pooled", "hits"] == folds["hits"].sum()


def test_held_out_cases_never_enter_the_cross_validation():
    cases, grid = two_value_grid()
    cases.loc[0, "night_fold"] = 5
    grid.loc[grid["case"] == 0, "label"] = 1  # would change every fold if it were used
    assert cross_validate(grid, cases, 5, 0.28).equals(cross_validate(*two_value_grid(), 5, 0.28))
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_experiment.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement, and move the case reader into `validation.py`**

When `matched_value` falls back, `cross_validate` logs a warning naming the fold; the caller adds the variant name.

- [ ] **Step 4: Run the tests and ruff**

Run: `pytest -q && ruff check . && ruff format --check .`
Expected: all pass, ruff clean

- [ ] **Step 5: Commit**

```bash
git add src/fogseg/experiment.py src/fogseg/validation.py src/fogseg/diagnosis.py tests/test_experiment.py tests/test_diagnosis.py
git commit -m "Choose thresholds at matched FAR inside a blocked cross-validation"
```

---

### Task 6: Labels of a variant at the cases

**Files:**
- Modify: `src/fogseg/experiment.py`
- Test: `tests/test_experiment.py`

**Interfaces:**
- Consumes: Task 3 `label_night`, Task 5 `GRID_COLUMNS`, `label_store.LABEL_CHANNELS`, `validation.night_of`.
- Produces:
  - `experiment.value_grid(params: LabelParams, stations_cfg: dict) -> np.ndarray`: `tuning.grid_values` of `stations_cfg["btd_fixed_grid"]` for `fog_index: btd`, of `stations_cfg["emissivity_grid"]` for `emissivity`.
  - `experiment.variant_grid(data: xr.Dataset, cases: pd.DataFrame, params: LabelParams, values: np.ndarray) -> pd.DataFrame` (`GRID_COLUMNS`): for every night that has a case, all stored scenes of that night are read once and labelled with `label_night` at each value (one threshold for every scene); the label at each case's `iy`, `ix` of its scene is kept. `ValueError` if a case's scene is not in `data`.

- [ ] **Step 1: Write the failing tests**

```python
def test_variant_grid_without_relax_is_the_scene_label(tmp_path):
    data, cases = tiny_season(tmp_path)  # 2 nights x 2 slots, 3 cases
    grid = variant_grid(data, cases, PARAMS, np.array([3.5, 4.5]))
    expected = label_scene(*scene_inputs(data, cases.loc[0, "scene_time"]), 4.5, PARAMS)
    row = grid[(grid["case"] == 0) & (grid["value"] == 4.5)]
    assert row["label"].item() == expected[cases.loc[0, "iy"], cases.loc[0, "ix"]]


def test_variant_grid_relabels_whole_nights(tmp_path):
    data, cases = tiny_season(tmp_path)
    params = dataclasses.replace(PARAMS, temporal_relax=0.5)
    grid = variant_grid(data, cases, params, np.array([4.0]))
    night = label_night(*night_inputs(data, cases.loc[1, "night"]), np.full(2, 4.0), params)
    assert (
        grid.loc[grid["case"] == 1, "label"].item()
        == night[SLOT_OF_CASE_1, cases.loc[1, "iy"], cases.loc[1, "ix"]]
    )


def test_case_of_a_missing_scene_is_an_error(tmp_path):
    data, cases = tiny_season(tmp_path)
    cases.loc[0, "scene_time"] = pd.Timestamp("2030-01-01T21:00")
    with pytest.raises(ValueError, match="not in the dataset"):
        variant_grid(data, cases, PARAMS, np.array([4.0]))
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_experiment.py -v`
Expected: FAIL (`variant_grid` not defined)

- [ ] **Step 3: Implement `value_grid` and `variant_grid`**

- [ ] **Step 4: Run the tests and ruff**

Run: `pytest -q && ruff check . && ruff format --check .`
Expected: all pass, ruff clean

- [ ] **Step 5: Commit**

```bash
git add src/fogseg/experiment.py tests/test_experiment.py
git commit -m "Relabel the case nights of a variant at every threshold of its grid"
```

---

### Task 7: `fogseg experiment --phase B`

**Files:**
- Modify: `src/fogseg/experiment.py`, `src/fogseg/cli.py`
- Test: `tests/test_experiment.py`

**Interfaces:**
- Consumes: Tasks 5 and 6; `splits.read_splits`, `splits.selection_cases`, `splits.splits_path`; `validation.read_case_table`, `validation.season_start`; `cli.require_tune_season`.
- Produces:
  - `experiment.PHASE_B_VARIANTS: list[tuple[str, dict]]`, in this order: `("1 baseline", {})`, `("2 texture btd_std3", {"texture": "btd_std3"})`, `("3 texture ir108_mad5", {"texture": "ir108_mad5"})`, `("4 texture blob", {"texture": "blob"})`, `("5 texture none", {"texture": "none"})`, `("6 band 0.25", {"uncertainty_band": 0.25})`, `("7 emissivity", {"fog_index": "emissivity"})`, `("8 split window kma", {"split_window": "kma"})`.
  - `experiment.combined_variants(table: pd.DataFrame, relax_grid: list[float]) -> list[tuple[str, dict]]`: variant 9 = the texture of the best of variants 1–5 (pooled `pod`) with the `fog_index` of the better of 1 and 7, named `"9 combined"`; then one variant per relax value, `9`'s options plus `{"temporal_relax": r}`, named `"10 temporal relax <r>"`.
  - `experiment.run_variants(data, cases, params, variants, stations_cfg) -> pd.DataFrame`: `cross_validate` rows of each variant with columns `variant` and `options` (the overrides as JSON) in front.
  - `experiment.winner(table: pd.DataFrame) -> str`: the variant with the highest pooled `pod`; ties go to the lower variant number.
  - `cli.cmd_experiment(args)`, the command `fogseg experiment --phase B [--season S] [--held-out]` (`--phase` choices `["B"]`, season default and only allowed value `stations.tune_season`): reads the split file and `cases_<labels.mode>_<season>.csv`, keeps the selection cases, runs variants 1–8, then the combined variants, writes `stations/experiment_B.csv` and logs the winner's pooled POD and FAR next to variant 1's. `--held-out` is Task 8.

- [ ] **Step 1: Write the failing tests**

```python
def test_combined_variants_take_the_best_texture_and_index():
    table = pooled_table(
        {
            "1 baseline": 0.40,
            "2 texture btd_std3": 0.42,
            "3 texture ir108_mad5": 0.45,
            "4 texture blob": 0.41,
            "5 texture none": 0.39,
            "7 emissivity": 0.44,
        }
    )
    combined = combined_variants(table, [0.5, 1.0])
    assert combined[0] == ("9 combined", {"texture": "ir108_mad5", "fog_index": "emissivity"})
    assert combined[2] == (
        "10 temporal relax 1.0",
        {"texture": "ir108_mad5", "fog_index": "emissivity", "temporal_relax": 1.0},
    )


def test_baseline_texture_can_win_variant_9():
    table = pooled_table(
        {
            "1 baseline": 0.46,
            "2 texture btd_std3": 0.42,
            "3 texture ir108_mad5": 0.45,
            "4 texture blob": 0.41,
            "5 texture none": 0.39,
            "7 emissivity": 0.44,
        }
    )
    assert combined_variants(table, [0.5])[0] == ("9 combined", {"texture": "ir108_std3", "fog_index": "btd"})


def test_experiment_writes_every_variant_and_fold(tmp_path, caplog):
    cfg = experiment_config(tmp_path)  # tiny dataset, case table and split file under tmp_path
    cmd_experiment(experiment_args(cfg))
    table = pd.read_csv(tmp_path / "stations" / "experiment_B.csv")
    assert table["variant"].nunique() == 8 + 1 + len(cfg["stations"]["temporal_relax_grid"])
    assert set(table["fold"].astype(str)) == {"1", "2", "3", "4", "pooled"}
    assert "winner" in caplog.text


def test_experiment_refuses_the_test_season(tmp_path):
    with pytest.raises(PermissionError):
        cmd_experiment(experiment_args(experiment_config(tmp_path), season="2024-25"))


def test_experiment_needs_the_split_file(tmp_path):
    cfg = experiment_config(tmp_path)
    (tmp_path / "stations" / "splits_2023-24.csv").unlink()
    with pytest.raises(FileNotFoundError, match="fogseg split"):
        cmd_experiment(experiment_args(cfg))
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_experiment.py -v`
Expected: FAIL (`combined_variants` not defined)

- [ ] **Step 3: Implement the variants, the run, the winner and the command**

Log one INFO line per variant (name, pooled POD and FAR, seconds taken): the real run labels ~300 scenes per grid value.

- [ ] **Step 4: Run the tests and ruff**

Run: `pytest -q && ruff check . && ruff format --check .`
Expected: all pass, ruff clean

- [ ] **Step 5: Commit**

```bash
git add src/fogseg/experiment.py src/fogseg/cli.py tests/test_experiment.py
git commit -m "Cross-validate the phase B label variants at matched FAR"
```

---

### Task 8: The held-out comparison, once

**Files:**
- Modify: `src/fogseg/experiment.py`, `src/fogseg/cli.py`
- Test: `tests/test_experiment.py`

**Interfaces:**
- Consumes: Tasks 5–7; `scores.paired_difference`, `splits.held_out_part`.
- Produces:
  - `experiment.KeepDecision` (frozen dataclass): `winner: str`, `value: float`, `pod_gain`, `pod_gain_low`, `pod_gain_high`, `far_winner`, `far_baseline`, `far_difference_low`, `far_difference_high` (floats); property `keep` = `pod_gain_low > 0 and far_winner <= far_baseline`.
  - `experiment.held_out_comparison(data, selection, held_out, params, winner_options: dict, stations_cfg, season_start) -> tuple[pd.DataFrame, KeepDecision]`: the winner's value is matched at `matched_far` on **all** selection cases; the baseline is the config's rule at `params.btd_fixed` (today 4.0 K). Both are labelled on the held-out cases only. Table rows: `rule` (`baseline`, `winner`) × `part` (`held_out`, `nights`, `stations`, `both`) with the `value_scores` columns. POD and FAR differences use `paired_difference` on `block_counts` of the held-out cases.
  - CLI `--held-out`: needs `stations/experiment_B.csv` (`FileNotFoundError` naming `fogseg experiment --phase B`); refuses with `FileExistsError` if `stations/experiment_B_held_out.csv` exists ("the held-out set is opened once per phase"); writes that file and logs the decision and, if `keep`, the config keys and the threshold key to set (`labels.btd_fixed` or `labels.emissivity_max`).

- [ ] **Step 1: Write the failing tests**

```python
def test_keep_needs_a_pod_gain_above_zero_and_no_higher_far():
    assert KeepDecision("9 combined", 4.0, 0.05, 0.01, 0.09, 0.24, 0.25, -0.03, 0.01).keep
    assert not KeepDecision("9 combined", 4.0, 0.05, -0.01, 0.11, 0.24, 0.25, -0.03, 0.01).keep
    assert not KeepDecision("9 combined", 4.0, 0.05, 0.01, 0.09, 0.26, 0.25, -0.01, 0.03).keep


def test_winner_threshold_is_chosen_on_selection_cases_only(tmp_path):
    data, selection, held_out = split_season(tmp_path)
    _, decision = held_out_comparison(
        data, selection, held_out, PARAMS, {"texture": "none"}, STATIONS_CFG, START
    )
    expected, _ = matched_value(value_scores(*selection_grid(data, selection, {"texture": "none"})), 0.28)
    assert decision.value == expected


def test_held_out_runs_once(tmp_path):
    cfg = experiment_config(tmp_path)
    cmd_experiment(experiment_args(cfg))
    cmd_experiment(experiment_args(cfg, held_out=True))
    with pytest.raises(FileExistsError, match="once"):
        cmd_experiment(experiment_args(cfg, held_out=True))


def test_held_out_needs_the_cross_validation_first(tmp_path):
    with pytest.raises(FileNotFoundError, match="experiment --phase B"):
        cmd_experiment(experiment_args(experiment_config(tmp_path), held_out=True))
```

`selection_grid` is a local helper returning `(grid, cases, mask)` of the winner's variant on the selection cases.

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_experiment.py -v`
Expected: FAIL (`KeepDecision` not defined)

- [ ] **Step 3: Implement**

- [ ] **Step 4: Run the tests and ruff**

Run: `pytest -q && ruff check . && ruff format --check .`
Expected: all pass, ruff clean

- [ ] **Step 5: Commit**

```bash
git add src/fogseg/experiment.py src/fogseg/cli.py tests/test_experiment.py
git commit -m "Score the phase B winner against the 4.0 K rule once on the held-out set"
```

---

### Task 9: Real-data run, config, notes and PR (controller)

**Files:**
- Modify: `configs/default.yaml` (only if the winner is kept), `docs/notes/better_labels.md`, `docs/DATASET_CARD.md`, `README.md`

- [ ] **Step 1: Cross-validation run** (background; ~300 scenes per grid value and variant)

Run: `fogseg experiment --phase B`
Expected: `stations/experiment_B.csv` with 8 + 1 + 3 variants × 5 rows; a winner logged.

- [ ] **Step 2: Held-out run, once**

Run: `fogseg experiment --phase B --held-out`
Expected: `stations/experiment_B_held_out.csv`; keep or not keep logged.

- [ ] **Step 3: If kept:** set the winner's options and threshold in `configs/default.yaml`, then run
`fogseg label --overwrite`, `fogseg validate --season 2023-24` and `fogseg diagnose`. If not kept, the config stays.

- [ ] **Step 4: Notes.** `better_labels.md` gains a phase B section: the variants, the cross-validation table
(pooled POD / FAR / CSI and the matched value per variant), the held-out table with the phase A baseline and the
winner, the keep decision with its intervals, and the second look at the held-out set. The card describes the label
options and, if kept, the new default; the README lists `fogseg experiment`.

- [ ] **Step 5: Verify, commit, push and open the PR**

Run: `pytest -q && ruff check . && ruff format --check .`
Then commit, push `feature/better-labels-rule` and open the PR with `~/.local/bin/gh` into
`feature/better-labels-evaluation`.
