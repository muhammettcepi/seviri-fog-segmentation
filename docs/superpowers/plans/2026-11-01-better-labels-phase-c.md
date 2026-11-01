# Better labels, phase C: station-trained teacher — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A per-pixel fog classifier trained on the station truth of the selection cases (the "teacher"), calibrated
and thresholded at FAR 0.28, with an optional local correction from nearby stations, scored once against the
configured rule on the held-out cases, and able to write dense labels (`fogseg label --mode teacher`).

**Architecture:** `features.py` builds one feature stack per night, shared by training at station pixels and by dense
prediction. `teacher.py` holds the model settings, the blocked out-of-fold cross-validation, isotonic calibration, the
matched-FAR threshold, the inverse-distance correction and the held-out comparison. `teacher_labels.py` writes the
dense label store. It reuses the phase B machinery (`experiment.value_scores`, `matched_value`, `block_counts`,
`KeepDecision`, the once-guard and the config record).

**Tech Stack:** Python 3.11+, numpy, pandas, xarray, zarr 3, scipy, **scikit-learn** (new; brings joblib).

**Spec:** `docs/superpowers/specs/2026-10-31-better-labels-design.md` (Sections 2, 3, 6, 7, 8, 9, 10, 11). Phase A and
B results: `docs/notes/better_labels.md`.

**Decisions this plan takes where the spec is silent** (each is in the task that owns it):

- The phase B winner was not kept, so the "phase B rule" of the spec is the configured rule: fixed 4.0 K. It is both
  the `rule` feature and the reference of the held-out comparison. The teacher refuses `labels.mode` other than
  `fixed`.
- Settings are numbered `"1 …"` to `"8 …"` so `experiment.winner` picks the best one (ties to the lower number). There
  are 2 logistic regressions and 6 gradient-boosting settings, within the spec's limit of 10.
- p* and the setting are both chosen on the pooled out-of-fold predictions, as spec 6.2 says. That is mildly
  optimistic, but equally so for every setting, and the held-out set stays clean.
- The pooled out-of-fold table counts a case once per test fold it is in, as phase B does. Its residual for the
  correction uses the mean of its out-of-fold probabilities.
- The correction shift is Δ = Σ wᵢ rᵢ / (1 + Σ wᵢ), with w = exp(−d / L) for d < 3 L. The 1 is the teacher's own
  weight, as if it were a station at distance 0, so a single far station cannot move p by its whole residual.
- The length scale is the one with the highest pooled POD among those whose pooled FAR is not above the plain
  teacher's. If none beats the plain teacher's POD, there is no correction.
- The correction is judged on the held-out "unseen stations only" part. A held-out night has no selection station in
  its scenes, so nothing can correct it.
- One held-out opening runs both comparisons: teacher against rule, and corrected teacher against teacher.
- A case whose scene is missing from ERA5 is not used for training and gets 255 (a miss in the full count). In dense
  labels its whole scene is 255, apart from obscured pixels, which stay 2.
- `teacher.feature_groups` lists the feature groups, so a run without the `era5` group is possible. Whether to run
  without it is the owner's call at run time (Task 8), not this plan's.
- The `rule` label enters as two indicators, `rule_fog` and `rule_ignore`. Obscured pixels are never taught or labelled
  by the teacher; they stay 2.
- Dense labels apply the correction only if it was kept, only in tuning-season scenes, and only from selection
  stations.

**Branch:** `feature/better-labels-teacher`, cut from `feature/better-labels-rule` (PR #7); its PR goes into
`feature/better-labels-rule`.

## Global Constraints

- Owner's code style: "we do not comment dumb stuff. Linting is always there the code should look absolutely readable. Even if you can make things much smaller if you see that the readability shrinks just select a simpler approach." Match the docstring style of the module you edit; every line ≤ 110 columns (ruff does not enforce E501 here, so check).
- `ruff check .` and `ruff format --check .` clean (ruff pinned `0.17.*`, line length 110).
- Tests use synthetic data only; no network call and no file under `/mnt/d` in any test.
- New dependency: `scikit-learn` in `pyproject.toml` `dependencies`; nothing else.
- The 2024-25 season is never used for a decision. `fogseg teacher` refuses any season but `stations.tune_season`. `fogseg label --mode teacher` may label 2024-25 scenes (prediction only), but no 2024-25 station ever feeds the correction.
- Nothing is chosen, fitted or calibrated on held-out cases (night fold 5 or station fold 5). `fogseg teacher --held-out` runs once and refuses a second run.
- Scored cases: `experiment.experiment_cases` (deduplicated, `hidden` false, truth defined). Full count: label `1` is "yes"; `0`, `2` and `255` are "no".
- Matched FAR: `stations.matched_far: 0.28`. p* is the value of `teacher.probability_grid` with the highest pooled out-of-fold full-count FAR not above 0.28 (`experiment.matched_value`).
- Teacher labels: `1` where p ≥ p* + `ignore_halfwidth`, `0` where p ≤ p* − `ignore_halfwidth`, `255` in between or where p is undefined. Untrusted pixels are `255` and obscured pixels `2`, as in the rule.
- Bootstrap unit: 14-day night blocks (`experiment.block_counts`), `stations.bootstrap_samples` draws, `stations.seed`. Model randomness also uses `stations.seed`.
- Keep rule: a candidate replaces its reference only if the 95 % interval of its held-out full-count POD gain lies above 0 and its held-out full-count FAR is not higher (`experiment.KeepDecision`).
- Config keys (new section): `teacher.model_path: "/mnt/d/fogseg-data/processed/teacher.joblib"`, `teacher.ignore_halfwidth: 0.1`, `teacher.idw_length_km: [25, 50, 100]`, `teacher.feature_groups: [bt, differences, texture, geometry, terrain, era5, time, rule]`, `teacher.probability_grid: [0.10, 0.90, 0.01]`.
- Run Python in WSL: `source ~/code/seviri-fog-segmentation/.venv/bin/activate; export PYTHONPATH=<worktree>/src`. Run git with `GIT_DIR=~/code/seviri-fog-segmentation/.git/worktrees/better-labels-4df4fc GIT_WORK_TREE=.`; never `git worktree prune`. Install scikit-learn into the venv once, with `pip install scikit-learn`.
- Commit messages: imperative sentence like the existing history, ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **A scene missing from ERA5:** its cases get NaN ERA5 features and `has_era5` false. They are not fitted, and they get label 255 when scored. Test in Tasks 2 and 3.
2. **The first stored scene of a night:** `ir108_change` is NaN, not 0, and `night_max_btd` is the maximum over the stored scenes only. Test in Task 1.
3. **No feeding station within 3 L:** the correction is 0 and p is unchanged. The target's own station and held-out stations never feed it. Test in Task 5.
4. **`fogseg label --mode teacher` with a bad model:** without `teacher.joblib`, or with a model whose feature list differs from `teacher.feature_groups`, it fails with a clear message before labelling anything. Test in Task 7.
5. **A NaN channel at a pixel:** that pixel's features are NaN without NumPy warnings. Its teacher label is 255 (or 2 if obscured). Test in Tasks 1 and 7.

---

### Task 1: The feature stack of a night

**Files:**
- Create: `src/fogseg/features.py`, `tests/test_features.py`
- Modify: `pyproject.toml` (`scikit-learn`), `configs/default.yaml` (`teacher:` section)

**Interfaces:**
- Consumes: `labels.local_std`, `labels.emissivity_39`, `labels.FOG`, `labels.IGNORE`, `parallax.unit_vector`, `parallax.SATELLITE_RADIUS_KM`, `parallax.EARTH_RADIUS_KM`.
- Produces:
  - `CHANNELS = ["IR_039", "WV_062", "WV_073", "IR_087", "IR_097", "IR_108", "IR_120", "IR_134"]`.
  - `FEATURE_GROUPS: dict[str, list[str]]`, in this order:
    - `"bt"`: `bt_<channel>` for each of `CHANNELS`
    - `"differences"`: `d108_039`, `d108_120`, `d087_108`, `d120_134`, `d108_062`, `emissivity_39`
    - `"texture"`: `std3_ir108`, `std3_btd`
    - `"geometry"`: `satellite_zenith`, `latitude`
    - `"terrain"`: `elevation_mean`, `elevation_std`
    - `"era5"`: `t2m_minus_ir108`, `skt_minus_ir108`, `t2m_minus_d2m`, `snow_depth`
    - `"time"`: `hours_after_21`, `night_max_btd`, `ir108_change`
    - `"rule"`: `rule_fog`, `rule_ignore`
  - `feature_names(groups: list[str]) -> list[str]`: the names of the groups in `FEATURE_GROUPS` order, whatever the order of `groups`. Raises `ValueError` naming the allowed groups for an unknown one.
  - `satellite_zenith(lon, lat) -> np.ndarray`: degrees, for a satellite over 0° E at `SATELLITE_RADIUS_KM`.
  - `hours_after_21(times: np.ndarray) -> np.ndarray`: hours since 21:00 UTC of the night (21:00 → 0, 00:00 → 3, 03:00 → 6, 05:00 → 8).
  - `StaticFields` (frozen dataclass): `latitude`, `satellite_zenith`, `elevation_mean`, `elevation_std` (2-D arrays; the elevation fields may be `None` when `terrain` is not used).
  - `night_feature_stack(bt, times, static, era5, rule_labels, groups) -> np.ndarray`:
    - inputs: `bt` of shape `(n, 8, y, x)` in `CHANNELS` order; `times` of shape `(n,)` for the stored scenes of one night in time order; `era5` a dict `t2m`, `skt`, `d2m`, `sd` of `(n, y, x)` arrays, or `None`; `rule_labels` of shape `(n, y, x)`.
    - returns float32 of shape `(n, len(feature_names(groups)), y, x)`.
    - `night_max_btd` is the NaN-ignoring maximum of IR10.8 − IR3.9 over the night's stored scenes.
    - `ir108_change` is IR10.8 minus the previous stored scene's IR10.8; NaN for the first scene.
    - `ValueError` naming `fogseg era5` if `"era5"` is in `groups` and `era5` is None.
- Config: the `teacher:` keys of the Global Constraints.

- [ ] **Step 1: Write the failing tests**

```python
def test_feature_names_follow_the_groups():
    assert len(feature_names(list(FEATURE_GROUPS))) == 29
    assert feature_names(["rule", "bt"]) == [f"bt_{c}" for c in CHANNELS] + ["rule_fog", "rule_ignore"]
    with pytest.raises(ValueError, match="era5"):
        feature_names(["radar"])


def test_satellite_zenith_by_hand():
    assert satellite_zenith(np.array([0.0]), np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-6)
    assert satellite_zenith(np.array([0.0]), np.array([50.0]))[0] == pytest.approx(57.3, abs=0.2)


def test_hours_after_21():
    times = np.array(["2023-11-20T21", "2023-11-21T00", "2023-11-21T03", "2023-11-21T05"], dtype="datetime64[ns]")
    assert hours_after_21(times).tolist() == [0, 3, 6, 8]


def test_stack_values_at_one_pixel():
    bt, times, static, era5, rule = night_inputs()  # 2 scenes, 4 x 4, known values at pixel (1, 2)
    stack = night_feature_stack(bt, times, static, era5, rule, list(FEATURE_GROUPS))
    at = dict(zip(feature_names(list(FEATURE_GROUPS)), stack[:, :, 1, 2].T))
    ir039, ir108 = bt[:, CHANNELS.index("IR_039"), 1, 2], bt[:, CHANNELS.index("IR_108"), 1, 2]
    assert np.allclose(at["d108_039"], ir108 - ir039)
    assert np.allclose(at["emissivity_39"], emissivity_39(ir039, ir108))
    assert np.allclose(at["night_max_btd"], (ir108 - ir039).max())
    assert np.isnan(at["ir108_change"][0]) and at["ir108_change"][1] == pytest.approx(ir108[1] - ir108[0])
    assert at["rule_fog"].tolist() == (rule[:, 1, 2] == FOG).tolist()


def test_era5_group_needs_era5():
    bt, times, static, _, rule = night_inputs()
    with pytest.raises(ValueError, match="fogseg era5"):
        night_feature_stack(bt, times, static, None, rule, ["bt", "era5"])


def test_nan_channel_gives_nan_features_without_warnings(recwarn):
    bt, times, static, era5, rule = night_inputs()
    bt[0, CHANNELS.index("IR_108"), 1, 2] = np.nan
    stack = night_feature_stack(bt, times, static, era5, rule, list(FEATURE_GROUPS))
    assert np.isnan(stack[0, feature_names(list(FEATURE_GROUPS)).index("d108_039"), 1, 2])
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]
```

`night_inputs` is a local helper with distinct values per channel and scene.

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_features.py -v`
Expected: FAIL (`fogseg.features` not found)

- [ ] **Step 3: Implement `features.py`, add `scikit-learn` and the `teacher:` config section**

`satellite_zenith`: the angle between the local vertical `unit_vector(lon, lat)` and the vector from the ground point
(`EARTH_RADIUS_KM` · unit vector) to the satellite at (`SATELLITE_RADIUS_KM`, 0, 0).

- [ ] **Step 4: Run the tests and ruff**

Run: `pytest -q && ruff check . && ruff format --check .`
Expected: all pass, ruff clean

- [ ] **Step 5: Commit**

```bash
git add src/fogseg/features.py tests/test_features.py pyproject.toml configs/default.yaml
git commit -m "Build the teacher's feature stack for the scenes of a night"
```

---

### Task 2: Features from the stores, and at the cases

**Files:**
- Modify: `src/fogseg/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes:
  - Task 1 `night_feature_stack`, `StaticFields`, `CHANNELS`, `feature_names`.
  - `dem.read_dem`, `labels.label_night`, `experiment.baseline_threshold`, `validation.night_of`.
- Produces:
  - `read_static(data: xr.Dataset, dem_path: Path, groups: list[str]) -> StaticFields`. Raises `FileNotFoundError` naming `fogseg dem` if `"terrain"` is in `groups` and the DEM store is missing.
  - `open_era5(era5_path: Path, groups: list[str]) -> xr.Dataset | None`. Returns `None` if `"era5"` is not in `groups`. Raises `FileNotFoundError` naming `fogseg era5` if it is in `groups` and the store is missing.
  - `night_features(data, scenes: np.ndarray, static, era5_store, params: LabelParams, groups) -> tuple[np.ndarray, np.ndarray]`:
    - `scenes` are the dataset indices of one night, in time order.
    - It reads `bt` and `sza` once and labels the night with `label_night` at `baseline_threshold(params)` for the `rule` group.
    - It returns the stack and `has_era5`, a bool per scene. A scene missing from the ERA5 store gets NaN ERA5 features and `False`; without the `era5` group, `has_era5` is all True.
  - `case_features(data, cases, static, era5_store, params, groups) -> pd.DataFrame`:
    - Columns: `feature_names(groups)` plus `has_era5`. The index is that of `cases`.
    - It reads every case night once.
    - Raises `ValueError` "not in the dataset" if a case's scene is missing.

- [ ] **Step 1: Write the failing tests**

```python
def test_case_features_match_the_night_stack(tmp_path):
    data, cases, static = tiny_teacher_season(tmp_path)  # 2 nights x 2 slots, 3 cases, no ERA5 group
    groups = ["bt", "differences", "time", "rule"]
    table = case_features(data, cases, static, None, TEST_PARAMS, groups)
    night = np.flatnonzero(night_of(data["time"].values) == cases.loc[1, "night"])
    stack, _ = night_features(data, night, static, None, TEST_PARAMS, groups)
    slot = list(data["time"].values[night]).index(cases.loc[1, "scene_time"])
    expected = stack[slot, :, cases.loc[1, "iy"], cases.loc[1, "ix"]]
    assert np.allclose(table.loc[1, feature_names(groups)].to_numpy(float), expected, equal_nan=True)


def test_scene_missing_from_era5_gives_nan_and_no_era5(tmp_path):
    data, cases, static = tiny_teacher_season(tmp_path)
    era5 = era5_store_without(data, cases.loc[0, "scene_time"])  # every scene but case 0's
    table = case_features(data, cases, static, era5, TEST_PARAMS, ["bt", "era5"])
    assert not table.loc[0, "has_era5"] and np.isnan(table.loc[0, "t2m_minus_ir108"])
    assert table.loc[1, "has_era5"] and np.isfinite(table.loc[1, "t2m_minus_ir108"])


def test_missing_stores_name_their_command(tmp_path):
    data, _, _ = tiny_teacher_season(tmp_path)
    with pytest.raises(FileNotFoundError, match="fogseg dem"):
        read_static(data, tmp_path / "no_dem.zarr", ["terrain"])
    with pytest.raises(FileNotFoundError, match="fogseg era5"):
        open_era5(tmp_path / "no_era5.zarr", ["era5"])
    assert open_era5(tmp_path / "no_era5.zarr", ["bt"]) is None


def test_case_of_a_missing_scene_is_an_error(tmp_path):
    data, cases, static = tiny_teacher_season(tmp_path)
    cases.loc[0, "scene_time"] = pd.Timestamp("2030-01-01T21:00")
    with pytest.raises(ValueError, match="not in the dataset"):
        case_features(data, cases, static, None, TEST_PARAMS, ["bt"])
```

`tiny_teacher_season` and `era5_store_without` are local helpers. Build them from `tests/test_experiment.py`'s `tiny_season`, adding the eight channels.

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_features.py -v`
Expected: FAIL (`case_features` not defined)

- [ ] **Step 3: Implement `read_static`, `open_era5`, `night_features`, `case_features`**

- [ ] **Step 4: Run the tests and ruff**

Run: `pytest -q && ruff check . && ruff format --check .`
Expected: all pass, ruff clean

- [ ] **Step 5: Commit**

```bash
git add src/fogseg/features.py tests/test_features.py
git commit -m "Read the teacher's features from the stores, once per case night"
```

---

### Task 3: Out-of-fold probabilities, calibration and p*

**Files:**
- Create: `src/fogseg/teacher.py`, `tests/test_teacher.py`

**Interfaces:**
- Consumes:
  - `splits.cv_folds`, `experiment.CV_COLUMNS`, `SCORE_COLUMNS`, `value_scores`, `matched_value`, `score_columns`, `tuning.grid_values`.
  - Task 2 `case_features` output: feature columns plus `has_era5`.
  - Cases from `experiment.scored_cases`, with a `case` column.
- Produces:
  - `TeacherParams` (frozen dataclass): `model_path: str`, `ignore_halfwidth: float`, `idw_length_km: list[float]`, `feature_groups: list[str]`, `probability_grid: list[float]`. `from_config(teacher_cfg)`.
  - `MODEL_SETTINGS: list[tuple[str, dict]]`, in this order:
    - `("1 logistic C=0.1", {"model": "logistic", "C": 0.1})`
    - `("2 logistic C=1", {"model": "logistic", "C": 1.0})`
    - six `("<n> hgb lr=<lr> leaves=<k>", {"model": "hgb", "learning_rate": lr, "max_leaf_nodes": k})`, for n = 3–8 and (lr, k) in (0.05, 15), (0.05, 31), (0.05, 63), (0.1, 15), (0.1, 31), (0.1, 63)
  - `make_model(options: dict, seed: int)`:
    - logistic: `Pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=…, max_iter=2000))`
    - hgb: `HistGradientBoostingClassifier(learning_rate=…, max_leaf_nodes=…, max_iter=300, random_state=seed)`
  - `teacher_labels(p: np.ndarray, p_star: float, halfwidth: float) -> np.ndarray`: uint8, per the Global Constraints.
  - `fold_probabilities(features, cases, options, n_folds, seed) -> pd.DataFrame`:
    - Columns `fold`, `case`, `raw`, with one row per test appearance of a case in each fold of `splits.cv_folds`.
    - Each fold fits on its training part, keeping only rows with `has_era5`, and predicts its test part.
    - Rows without `has_era5` get NaN `raw`.
  - `fit_calibrator(fold_probs, cases) -> IsotonicRegression`: `out_of_bounds="clip"`, fitted on the finite `raw` against `truth_fls`.
  - `pooled_value_scores(fold_probs, cases, values, halfwidth) -> pd.DataFrame`:
    - Input: `fold_probs` with a calibrated column `p`.
    - Output: `SCORE_COLUMNS` per value, summed over the folds' test parts, with the labels from `teacher_labels(p, value, halfwidth)`.
  - `cross_validate_settings(features, cases, teacher_params, stations_cfg) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]`:
    - First element, the table: for each setting, columns `variant` (the setting name) and `options` (JSON), then `CV_COLUMNS`.
      - Rows: folds 1–4 scored at the pooled p*, then a `"pooled"` row.
      - `value` is p*; `matched` comes from `matched_value` on the pooled scores.
    - Second element: each setting's `fold_probs` with `raw` and calibrated `p`.

- [ ] **Step 1: Write the failing tests**

```python
def test_teacher_labels_have_an_ignore_band():
    p = np.array([0.20, 0.35, 0.45, 0.60, np.nan])
    assert teacher_labels(p, 0.4, 0.1).tolist() == [NO_FOG, IGNORE, IGNORE, FOG, IGNORE]


def test_settings_are_numbered_and_at_most_ten():
    assert len(MODEL_SETTINGS) <= 10
    assert [int(name.split()[0]) for name, _ in MODEL_SETTINGS] == list(range(1, len(MODEL_SETTINGS) + 1))


def test_a_case_never_predicts_itself():
    features, cases = learnable_cases()  # one informative feature, folds 1-4, both truths present
    before = fold_probabilities(features, cases, {"model": "logistic", "C": 1.0}, 5, 0)
    cases.loc[0, "truth_fls"] = 1 - cases.loc[0, "truth_fls"]
    after = fold_probabilities(features, cases, {"model": "logistic", "C": 1.0}, 5, 0)
    assert np.allclose(before.loc[before["case"] == 0, "raw"], after.loc[after["case"] == 0, "raw"])


def test_held_out_cases_never_enter(monkeypatch):
    features, cases = learnable_cases(held_out=6)  # 6 cases in night fold 5
    base, _ = cross_validate_settings(features, cases, TEACHER_PARAMS, STATIONS_CFG)
    cases.loc[cases["night_fold"] == 5, "truth_fls"] = 1
    flipped, _ = cross_validate_settings(features, cases, TEACHER_PARAMS, STATIONS_CFG)
    assert base.equals(flipped)


def test_p_star_is_the_matched_far_value():
    fold_probs, cases = hand_probabilities()  # pooled FAR 0.30 at 0.40, 0.25 at 0.45, 0.20 at 0.50
    scores = pooled_value_scores(fold_probs, cases, np.array([0.40, 0.45, 0.50]), 0.0)
    assert matched_value(scores, 0.28) == (0.45, True)


def test_cases_without_era5_are_not_fitted_and_count_as_misses():
    features, cases = learnable_cases()
    features.loc[0, "has_era5"] = False
    probs = fold_probabilities(features, cases, {"model": "logistic", "C": 1.0}, 5, 0)
    assert probs.loc[probs["case"] == 0, "raw"].isna().all()
```

`learnable_cases` and `hand_probabilities` are local helpers. `TEACHER_PARAMS` uses `feature_groups` matching the helper's columns and a short grid.

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_teacher.py -v`
Expected: FAIL (`fogseg.teacher` not found)

- [ ] **Step 3: Implement**

- [ ] **Step 4: Run the tests and ruff**

Run: `pytest -q && ruff check . && ruff format --check .`
Expected: all pass, ruff clean

- [ ] **Step 5: Commit**

```bash
git add src/fogseg/teacher.py tests/test_teacher.py
git commit -m "Cross-validate the teacher settings out of fold and choose p* at matched FAR"
```

---

### Task 4: The final teacher and `fogseg teacher`

**Files:**
- Modify: `src/fogseg/teacher.py`, `src/fogseg/cli.py`
- Test: `tests/test_teacher.py`

**Interfaces:**
- Consumes:
  - Task 3.
  - `experiment.experiment_cases`, `winner`, `config_record`, `check_same_config`.
  - `cli.require_tune_season`, `cli.write_atomically`.
  - Task 2 `read_static`, `open_era5`, `case_features`.
- Produces:
  - `TeacherModel` (frozen dataclass): `setting: str`, `options: dict`, `model`, `calibrator`, `feature_names: list[str]`, `feature_groups: list[str]`, `p_star: float`, `ignore_halfwidth: float`, `correction_km: float | None` (None until Task 5).
  - `fit_teacher(features, cases, setting: tuple[str, dict], fold_probs, p_star, teacher_params, seed) -> TeacherModel`: fits on all selection rows with `has_era5`; the calibrator is `fit_calibrator(fold_probs, cases)`.
  - `predict_probability(teacher: TeacherModel, rows: np.ndarray) -> np.ndarray`: calibrated p for rows of shape `(n, len(feature_names))`.
  - `save_teacher(teacher, path)` uses joblib, through a temporary `.part` file and `os.replace`.
  - `load_teacher(path, groups) -> TeacherModel`:
    - `FileNotFoundError` naming `fogseg teacher` if the file is missing.
    - `ValueError` naming both lists if `teacher.feature_names != feature_names(groups)`.
  - `oof_table(fold_probs, cases) -> pd.DataFrame`: one row per selection case, with `scene_time`, `station`, `lat`, `lon`, `iy`, `ix`, `truth_fls` and `p_oof` (the mean calibrated out-of-fold p of the case).
  - CLI `fogseg teacher [--season S]`:
    - `require_tune_season`.
    - Refuses `labels.mode` other than `fixed` (`ValueError` naming `labels.mode`).
    - Refuses (`FileExistsError`) when `stations/teacher_held_out.csv` or `.json` exists.
    - Writes `stations/teacher_cv.csv`, `stations/teacher_oof.csv` (the winner's `oof_table`), `teacher.model_path`, and `stations/teacher_config.json`. The record is `experiment.config_record` plus `"teacher": dataclasses.asdict(teacher_params)`.
    - Logs one INFO line per setting (name, pooled POD, FAR, p*, seconds) and the winner.

- [ ] **Step 1: Write the failing tests**

```python
def test_saved_teacher_predicts_like_the_fitted_one(tmp_path):
    teacher, rows = fitted_teacher()
    save_teacher(teacher, tmp_path / "teacher.joblib")
    loaded = load_teacher(tmp_path / "teacher.joblib", teacher.feature_groups)
    assert np.allclose(predict_probability(loaded, rows), predict_probability(teacher, rows))


def test_loading_needs_the_same_features(tmp_path):
    teacher, _ = fitted_teacher()
    save_teacher(teacher, tmp_path / "teacher.joblib")
    with pytest.raises(ValueError, match="rule_fog"):
        load_teacher(tmp_path / "teacher.joblib", ["bt"])
    with pytest.raises(FileNotFoundError, match="fogseg teacher"):
        load_teacher(tmp_path / "missing.joblib", ["bt"])


def test_teacher_command_writes_its_outputs(tmp_path, caplog):
    cfg = teacher_config(tmp_path)  # tiny dataset, DEM, case table, split file; feature groups without era5
    cmd_teacher(teacher_args(cfg))
    table = pd.read_csv(tmp_path / "stations" / "teacher_cv.csv")
    assert table["variant"].nunique() == len(MODEL_SETTINGS)
    assert (tmp_path / "processed" / "teacher.joblib").exists()
    assert (tmp_path / "stations" / "teacher_config.json").exists()
    assert "winner" in caplog.text


def test_teacher_refuses_the_test_season_and_dynamic_labels(tmp_path):
    cfg = teacher_config(tmp_path)
    with pytest.raises(PermissionError):
        cmd_teacher(teacher_args(cfg, season="2024-25"))
    with pytest.raises(ValueError, match="labels.mode"):
        cmd_teacher(teacher_args(cfg, mode="dynamic"))
```

`fitted_teacher`, `teacher_config` and `teacher_args` are local helpers. `teacher_config` reuses `tests/test_experiment.py`'s `experiment_config` pattern and adds a tiny DEM store.

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_teacher.py -v`
Expected: FAIL (`fit_teacher` not defined)

- [ ] **Step 3: Implement**

- [ ] **Step 4: Run the tests and ruff**

Run: `pytest -q && ruff check . && ruff format --check .`
Expected: all pass, ruff clean

- [ ] **Step 5: Commit**

```bash
git add src/fogseg/teacher.py src/fogseg/cli.py tests/test_teacher.py
git commit -m "Fit and save the chosen teacher with fogseg teacher"
```

---

### Task 5: Local correction from nearby stations

**Files:**
- Modify: `src/fogseg/teacher.py`, `src/fogseg/cli.py` (`cmd_teacher`)
- Test: `tests/test_teacher.py`

**Interfaces:**
- Consumes:
  - Tasks 3–4 (`fold_probs` with `p`, `oof_table`, `teacher_labels`, `TeacherModel.correction_km`).
  - `validation.haversine_km(lon, lat, point_lon, point_lat)`.
- Produces:
  - `idw_shift(target_lon, target_lat, feed_lon, feed_lat, residuals, length_km) -> np.ndarray`: for each target, Δ = Σ wᵢ rᵢ / (1 + Σ wᵢ), with wᵢ = exp(−dᵢ / L) for dᵢ < 3 L and 0 beyond. It is 0 with no feeder in range.
  - `corrected_probability(targets: pd.DataFrame, p: np.ndarray, feeders: pd.DataFrame, length_km) -> np.ndarray`:
    - `targets` and `feeders` carry `scene_time`, `lon`, `lat`, `station`; `feeders` also carries `residual` = `truth_fls − p_oof`.
    - Only feeders of the same scene count, and a target's own station never feeds it.
    - The result is clipped to [0, 1].
  - `cross_validate_correction(fold_probs, cases, oof, p_star, halfwidth, lengths, n_folds) -> pd.DataFrame`:
    - Rows: for `length_km` in `"none"` and each L, folds 1–4 plus `"pooled"`, with `SCORE_COLUMNS[1:]`.
      `length_km` is a string column (`"none"`, `"25.0"`, …), so the CSV reads back the same.
    - In fold k, a test case's feeders are the same-scene cases of `oof` whose `station_fold` is neither k nor `n_folds`.
  - `chosen_length(table) -> float | None`: the L with the highest pooled POD among those whose pooled FAR is not above the `"none"` row's FAR, if that POD beats `"none"`; otherwise None.
  - `cmd_teacher` writes `stations/teacher_correction_cv.csv` and saves `correction_km` in the model.

- [ ] **Step 1: Write the failing tests**

```python
def test_idw_shift_by_hand():
    zero = np.array([0.0])
    assert idw_shift(zero, zero, zero, zero, np.array([0.4]), 50.0)[0] == pytest.approx(0.2)  # w = 1: 0.4 / 2
    far_lon = np.array([3.0])  # ~334 km at the equator, beyond 3 L = 150 km
    assert idw_shift(zero, zero, far_lon, zero, np.array([0.4]), 50.0)[0] == 0.0


def test_two_feeders_by_hand():
    zero = np.array([0.0])
    d = 50.0 / 111.195  # one L in degrees at the equator
    shift = idw_shift(zero, zero, np.array([0.0, d]), np.array([0.0, 0.0]), np.array([0.4, -0.2]), 50.0)
    w = np.array([1.0, np.exp(-1.0)])
    assert shift[0] == pytest.approx((w @ [0.4, -0.2]) / (1 + w.sum()), rel=1e-3)


def test_own_station_and_other_scenes_never_feed():
    targets, feeders = feeder_tables()  # target station A; feeders: A same scene, B other scene
    assert corrected_probability(targets, np.array([0.3]), feeders, 50.0)[0] == pytest.approx(0.3)


def test_fold_feeders_skip_the_test_fold_and_the_held_out_stations():
    fold_probs, cases, oof = correction_cases()  # feeders in station folds k and 5 carry residual 1
    table = cross_validate_correction(fold_probs, cases, oof, 0.5, 0.0, [50.0], 5)
    pooled = table[table["fold"].astype(str) == "pooled"].set_index("length_km")
    assert pooled.loc["50.0", "hits"] == pooled.loc["none", "hits"]


def test_no_length_is_chosen_without_a_gain():
    table = correction_table(pod={"none": 0.40, "25.0": 0.39, "50.0": 0.40}, far={"none": 0.27, "25.0": 0.26})
    assert chosen_length(table) is None
```

`feeder_tables`, `correction_cases`, `correction_table` are local helpers. In `test_fold_feeders_…`, every feeder that is allowed carries residual 0, so a correct implementation changes no label.

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_teacher.py -v`
Expected: FAIL (`idw_shift` not defined)

- [ ] **Step 3: Implement, and run the correction cross-validation in `cmd_teacher` after the winner is chosen**

- [ ] **Step 4: Run the tests and ruff**

Run: `pytest -q && ruff check . && ruff format --check .`
Expected: all pass, ruff clean

- [ ] **Step 5: Commit**

```bash
git add src/fogseg/teacher.py src/fogseg/cli.py tests/test_teacher.py
git commit -m "Correct the teacher with residuals of nearby stations and choose the length on the folds"
```

---

### Task 6: The held-out comparison, once

**Files:**
- Modify: `src/fogseg/teacher.py`, `src/fogseg/cli.py`
- Test: `tests/test_teacher.py`

**Interfaces:**
- Consumes:
  - Tasks 2–5.
  - `experiment.variant_grid`, `baseline_threshold`, `block_counts`, `value_scores`, `KeepDecision`, `GRID_COLUMNS`, `check_same_config`.
  - `scores.paired_difference`, `splits.held_out_part`.
- Produces:
  - `teacher_held_out(data, selection, held_out, teacher, oof, params, stations_cfg, season_start, static, era5_store) -> tuple[pd.DataFrame, KeepDecision, KeepDecision | None]`:
    - Raises `ValueError` if `held_out` holds a selection case or `selection` a held-out one.
    - `rule` labels the held-out cases with `variant_grid` at `baseline_threshold(params)`.
    - `teacher` labels them with `teacher_labels(predict_probability(...), p_star, halfwidth)`, and 255 without `has_era5`.
    - `corrected` exists only if `teacher.correction_km` is set: the teacher, corrected from the selection cases of `oof`.
    - Table rows: `rule` × `part` (`held_out`, `nights`, `stations`, `both`) with the `value_scores` columns.
    - First decision: teacher against rule, over all held-out cases.
    - Second decision: corrected against teacher, over the `stations` part only; None without a correction.
    - Intervals come from `paired_difference` on `block_counts`. Each labelling is a frame of
      `experiment.GRID_COLUMNS` (`value` is the threshold for `rule` and p* for the teacher), so `block_counts`
      and `value_scores` apply unchanged.
    - `KeepDecision.winner` holds the candidate (`"teacher"` or `"corrected"`) and its `*_baseline` fields the
      reference (the rule, or the plain teacher).
  - CLI `fogseg teacher --held-out`:
    - `require_tune_season`.
    - Refuses with `FileExistsError` ("once") if `stations/teacher_held_out.csv` or `.json` exists, before any labelling.
    - Needs `teacher.model_path` and `stations/teacher_config.json`; `FileNotFoundError` names `fogseg teacher`.
    - Refuses on a config mismatch (`check_same_config`).
    - Logs the decisions before writing. Writes both files atomically.
    - The JSON holds both `KeepDecision`s, `keep` flags, `p_star`, `setting`, `correction_km`, and `config_changes`: `{"labels.mode": "teacher"}` if the teacher is kept.

- [ ] **Step 1: Write the failing tests**

```python
def test_teacher_gain_is_teacher_minus_rule(tmp_path):
    table, decision, _ = held_out_run(tmp_path, teacher_hits_more=True)
    rows = table.set_index(["rule", "part"])
    gain = rows.loc[("teacher", "held_out"), "pod"] - rows.loc[("rule", "held_out"), "pod"]
    assert decision.pod_gain == pytest.approx(gain) and gain > 0


def test_correction_is_judged_on_unseen_stations_only(tmp_path):
    table, _, correction = held_out_run(tmp_path, correction_km=50.0)
    rows = table.set_index(["rule", "part"])
    assert correction is not None
    assert rows.loc[("corrected", "nights"), "hits"] == rows.loc[("teacher", "nights"), "hits"]


def test_held_out_runs_once(tmp_path):
    cfg = teacher_config(tmp_path)
    cmd_teacher(teacher_args(cfg))
    cmd_teacher(teacher_args(cfg, held_out=True))
    with pytest.raises(FileExistsError, match="once"):
        cmd_teacher(teacher_args(cfg, held_out=True))


def test_held_out_needs_the_teacher_first(tmp_path):
    with pytest.raises(FileNotFoundError, match="fogseg teacher"):
        cmd_teacher(teacher_args(teacher_config(tmp_path), held_out=True))


def test_held_out_refuses_a_changed_config(tmp_path):
    cfg = teacher_config(tmp_path)
    cmd_teacher(teacher_args(cfg))
    cfg["teacher"]["ignore_halfwidth"] = 0.2
    with pytest.raises(ValueError, match="ignore_halfwidth"):
        cmd_teacher(teacher_args(cfg, held_out=True))
```

`held_out_run` is a local helper that builds a split season with a fitted toy teacher.

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_teacher.py -v`
Expected: FAIL (`teacher_held_out` not defined)

- [ ] **Step 3: Implement**

- [ ] **Step 4: Run the tests and ruff**

Run: `pytest -q && ruff check . && ruff format --check .`
Expected: all pass, ruff clean

- [ ] **Step 5: Commit**

```bash
git add src/fogseg/teacher.py src/fogseg/cli.py tests/test_teacher.py
git commit -m "Score the teacher against the configured rule once on the held-out set"
```

---

### Task 7: Dense teacher labels

**Files:**
- Create: `src/fogseg/teacher_labels.py`, `tests/test_teacher_labels.py`
- Modify: `src/fogseg/label_store.py` (`MODES`, `write_batches` → `write_store`), `src/fogseg/cli.py` (`cmd_label`)

**Interfaces:**
- Consumes:
  - Tasks 2, 4, 5 (`night_features`, `load_teacher`, `predict_probability`, `teacher_labels`, `corrected_probability`).
  - `label_store.night_batches`, `label_fractions`, `write_metadata`, `store_attrs`, `TIME_ENCODING`.
- Produces:
  - `label_store.MODES = ("fixed", "dynamic", "teacher")`.
  - `label_store.write_store(data, store, make_batch: Callable[[int, int], xr.Dataset], encoding: dict, batch: int)`: writes the `night_batches` spans. `write_batches` calls it, and existing stores are written as before.
  - `teacher_labels.label_teacher_dataset(dataset_path, out_path, cfg, overwrite=False, batch=32) -> int`:
    - Loads the teacher with `load_teacher(model_path, feature_groups)` before opening any batch.
    - Variables: `label` (uint8, 0 / 1 / 2 / 255), `fog_probability` (uint8: round(p × 250), 255 where p is undefined), `threshold_source` (`"teacher"`), and `fog_fraction`, `ignore_fraction`, `obscured_fraction`.
    - Untrusted pixels are 255. Obscured pixels (rule label 2) are 2. A scene without ERA5 is 255 everywhere but its obscured pixels.
    - If `stations/teacher_held_out.json` says the correction was kept, tuning-season scenes get `corrected_probability` from `stations/teacher_oof.csv`. Every other scene, 2024-25 included, gets none.
    - Attrs: `store_attrs` of the rule plus `mode: teacher`, `teacher_setting`, `teacher_p_star`, `teacher_ignore_halfwidth`, `teacher_feature_groups`, `teacher_correction_km` (null if not applied), `teacher_model_path`.
  - `cmd_label --mode teacher` calls it.

- [ ] **Step 1: Write the failing tests**

```python
def test_teacher_store_layout(tmp_path):
    cfg = teacher_label_config(tmp_path)  # tiny dataset, a saved toy teacher, no held-out file
    label_teacher_dataset(Path(cfg["data"]["processed_path"]), tmp_path / "labels_teacher.zarr", cfg)
    store = xr.open_zarr(tmp_path / "labels_teacher.zarr", consolidated=False)
    assert store["label"].dtype == np.uint8 and store["fog_probability"].dtype == np.uint8
    assert set(np.unique(store["label"].values)) <= {0, 1, 2, 255}
    assert store.attrs["mode"] == "teacher" and "teacher_p_star" in store.attrs


def test_untrusted_and_obscured_follow_the_rule(tmp_path):
    cfg, rule = teacher_label_config(tmp_path, with_rule=True)
    label_teacher_dataset(Path(cfg["data"]["processed_path"]), tmp_path / "labels_teacher.zarr", cfg)
    labels = xr.open_zarr(tmp_path / "labels_teacher.zarr", consolidated=False)["label"].values
    assert np.array_equal(labels == 2, rule == 2) and (labels[rule == 255] != 1).all()


def test_missing_or_mismatched_model_stops_before_labelling(tmp_path):
    cfg = teacher_label_config(tmp_path)
    cfg["teacher"]["feature_groups"] = ["bt"]
    with pytest.raises(ValueError, match="feature"):
        label_teacher_dataset(Path(cfg["data"]["processed_path"]), tmp_path / "labels_teacher.zarr", cfg)
    assert not (tmp_path / "labels_teacher.zarr").exists()


def test_correction_only_when_kept_and_only_in_the_tuning_season(tmp_path):
    cfg = teacher_label_config(tmp_path, correction_kept=True, seasons=["2023-24", "2024-25"])
    label_teacher_dataset(Path(cfg["data"]["processed_path"]), tmp_path / "labels_teacher.zarr", cfg)
    store = xr.open_zarr(tmp_path / "labels_teacher.zarr", consolidated=False)
    assert corrected_scenes(store) == tuning_season_scenes(store)


def test_nan_channel_is_ignored_without_warnings(tmp_path, recwarn):
    cfg = teacher_label_config(tmp_path, nan_pixel=(2, 2))
    label_teacher_dataset(Path(cfg["data"]["processed_path"]), tmp_path / "labels_teacher.zarr", cfg)
    store = xr.open_zarr(tmp_path / "labels_teacher.zarr", consolidated=False)
    assert (store["label"].values[:, 2, 2] == 255).all() and (store["fog_probability"].values[:, 2, 2] == 255).all()
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]
```

`teacher_label_config`, `corrected_scenes` and `tuning_season_scenes` are local helpers. The toy teacher's correction feeder should carry a residual large enough to change `fog_probability`, so `corrected_scenes` can find the scenes where it changed.

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_teacher_labels.py -v`
Expected: FAIL (`fogseg.teacher_labels` not found)

- [ ] **Step 3: Implement, keeping the label store's temporary-store and rename pattern of `label_dataset`**

- [ ] **Step 4: Run the tests and ruff**

Run: `pytest -q && ruff check . && ruff format --check .`
Expected: all pass (the existing label-store tests unchanged), ruff clean

- [ ] **Step 5: Commit**

```bash
git add src/fogseg/teacher_labels.py src/fogseg/label_store.py src/fogseg/cli.py tests/test_teacher_labels.py
git commit -m "Write dense teacher labels with fogseg label --mode teacher"
```

---

### Task 8: Real-data run, notes and PR (controller)

**Files:**
- Modify: `configs/default.yaml` (only if the teacher is kept), `docs/notes/better_labels.md`, `docs/DATASET_CARD.md`, `README.md`

- [ ] **Step 1: Preconditions.**
  - `pip install scikit-learn` in the venv.
  - Check that `processed/dem_ceu.zarr` exists.
  - Check whether `era5_ceu.zarr` exists. If not and `~/.cdsapirc` exists, run `fogseg era5` and then `fogseg diagnose` to fill the phase A "pending" row.
  - If neither exists, **stop and ask the owner** whether to wait for the key or to run without the `era5` group. Never read or print the key.

- [ ] **Step 2: Cross-validation and fit** (background)

Run: `fogseg teacher`
Expected:
- `stations/teacher_cv.csv` with 8 settings × 5 rows.
- `teacher_correction_cv.csv`, `teacher_oof.csv`, `teacher_config.json` and `teacher.joblib`.
- A winner logged with its pooled POD, FAR and p*.

- [ ] **Step 3: Held-out run, once.** Before it, check:
  - the CV table is complete;
  - the config record is from this run;
  - nothing was re-validated or re-split since.

Run: `fogseg teacher --held-out`
Expected: `stations/teacher_held_out.csv` and `.json`, with both decisions logged.

- [ ] **Step 4: Dense labels.** Run `fogseg label --mode teacher`. If the teacher is kept, set `labels.mode: teacher` in the config and run `fogseg validate --season 2023-24`. Otherwise the config stays, and the store is a documented benchmark.

- [ ] **Step 5: Notes.** `better_labels.md` gains a phase C section:
  - the features and settings;
  - the cross-validation table (pooled POD, FAR, CSI and p* per setting);
  - the correction table;
  - the held-out table with the rule, which is the phase A baseline, the teacher and the corrected teacher, with intervals and both decisions;
  - the third look at the held-out set.

  The card describes the teacher labels and `fog_probability`; the README lists `fogseg teacher` and `--mode teacher`.

- [ ] **Step 6: Verify, commit, push and open the PR**

Run: `pytest -q && ruff check . && ruff format --check .`
Then:
- commit;
- push `feature/better-labels-teacher`;
- open the PR into `feature/better-labels-rule` with `~/.local/bin/gh`.

- [ ] **Step 7: Ask the owner** whether to open the 2024-25 test season once for the frozen baseline and the final labels (spec Section 1). Nothing runs on it without that answer.
