# Better labels, phase A: evaluation protocol, matching, data, diagnosis, obscured class — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A frozen held-out evaluation inside 2023-24, fairer station matching, ERA5 and DEM on the SEVIRI grid, a
diagnosis of the low-BTD misses and an "obscured" label value, with the fixed 4.0 K rule re-scored as the baseline.

**Architecture:** New pure modules (`splits.py`, `parallax.py`, `dem.py`, `era5.py`, `diagnosis.py`) with thin CLI
commands, following the existing pattern: downloads write files once, scoring reads files offline. `scores.py`,
`validation.py`, `labels.py`, `label_store.py`, `metar.py` and `observations.py` are extended, not restructured.
The matched-FAR threshold choice of spec Section 3.3 is first needed by the phase B variants, so it is built in the
phase B plan.

**Tech Stack:** Python 3.11+, numpy, pandas, xarray, zarr 3, scipy; new dependencies `cdsapi` (ERA5) and `rasterio`
(DEM GeoTIFFs).

**Spec:** `docs/superpowers/specs/2026-10-31-better-labels-design.md` (Sections 3, 4, 7, 8, 9, 10)

## Global Constraints

- Owner's code style: "we do not comment dumb stuff. Linting is always there the code should look absolutely readable. Even if you can make things much smaller if you see that the readability shrinks just select a simpler approach." Match the docstring style of the module you edit.
- `ruff check .` and `ruff format --check .` clean (ruff pinned `0.17.*`, line length 110).
- Tests use synthetic data only; no network call and no file under `/mnt/d` in any test.
- The 2024-25 season is never used for a decision: `split` and `diagnose` refuse any season but `stations.tune_season`; `validate` keeps `check_season_allowed`.
- Never read, print or log the CDS key; only test whether `~/.cdsapirc` exists or `CDSAPI_KEY` is set.
- Label values: `0` no fog, `1` fog, `2` obscured, `255` ignore.
- Full count: label `1` is "yes"; labels `0`, `2`, `255` are "no"; truth `-1` is skipped.
- Folds: `stations.n_folds: 5`; fold 5 is held out; night blocks of `stations.night_block_days: 14` from the season start, block *i* → fold *i* mod 5 + 1; station cells of `stations.station_cell_deg: 2.0` from the bbox south-west corner, dealt to folds at random with `stations.seed`.
- Parallax cloud top: station elevation (0 if missing) + `stations.parallax_cloud_top_m: 500.0`.
- Run Python in WSL: `source ~/code/seviri-fog-segmentation/.venv/bin/activate; export PYTHONPATH=<worktree>/src` (the editable install points at the main checkout). Run git with `GIT_DIR=~/code/seviri-fog-segmentation/.git/worktrees/better-labels-4df4fc GIT_WORK_TREE=.`; never `git worktree prune`.
- Commit messages: imperative sentence like the existing history, ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. A station without elevation must get a parallax shift for 0 m + 500 m, not a NaN pixel — test in Task 4.
2. A station on the grid but just outside the bbox (e.g. lon 1.95) must still get a station cell and fold — test in Task 2.
3. A scene time whose ERA5 month failed to download must give NaN fields, not a crash — test in Task 7.
4. A DEM tile that does not exist (sea, HTTP 404) must count as elevation 0 and not be requested again — test in Task 6.
5. An observation table written before Task 3 (no `auto`, `ceiling_amount`, `shallow_fog` columns) must still validate, with those columns empty — test in Task 8.

---

### Task 1: Full-count scores, HSS and a generic paired difference

**Files:**
- Modify: `src/fogseg/scores.py`, `src/fogseg/tuning.py:15,177`
- Test: `tests/test_scores.py`, `tests/test_tuning.py`

**Interfaces:**
- Produces:
  - `contingency(label, truth, full_count: bool = False) -> np.ndarray` — label `2` always counts as "no"; with `full_count` label `255` counts as "no" too instead of being skipped.
  - `counts_by_night(night, label, truth, nights, full_count: bool = False) -> np.ndarray`
  - `skill(counts)` gains `"hss"`; `score_summary` rows gain an `hss` column after `bias`.
  - `paired_difference(counts_a, counts_b, score: str, n_samples: int, seed: int) -> tuple[float, float]`; `paired_csi_difference` is removed and `tuning.py` calls `paired_difference(..., "csi", ...)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_contingency_counts_obscured_as_no():
    label = np.array([1, 2, 2, 0])
    truth = np.array([1, 1, 0, 0])
    assert contingency(label, truth).tolist() == [1, 1, 0, 2]


def test_contingency_full_count_counts_ignored_as_no():
    label = np.array([1, 255, 255, 0])
    truth = np.array([1, 1, 0, -1])
    assert contingency(label, truth).tolist() == [1, 0, 0, 0]
    assert contingency(label, truth, full_count=True).tolist() == [1, 1, 0, 1]


def test_skill_hss_matches_hand_value():
    # a=hits 50, c=misses 10, b=false alarms 20, d=correct negatives 120
    hss = skill(np.array([50, 10, 20, 120]))["hss"]
    assert hss == pytest.approx(2 * (50 * 120 - 20 * 10) / ((50 + 10) * (10 + 120) + (50 + 20) * (20 + 120)))


def test_paired_difference_is_zero_for_identical_counts():
    counts = np.array([[5, 2, 1, 10], [3, 3, 2, 8], [6, 1, 0, 9]])
    assert paired_difference(counts, counts, "pod", 200, 0) == (0.0, 0.0)


def test_paired_difference_sign_follows_order():
    better = np.array([[8, 2, 1, 10], [7, 3, 2, 8]])
    worse = np.array([[5, 5, 1, 10], [4, 6, 2, 8]])
    low, high = paired_difference(better, worse, "pod", 200, 0)
    assert low > 0 and high > 0
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_scores.py -v`
Expected: FAIL (`full_count` unexpected keyword, `hss` KeyError, `paired_difference` not defined)

- [ ] **Step 3: Implement the changes in `scores.py`, replace the call in `tuning.py`**

HSS from counts in `COUNT_NAMES` order: `2 (a d − b c) / ((a + c)(c + d) + (a + b)(b + d))` with `a` hits, `b` false alarms, `c` misses, `d` correct negatives, via `ratio`. Update existing score-table column expectations in `tests/test_tuning.py` and `tests/test_validation.py` for the new `hss` column.

- [ ] **Step 4: Run the whole suite**

Run: `pytest -q && ruff check . && ruff format --check .`
Expected: all pass, ruff clean

- [ ] **Step 5: Commit**

```bash
git add src/fogseg/scores.py src/fogseg/tuning.py tests/
git commit -m "Score the full count and HSS, and pair any score in the bootstrap"
```

---

### Task 2: Night blocks, station cells and the split file

**Files:**
- Create: `src/fogseg/splits.py`, `tests/test_splits.py`
- Modify: `src/fogseg/cli.py` (new `split` command), `configs/default.yaml` (`stations:` keys)

**Interfaces:**
- Consumes: `validation.season_bounds(cfg, season)`, `validation.night_of`, `validation.read_observations`, `validation.on_grid_stations`.
- Produces:
  - `night_block_start(night: np.ndarray, season_start: np.datetime64, block_days: int) -> np.ndarray` (datetime64[D] start of each night's block)
  - `station_cell(lon: np.ndarray, lat: np.ndarray, bbox: list[float], cell_deg: float) -> np.ndarray` (str ids `"c{col}_{row}"`, col/row clipped to the bbox cell grid)
  - `make_splits(season_start: np.datetime64, season_end: np.datetime64, bbox: list[float], stations_cfg: dict) -> pd.DataFrame` with columns `kind` (`night_block` / `station_cell`), `key` (ISO date or cell id), `fold` (1..n_folds)
  - `assign_folds(cases: pd.DataFrame, splits: pd.DataFrame, season_start: np.datetime64, bbox: list[float], stations_cfg: dict) -> pd.DataFrame` — adds `night_fold`, `station_fold` (int); raises `ValueError` naming the key if a block or cell is missing from `splits`
  - `held_out(cases, n_folds) -> pd.Series[bool]` — `night_fold == n_folds or station_fold == n_folds`
  - `held_out_part(cases, n_folds) -> pd.Series[str]` — `"selection"`, `"nights"`, `"stations"` or `"both"`
  - `cv_folds(cases, n_folds) -> Iterator[tuple[int, pd.Series, pd.Series]]` — for k in 1..n_folds−1 over selection cases: `(k, train, test)`, train = night_fold ≠ k and station_fold ≠ k, test = night_fold = k or station_fold = k
  - `splits_path(data_dir: Path, season: str) -> Path` → `data_dir / f"splits_{season}.csv"`
  - CLI: `fogseg split [--season S] [--overwrite]`, default season `stations.tune_season`; refuses other seasons with `PermissionError`; refuses an existing file without `--overwrite`.
- Config (`stations:`): `night_block_days: 14`, `station_cell_deg: 2.0`, `n_folds: 5`, `matched_far: 0.28`, `emissivity_grid: [0.80, 1.00, 0.01]`, `parallax_cloud_top_m: 500.0`, `elevation_diff_bins: [-100.0, 100.0]`.

- [ ] **Step 1: Write the failing tests**

```python
STATIONS_CFG = {"night_block_days": 14, "station_cell_deg": 2.0, "n_folds": 5, "seed": 0}
BBOX = [2, 43, 20, 55]


def test_night_blocks_are_dealt_in_turn():
    splits = make_splits(np.datetime64("2023-10-01"), np.datetime64("2024-02-29"), BBOX, STATIONS_CFG)
    blocks = splits[splits["kind"] == "night_block"].sort_values("key")
    assert blocks["key"].iloc[0] == "2023-10-01"
    assert blocks["fold"].tolist()[:6] == [1, 2, 3, 4, 5, 1]


def test_station_cells_cover_the_bbox_and_are_balanced():
    cells = make_splits(np.datetime64("2023-10-01"), np.datetime64("2024-02-29"), BBOX, STATIONS_CFG)
    cells = cells[cells["kind"] == "station_cell"]
    assert len(cells) == 9 * 6
    assert cells["fold"].value_counts().max() - cells["fold"].value_counts().min() <= 1


def test_station_just_outside_the_bbox_gets_the_edge_cell():
    assert station_cell(np.array([1.95]), np.array([43.1]), BBOX, 2.0).tolist() == ["c0_0"]


def test_held_out_and_selection_are_disjoint_and_cv_never_trains_on_test():
    cases = pd.DataFrame({"night_fold": [1, 2, 5, 3, 1], "station_fold": [1, 5, 2, 3, 2]})
    assert held_out(cases, 5).tolist() == [False, True, True, False, False]
    assert held_out_part(cases, 5).tolist() == ["selection", "stations", "nights", "selection", "selection"]
    for k, train, test in cv_folds(cases, 5):
        assert not (train & test).any()
        assert not (train & held_out(cases, 5)).any()
        assert ((cases["night_fold"] == k) | (cases["station_fold"] == k))[test].all()


def test_assign_folds_names_a_missing_cell():
    splits = pd.DataFrame({"kind": ["night_block"], "key": ["2023-10-01"], "fold": [1]})
    cases = pd.DataFrame({"night": [np.datetime64("2023-10-02")], "lon": [5.0], "lat": [50.0]})
    with pytest.raises(ValueError, match="c1_3"):
        assign_folds(cases, splits, np.datetime64("2023-10-01"), BBOX, STATIONS_CFG)
```

Also a CLI test with the existing `make_station_dataset` helper: `split` writes the file, a second run without `--overwrite` raises `FileExistsError`, `--season 2024-25` raises `PermissionError`.

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_splits.py -v`
Expected: FAIL (module `fogseg.splits` not found)

- [ ] **Step 3: Implement `splits.py`, the `split` command and the config keys**

Cells are dealt like blocks but over `np.random.default_rng(seed).permutation` of the cell ids, so fold sizes differ by at most one.

- [ ] **Step 4: Run tests and ruff**

Run: `pytest -q && ruff check . && ruff format --check .`
Expected: all pass, ruff clean

- [ ] **Step 5: Commit**

```bash
git add src/fogseg/splits.py src/fogseg/cli.py configs/default.yaml tests/test_splits.py
git commit -m "Split the tuning season into night blocks and station cells with a held-out fold"
```

---

### Task 3: METAR AUTO, ceiling amount and shallow fog

**Files:**
- Modify: `src/fogseg/metar.py`, `src/fogseg/observations.py:29,139-169,193`
- Test: `tests/test_metar.py`, `tests/test_observations.py`

**Interfaces:**
- Produces:
  - `metar.is_auto(metar: str) -> bool` — `AUTO` token in the main body
  - `metar.ceiling_amount(metar: str) -> str` — amount (`"BKN"`, `"OVC"` or `"VV"`) of the layer `ceiling_m` uses; `""` when there is none
  - `metar.has_shallow_fog(metar: str) -> bool` — `MIFG` or `BCFG` (with optional `+`/`-`/`VC` prefix) in the main body
  - `observations.OBSERVATION_COLUMNS` gains `auto`, `ceiling_amount`, `shallow_fog`; METAR rows fill them, SYNOP rows hold `pd.NA` / `""`.

- [ ] **Step 1: Write the failing tests**

```python
def test_is_auto():
    assert is_auto("EDDF 202050Z AUTO 24005KT 0300 FG VV001 04/04 Q1020")
    assert not is_auto("EDDF 202050Z 24005KT 9999 FEW040 04/02 Q1020 RMK AUTO")


def test_ceiling_amount_is_the_layer_of_the_ceiling():
    assert ceiling_amount("LOWW 202050Z 0800 BR SCT002 BKN004 OVC010 03/03 Q1025") == "BKN"
    assert ceiling_amount("LOWW 202050Z 0100 FG VV001 03/03 Q1025") == "VV"
    assert ceiling_amount("LOWW 202050Z CAVOK 03/01 Q1025") == ""


def test_shallow_fog():
    assert has_shallow_fog("EBBR 202050Z 4000 MIFG SKC 01/01 Q1030")
    assert has_shallow_fog("EBBR 202050Z 3000 BCFG FEW003 01/01 Q1030")
    assert not has_shallow_fog("EBBR 202050Z 0200 FG VV001 01/01 Q1030")


def test_parse_metar_csv_adds_the_new_columns():
    text = "station,valid,metar\nEDDF,2023-11-20 20:50,EDDF 202050Z AUTO 0800 MIFG BKN003 04/04 Q1020\n"
    row = parse_metar_csv(text).iloc[0]
    assert (row["auto"], row["ceiling_amount"], row["shallow_fog"]) == (True, "BKN", True)
```

And `parse_synop_text` rows have `ceiling_amount == ""` and missing `auto` / `shallow_fog`.

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_metar.py tests/test_observations.py -v`
Expected: FAIL (names not defined, missing columns)

- [ ] **Step 3: Implement**

`ceiling_amount` walks the main body with the same layer patterns as `ceiling_m`, so both always agree on the layer; refactor a shared `_ceiling_layer(metar) -> re.Match | None` only if it keeps both functions readable.

- [ ] **Step 4: Run tests and ruff**

Run: `pytest -q && ruff check . && ruff format --check .`
Expected: all pass, ruff clean

- [ ] **Step 5: Commit**

```bash
git add src/fogseg/metar.py src/fogseg/observations.py tests/test_metar.py tests/test_observations.py
git commit -m "Decode AUTO, the ceiling cover amount and shallow fog from METARs"
```

---

### Task 4: Parallax shift of the station pixel

**Files:**
- Create: `src/fogseg/parallax.py`, `tests/test_parallax.py`
- Modify: `src/fogseg/validation.py:153-162` (`on_grid_stations`)
- Test: `tests/test_validation.py`

**Interfaces:**
- Produces:
  - `apparent_position(lon: np.ndarray, lat: np.ndarray, height_m: np.ndarray, satellite_lon: float = 0.0) -> tuple[np.ndarray, np.ndarray]` — where a point at `height_m` above a spherical Earth (R 6371 km, satellite 42,164 km from the centre over the equator) appears on the ground seen from the satellite
  - `on_grid_stations` shifts each station to `apparent_position(lon, lat, elevation_m.fillna(0) + stations_cfg["parallax_cloud_top_m"])` before `select_stations`; the returned `lat`/`lon` stay the true station position.

- [ ] **Step 1: Write the failing tests**

```python
def test_shift_points_away_from_the_subsatellite_point():
    lon, lat = apparent_position(np.array([0.0]), np.array([50.0]), np.array([1000.0]))
    assert lat[0] > 50.0 and lon[0] == pytest.approx(0.0, abs=1e-9)


def test_shift_size_at_50_north_for_one_kilometre():
    _, lat = apparent_position(np.array([0.0]), np.array([50.0]), np.array([1000.0]))
    shift_km = (lat[0] - 50.0) * np.pi / 180 * 6371
    assert shift_km == pytest.approx(1.6, abs=0.15)


def test_zero_height_is_no_shift():
    lon, lat = apparent_position(np.array([10.0]), np.array([45.0]), np.array([0.0]))
    assert (lon[0], lat[0]) == pytest.approx((10.0, 45.0))
```

Add `test_station_without_elevation_is_shifted_for_the_cloud_top_alone` in `tests/test_validation.py`: a synthetic
grid whose pixel rows are 0.5 km apart around 50° N, one station with `elevation_m` NaN; `on_grid_stations` must
select the row of `apparent_position(..., 500 m)` (one row north of the true position), never a NaN or missing row.

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_parallax.py tests/test_validation.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement `parallax.py` and the shift in `on_grid_stations`**

Ray from the satellite through the elevated point, first intersection with the sphere of radius R beyond the point:

```python
satellite = np.array([SATELLITE_RADIUS_KM * cos(sat_lon), SATELLITE_RADIUS_KM * sin(sat_lon), 0.0])
point = (EARTH_RADIUS_KM + height_km) * unit_vector(lon, lat)
direction = point - satellite
# |satellite + t * direction| = EARTH_RADIUS_KM, take the smaller root; t >= 1 for any height >= 0
```

- [ ] **Step 4: Run tests and ruff**

Run: `pytest -q && ruff check . && ruff format --check .`
Expected: all pass, ruff clean

- [ ] **Step 5: Commit**

```bash
git add src/fogseg/parallax.py src/fogseg/validation.py tests/test_parallax.py tests/test_validation.py
git commit -m "Match each station to the pixel where its low cloud appears from the satellite"
```

---

### Task 5: Obscured label value

**Files:**
- Modify: `src/fogseg/labels.py`, `src/fogseg/label_store.py`, `src/fogseg/inspect.py`, `docs/DATASET_CARD.md` (Labels section)
- Test: `tests/test_labels.py`, `tests/test_label_store.py`, `tests/test_inspect.py`

**Interfaces:**
- Produces:
  - `labels.OBSCURED = 2`; `label_scene` sets it on trusted pixels failing `warm_and_not_cirrus`, before the fog tests, so it never overlaps 1 or 255.
  - `label_store.label_fractions(labels) -> tuple[float, float, float]` — fog share of decided pixels (0 or 1), ignored share and obscured share of all pixels; the store gains `obscured_fraction` (time) and the `label` long name lists value 2.
  - `inspect.draw_labels_panel` shows obscured pixels in their own colour with a legend entry `obscured (2)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_cold_or_cirrus_trusted_pixel_is_obscured():
    # pixel 0: cold top (IR_108 250 K); pixel 1: cirrus split 2 K; pixel 2: clear warm ground
    labels = label_scene(ir039, ir108, ir120, sza, 4.0, PARAMS)
    assert labels.tolist() == [[OBSCURED, OBSCURED, NO_FOG]]


def test_untrusted_cold_pixel_stays_ignored():
    labels = label_scene(ir039, ir108, ir120, np.full_like(ir108, 80.0), 4.0, PARAMS)
    assert labels[0, 0] == IGNORE


def test_label_fractions_report_obscured_separately():
    fog, ignored, obscured = label_fractions(np.array([1, 0, 2, 2, 255], dtype=np.uint8))
    assert (fog, ignored, obscured) == (0.5, 0.2, 0.4)
```

Plus: the store written by `label_dataset` has `obscured_fraction`; the labels panel test finds the `obscured (2)` legend entry.

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_labels.py tests/test_label_store.py tests/test_inspect.py -v`
Expected: FAIL

- [ ] **Step 3: Implement and update the dataset card**

Card: value table gains `2 | obscured: a higher or thin cirrus cloud hides what is below; neither fog nor clear`, the variable table gains `obscured_fraction`, the decision diagram gets the obscured branch, `fog_fraction` text says decided means 0 or 1.

- [ ] **Step 4: Run tests and ruff**

Run: `pytest -q && ruff check . && ruff format --check .`
Expected: all pass, ruff clean

- [ ] **Step 5: Commit**

```bash
git add src/fogseg/labels.py src/fogseg/label_store.py src/fogseg/inspect.py docs/DATASET_CARD.md tests/
git commit -m "Label pixels under high or cirrus cloud as obscured instead of no fog"
```

---

### Task 6: DEM on the SEVIRI grid

**Files:**
- Create: `src/fogseg/dem.py`, `tests/test_dem.py`
- Modify: `src/fogseg/cli.py` (new `dem` command), `configs/default.yaml` (`dem:`), `pyproject.toml` (`rasterio`)

**Interfaces:**
- Produces:
  - `tile_name(lat: int, lon: int) -> str` → e.g. `tile_name(47, 8) == "Copernicus_DSM_COG_30_N47_00_E008_00_DEM"`
  - `tile_url(name: str) -> str` → `https://copernicus-dem-90m.s3.amazonaws.com/{name}/{name}.tif`
  - `tiles_for_grid(lon: np.ndarray, lat: np.ndarray) -> list[tuple[int, int]]` — south-west corners of all 1° tiles touching the grid's lon/lat range
  - `fetch_tile(lat: int, lon: int, raw_dir: Path, fetch: Callable[[str, Path], bool]) -> Path | None` — cached `.tif`; a missing tile writes `<name>.sea` and returns None, also on later calls without fetching
  - `tile_samples(path: Path, factor: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]` — lon, lat, block-mean elevation of `factor`×`factor` blocks (rasterio, transform for coordinates)
  - `grid_statistics(sample_lon, sample_lat, sample_elevation, lon, lat) -> tuple[np.ndarray, np.ndarray]` — mean and std of the samples nearest to each grid pixel (`scipy.spatial.cKDTree` on unit vectors); pixels without samples get 0 and 0 (sea)
  - `build_dem_store(data: xr.Dataset, raw_dir: Path, out_path: Path, fetch) -> None` — writes `elevation_mean`, `elevation_std` (y, x, metres) with the dataset's `y`, `x`, `lat`, `lon`
  - `read_dem(path: Path) -> xr.Dataset | None` — None with a warning when the store is missing
  - CLI: `fogseg dem [--overwrite]`; config `dem: {raw_dir: "/mnt/d/fogseg-data/raw/dem", out_path: "/mnt/d/fogseg-data/processed/dem_ceu.zarr"}`

- [ ] **Step 1: Write the failing tests**

```python
def test_tile_name_for_negative_longitude():
    assert tile_name(43, -1) == "Copernicus_DSM_COG_30_N43_00_W001_00_DEM"


def test_missing_tile_is_marked_sea_and_not_fetched_again(tmp_path):
    calls = []

    def fetch(url, path):
        calls.append(url)
        return False

    assert fetch_tile(54, 2, tmp_path, fetch) is None
    assert fetch_tile(54, 2, tmp_path, fetch) is None
    assert len(calls) == 1


def test_grid_statistics_mean_std_and_sea():
    lon, lat = np.meshgrid([5.0, 6.0], [45.0])
    mean, std = grid_statistics(
        np.array([5.0, 5.001, 5.002]), np.array([45.0, 45.0, 45.0]), np.array([100.0, 200.0, 300.0]), lon, lat
    )
    assert mean.tolist() == [[200.0, 0.0]] and std[0, 0] == pytest.approx(np.std([100, 200, 300]))
```

Plus `tile_samples` on a tiny GeoTIFF written in the test with `rasterio` (4×4 pixels, factor 2) and `build_dem_store` with a fake `fetch` that copies that file.

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_dem.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement**

The CLI passes a `fetch` that uses `urllib.request` and returns False on HTTP 404 (raising on other errors), writing through a `.part` file like `observations.download`.

- [ ] **Step 4: Run tests and ruff**

Run: `pytest -q && ruff check . && ruff format --check .`
Expected: all pass, ruff clean

- [ ] **Step 5: Commit**

```bash
git add src/fogseg/dem.py src/fogseg/cli.py configs/default.yaml pyproject.toml tests/test_dem.py
git commit -m "Average the Copernicus DEM onto the SEVIRI grid"
```

---

### Task 7: ERA5 on the SEVIRI grid

**Files:**
- Create: `src/fogseg/era5.py`, `tests/test_era5.py`
- Modify: `src/fogseg/cli.py` (new `era5` command), `configs/default.yaml` (`era5:`), `pyproject.toml` (`cdsapi`)

**Interfaces:**
- Produces:
  - `ERA5_VARIABLES = {"2m_temperature": "t2m", "skin_temperature": "skt", "2m_dewpoint_temperature": "d2m", "snow_depth": "sd"}`
  - `month_requests(times: np.ndarray, area: list[float], variables: list[str]) -> list[tuple[str, dict]]` — one `(f"{year}-{month:02d}", request)` per month with scenes; request keys `product_type: ["reanalysis"]`, `variable`, `year`, `month`, `day` (sorted days of that month's scenes), `time` (sorted `"HH:00"` of the scenes), `area` [N, W, S, E], `data_format: "netcdf"`, `download_format: "unarchived"`
  - `era5_area(bbox: list[float], margin: float = 0.5) -> list[float]` → `[55.5, 1.5, 42.5, 20.5]` for the project bbox
  - `has_cds_credentials() -> bool` — `~/.cdsapirc` exists or `CDSAPI_KEY` is set; never opens the file
  - `download_months(requests, raw_dir: Path, retrieve: Callable[[dict, Path], None]) -> list[str]` — skips existing `era5_<month>.nc`, writes through `.part`, logs and returns the months that failed
  - `regrid(field: xr.DataArray, lon: np.ndarray, lat: np.ndarray) -> np.ndarray` — bilinear from the regular ERA5 lat/lon grid (either latitude order) to the 2-D grid
  - `build_era5_store(times: np.ndarray, lon, lat, y, x, raw_dir: Path, out_path: Path) -> None` — variables `t2m`, `skt`, `d2m` (K) and `sd` (m water equivalent), dims (time, y, x), one field per scene time at the ERA5 hour of the scene; times without a file or hour are NaN; ERA5 time coordinate may be `valid_time` or `time`
  - CLI: `fogseg era5 [--overwrite]` — both seasons; without credentials it stops with a message pointing to `https://cds.climate.copernicus.eu/how-to-api`; config `era5: {raw_dir: "/mnt/d/fogseg-data/raw/era5", out_path: "/mnt/d/fogseg-data/processed/era5_ceu.zarr", variables: [2m_temperature, skin_temperature, 2m_dewpoint_temperature, snow_depth]}`

- [ ] **Step 1: Write the failing tests**

```python
def test_month_requests_group_scene_days_and_hours():
    times = np.array(["2023-11-20T21:00", "2023-11-21T03:00", "2023-12-01T00:00"], dtype="datetime64[ns]")
    requests = month_requests(times, [55.5, 1.5, 42.5, 20.5], ["2m_temperature"])
    assert [month for month, _ in requests] == ["2023-11", "2023-12"]
    november = requests[0][1]
    assert november["day"] == ["20", "21"] and november["time"] == ["03:00", "21:00"]


def test_regrid_is_bilinear_for_descending_latitudes():
    field = xr.DataArray([[2.0, 2.0], [0.0, 0.0]], coords={"latitude": [46.0, 45.0], "longitude": [5.0, 6.0]})
    assert regrid(field, np.array([[5.5]]), np.array([[45.5]]))[0, 0] == pytest.approx(1.0)


def test_scene_without_era5_file_is_nan(tmp_path):
    # one month file written by the test for 2023-11; a scene in 2023-12 has no file
    ...
    assert np.isnan(store["t2m"].sel(time="2023-12-01T00:00")).all()


def test_failed_month_is_reported_and_not_written(tmp_path):
    def retrieve(request, path):
        raise RuntimeError("queue full")

    assert download_months([("2023-11", {})], tmp_path, retrieve) == ["2023-11"]
    assert not list(tmp_path.glob("*.nc"))
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_era5.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement**

The CLI builds `retrieve` from `cdsapi.Client().retrieve("reanalysis-era5-single-levels", request, str(path))`; `cdsapi` is imported inside the command only.

- [ ] **Step 4: Run tests and ruff**

Run: `pytest -q && ruff check . && ruff format --check .`
Expected: all pass, ruff clean

- [ ] **Step 5: Commit**

```bash
git add src/fogseg/era5.py src/fogseg/cli.py configs/default.yaml pyproject.toml tests/test_era5.py
git commit -m "Download ERA5 surface fields for the scene times and put them on the SEVIRI grid"
```

---

### Task 8: Case and score tables with strata, folds and the full count

**Files:**
- Modify: `src/fogseg/validation.py`, `src/fogseg/cli.py` (`cmd_validate`)
- Test: `tests/test_validation.py`

**Interfaces:**
- Consumes: Task 1 `counts_by_night(..., full_count=True)`, `paired_difference`; Task 2 `splits_path`, `assign_folds`, `held_out_part`; Task 3 observation columns; Task 6 `read_dem`.
- Produces:
  - `CASE_COLUMNS` gains `auto`, `ceiling_amount`, `shallow_fog`, `elevation_diff_m`; `station_cases(..., dem: xr.Dataset | None = None)` fills `elevation_diff_m` = station elevation − `elevation_mean` at the case pixel (NaN without DEM) and takes the METAR columns from the observations when present (empty otherwise).
  - `season_cases(..., dem=None)` passes it on.
  - `score_row` adds full-count columns `pod_full`, `pod_full_low`, `pod_full_high`, `far_full`, `far_full_low`, `far_full_high`, `csi_full`, `hss_full`, `bias_full`.
  - `score_table(cases, stations_cfg, n_samples, seed)` (takes `stations_cfg` for `lat_bands` and `elevation_diff_bins`) adds views `auto` (`true`/`false`, METAR only), `ceiling_amount` (`BKN`/`OVC`/`VV`, FLS-by-ceiling cases only), `shallow_fog` (`true`), `elevation_diff` (`<-100`, `-100-100`, `>=100`), and, when the cases have `night_fold`, view `split` with groups `selection`, `held_out`, `held_out_nights`, `held_out_stations`, `held_out_both`.
  - `cmd_validate`: reads the DEM store if it exists; for the tuning season with a split file, adds `night_fold` / `station_fold` to the cases before writing them; logs the full-count POD / FAR of `main` and of `split/held_out`.

- [ ] **Step 1: Write the failing tests**

```python
def test_full_count_columns_count_ignored_fls_as_misses():
    cases = case_frame(label=[1, 255, 0, 1], truth_fls=[1, 1, 1, 0])
    row = score_row(View("main", "all", cases), 50, 0)
    assert (row["pod"], row["pod_full"]) == (pytest.approx(1 / 2), pytest.approx(1 / 3))
    assert row["far_full"] == pytest.approx(1 / 2)


def test_old_observation_table_gives_empty_metar_columns():
    # observations without auto / ceiling_amount / shallow_fog columns
    cases = station_cases(data, labels, old_observations, STATIONS_CFG, PARAMS)
    assert cases["ceiling_amount"].eq("").all() and cases["auto"].isna().all()


def test_elevation_diff_uses_the_dem_at_the_case_pixel():
    ...
    assert cases["elevation_diff_m"].tolist() == [station_elevation - 120.0]


def test_split_view_groups():
    table = score_table(cases_with_folds, STATIONS_CFG, 50, 0)
    groups = table.loc[table["view"] == "split", "group"].tolist()
    assert groups == ["selection", "held_out", "held_out_nights", "held_out_stations", "held_out_both"]
```

`case_frame` is a small local helper building a deduplicated, visible case table with the given labels and truth.

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_validation.py -v`
Expected: FAIL

- [ ] **Step 3: Implement**

Update `cmd_tune` for the new `score_table` signature only if it calls it; keep `tune` behaviour otherwise unchanged.

- [ ] **Step 4: Run tests and ruff**

Run: `pytest -q && ruff check . && ruff format --check .`
Expected: all pass, ruff clean

- [ ] **Step 5: Commit**

```bash
git add src/fogseg/validation.py src/fogseg/cli.py tests/test_validation.py
git commit -m "Score the full count, held-out folds and the new station strata"
```

---

### Task 9: Diagnosis of the misses

**Files:**
- Create: `src/fogseg/diagnosis.py`, `tests/test_diagnosis.py`
- Modify: `src/fogseg/cli.py` (new `diagnose` command)

**Interfaces:**
- Consumes: Task 2 split helpers, Task 7 store (`era5_ceu.zarr`, optional), Task 8 case table, `labels.local_std`, `validation.night_of`.
- Produces:
  - `LOW_BTD_K = 3.5`, `BAND_HIGH_K = 4.5`
  - `case_features(data: xr.Dataset, labels: xr.Dataset, era5: xr.Dataset | None, cases: pd.DataFrame) -> pd.DataFrame` — adds `btd`, `ir108_std3`, `sza`, `ir108_minus_wv062`, `t2m_minus_ir108` (NaN without ERA5), `fog_other_slot` (fog at the pixel in another scene of the same night)
  - `first_cause(cases: pd.DataFrame, params: LabelParams) -> pd.Series` — in this order: `twilight` (sza ≤ `night_min_sza`), `no_data` (btd NaN), `low_btd` (btd ≤ 3.5), `band` (btd < 4.5 and std < `smooth_max_std`), `texture` (std ≥ `smooth_max_std`), else `small_blob_or_other`
  - `diagnosis_table(cases: pd.DataFrame, params: LabelParams) -> pd.DataFrame` — long table with columns `section`, `variable`, `statistic`, `fls_missed`, `no_fls`:
    - section `first_cause`: one row per cause, statistic `count` and `share` (no_fls empty);
    - section `low_btd`: visible night cases with btd ≤ 3.5; categorical variables `ceiling_amount`, `auto`, `slot`, `fog_other_slot` give a `share:<value>` row per value; numeric variables `ir108_minus_wv062`, `t2m_minus_ir108`, `elevation_diff_m` give `q10`, `q25`, `q50`, `q75`, `q90`
  - CLI: `fogseg diagnose [--season S]`, default and only allowed season `stations.tune_season`; needs the split file (FileNotFoundError naming `fogseg split` otherwise) and uses **selection cases only**; reads `labels.mode`'s store and, if present, the ERA5 store; writes `stations/diagnosis_<season>.csv` and logs the first-cause shares.

- [ ] **Step 1: Write the failing tests**

```python
def test_first_cause_order():
    cases = pd.DataFrame(
        {
            "sza": [90, 100, 100, 100, 100, 100],
            "btd": [5.0, np.nan, 2.0, 4.2, 5.0, 5.0],
            "ir108_std3": [0.1, 0.1, 0.1, 0.1, 2.0, 0.1],
        }
    )
    assert first_cause(cases, PARAMS).tolist() == [
        "twilight",
        "no_data",
        "low_btd",
        "band",
        "texture",
        "small_blob_or_other",
    ]


def test_low_btd_section_compares_missed_fls_with_no_fls():
    table = diagnosis_table(cases_fixture(), PARAMS)
    row = table[
        (table["section"] == "low_btd")
        & (table["variable"] == "ceiling_amount")
        & (table["statistic"] == "share:OVC")
    ].iloc[0]
    assert (row["fls_missed"], row["no_fls"]) == (pytest.approx(0.5), pytest.approx(0.0))


def test_fog_other_slot_looks_only_inside_the_night():
    # one pixel: fog at 21:00 of night A, not fog at 03:00 of night A, fog at 21:00 of night B (not fog at its 03:00)
    features = case_features(data, labels, None, cases_at_both_slots_of_both_nights)
    assert features["fog_other_slot"].tolist() == [
        False,
        True,
        False,
        True,
    ]  # A 21:00, A 03:00, B 21:00, B 03:00
```

Plus a CLI test: `diagnose --season 2024-25` raises `PermissionError`; without a split file raises `FileNotFoundError` mentioning `fogseg split`.

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_diagnosis.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement**

- [ ] **Step 4: Run tests and ruff**

Run: `pytest -q && ruff check . && ruff format --check .`
Expected: all pass, ruff clean

- [ ] **Step 5: Commit**

```bash
git add src/fogseg/diagnosis.py src/fogseg/cli.py tests/test_diagnosis.py
git commit -m "Diagnose why visible station fog is missed, on selection cases only"
```

---

### Task 10: Real-data run, notes and PR

Run by the controller session, not a subagent: it needs the owner's CDS key and long downloads (run in the background).

**Files:**
- Create: `docs/notes/better_labels.md`
- Modify: `README.md` (commands `split`, `dem`, `era5`, `diagnose`), `docs/notes/station_validation.md` (link in "Next step"), `docs/DATASET_CARD.md` (DEM and ERA5 sources and licences: Copernicus DEM © DLR/Airbus, ERA5 under the Copernicus licence)

- [ ] **Step 1: Install the new dependencies into the venv**

Run: `pip install cdsapi rasterio` in the shared venv (its editable install points at the main checkout, so do not
reinstall the package from the worktree).
Expected: `python -c "import cdsapi, rasterio"` succeeds.

- [ ] **Step 2: Rebuild the inputs**

```bash
fogseg label --overwrite                 # obscured value 2
fogseg stations --source metar           # re-parse the cached raw METARs, no download
fogseg dem
fogseg era5                              # needs ~/.cdsapirc; background, resumable
fogseg split
```

Expected: `labels_fixed.zarr` has `obscured_fraction`; `observations_metar.csv` has `auto`, `ceiling_amount`, `shallow_fog`; `dem_ceu.zarr`, `era5_ceu.zarr`, `splits_2023-24.csv` exist.

- [ ] **Step 3: Re-score the baseline and diagnose**

```bash
fogseg validate --season 2023-24
fogseg diagnose
```

Expected: `scores_fixed_2023-24.csv` with full-count and `split` rows; `diagnosis_2023-24.csv`.

- [ ] **Step 4: Write `docs/notes/better_labels.md`**

Sections: the first diagnostic (all cases, descriptive) and the selection-only diagnosis; the protocol (folds, sizes of selection and held-out sets, three held-out parts); the baseline table (decided and full-count POD / FAR / CSI / HSS with intervals for `main`, `split/selection`, `split/held_out` and its three parts); what the strata show; what this means for phases B and C. Update README, station notes and dataset card.

- [ ] **Step 5: Verify, commit, push, open the PR**

```bash
pytest -q && ruff check . && ruff format --check .
git add docs README.md
git commit -m "Document the phase A protocol, the re-scored baseline and the miss diagnosis"
git push -u origin feature/better-labels-evaluation
~/.local/bin/gh pr create --base feature/station-validation --head feature/better-labels-evaluation
```

PR body ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`. Never merge.
