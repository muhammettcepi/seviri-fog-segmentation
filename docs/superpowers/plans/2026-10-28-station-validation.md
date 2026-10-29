# Station Validation and Threshold Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score the fog label stores against METAR and SYNOP observations (POD / FAR / CSI) and tune
`labels.dynamic.offset` and `labels.btd_fixed` on training-season stations.

**Architecture:** Pure decoders (`metar.py`, `synop.py`) and pure scoring (`scores.py`) are tested on hand-made input.
`observations.py` downloads (IEM, Ogimet) into a resumable raw cache and writes one observation table per source.
`validation.py` turns observations into truth, matches them to pixels and scenes, and builds the case and score
tables. `tuning.py` re-labels the training scenes over a threshold grid. Three CLI commands: `stations`, `validate`,
`tune`.

**Tech Stack:** Python ≥ 3.11, numpy, pandas, xarray / zarr, scipy (already used), stdlib `urllib` for downloads,
pytest, ruff 0.17.

**Spec:** `docs/superpowers/specs/2026-10-28-station-validation-design.md`

## Global Constraints

- Code style of the project owner: no comments that restate the code; readable over short; ruff check and ruff format
  clean (`line-length = 110`). Match the docstring style of `labels.py` and `label_store.py`.
- Every module starts with a one-line docstring and `from __future__ import annotations`.
- Tests use synthetic data only: no network, no files outside `tmp_path`. Network calls go through one
  module-level function `fetch_text(url: str, timeout: float) -> str` in `observations.py`, which tests monkeypatch.
- No new dependencies in `pyproject.toml`.
- The test season (2024-25) is never used for tuning. `tune` reads its season from `stations.tune_season`;
  `validate` refuses any other season unless `--final`.
- Truth limit 1000 m for visibility and ceiling; cover > 5 oktas (METAR: BKN, OVC or VV).
- Matching: nearest pixel within 5 km, station elevation ≤ 1000 m, observation closest to nominal time + 12 min
  within ± 30 min.
- Bootstrap: 1000 samples over whole nights, seed 0, 95 % interval (2.5 and 97.5 percentiles).
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Git in this worktree must run from WSL with `GIT_DIR=~/code/seviri-fog-segmentation/.git/worktrees/metar-validation-419635`
  and `GIT_WORK_TREE=.` (the `.git` file holds a Windows UNC path). The venv is `~/code/seviri-fog-segmentation/.venv`.

## Review Focus

1. A METAR with RVR (`R26/1200`) or a trend (`TEMPO 0800 FG`) after the main body: only the prevailing visibility of
   the main body counts. → Task 1 tests.
2. A SYNOP message with `NIL`, a missing `8` group, or `/` in `h`, `VV` or `N`: decoded as NaN or from `N`, never a
   crash. → Task 2 tests.
3. Two stations (or METAR and SYNOP) on the same pixel and scene: counted once in the pooled view, METAR kept.
   → Task 4 tests.
4. A score group with no cases or no FLS cases: POD / FAR / CSI are NaN, the table still writes. → Task 3 tests.
5. An Ogimet answer that is empty or has no `AAXX`: retried, not saved, run continues. → Task 5 tests.

---

### Task 1: METAR decoder

**Files:**
- Create: `src/fogseg/metar.py`
- Test: `tests/test_metar.py`

**Interfaces:**
- Produces:
  - `prevailing_visibility_m(metar: str) -> float` — first visibility group of the main body: `\d{4}` (with
    optional `NDV`) in m, `9999` → 10000, `CAVOK` → 10000, `////` or none → NaN.
  - `ceiling_m(metar: str) -> float` — base in m (`ft * 0.3048`) of the lowest `BKN`, `OVC` or `VV` layer of the
    main body; `inf` when the body has cloud information but no such layer (`FEW` / `SCT` only, `NSC`, `NCD`,
    `SKC`, `CLR`, `CAVOK`); NaN when the lowest `BKN`/`OVC`/`VV` layer has height `///`, or there is no cloud
    information at all.
- The main body ends at the first `TEMPO`, `BECMG`, `NOSIG` or `RMK` token. Cloud tokens:
  `(FEW|SCT|BKN|OVC)(\d{3}|///)(CB|TCU|///)?`, `VV(\d{3}|///)`. Tokens before the time group (`\d{6}Z`) are skipped,
  so `COR EDDM 280650Z ...` works.

- [ ] **Step 1: Write the failing tests** in `tests/test_metar.py`:
  - `test_visibility_from_main_body`: `"EDDM 280450Z AUTO 05002KT 7000 1300 R08L/P2000N BR NCD M04/M04 Q1034 NOSIG"` → 7000.
  - `test_visibility_ignores_rvr_and_trend`: `"LOWW 010050Z 00000KT 0400 R29/1200N FG VV002 01/01 Q1030 TEMPO 2000 BR"` → 400.
  - `test_cavok_and_9999_mean_10_km`: `"EDDF 010050Z 22005KT CAVOK 05/02 Q1020"` → 10000 and `ceiling_m` = inf;
    `"... 9999 FEW030 ..."` → 10000.
  - `test_missing_visibility_is_nan`: `"EDDM 010050Z AUTO 22005KT //// NCD 05/02 Q1020"` → NaN.
  - `test_ceiling_is_lowest_broken_layer`: `"... 9999 FEW005 SCT008 BKN012 OVC030 ..."` → `12 * 100 * 0.3048` (365.76).
  - `test_vertical_visibility_is_a_ceiling`: `"... 0100 FG VV001 ..."` → 30.48.
  - `test_few_and_scattered_are_no_ceiling`: `"... 9999 FEW005 SCT008 ..."` → inf.
  - `test_unknown_layer_height_is_nan`: `"... 9999 BKN/// ..."` → NaN; `"... 9999 //////CB ..."` (no other cloud) → NaN.
  - `test_trend_layers_are_ignored`: `"... 9999 SCT040 Q1020 BECMG BKN005"` → inf.
  - `test_correction_prefix`: `"COR EDDM 280650Z AUTO VRB01KT 4400 BR NCD M04/M05 Q1035"` → 4400.

- [ ] **Step 2: Run** `pytest tests/test_metar.py -v` — Expected: FAIL (module not found).
- [ ] **Step 3: Implement** both functions in `src/fogseg/metar.py` with a shared `main_body(metar: str) -> list[str]`
  (tokens after the time group, before the first trend / remark token). Module-level compiled regexes with readable names.
- [ ] **Step 4: Run** `pytest tests/test_metar.py -v && ruff check src tests && ruff format --check src tests` — Expected: PASS.
- [ ] **Step 5: Commit** `Decode visibility and ceiling from raw METAR`.

### Task 2: SYNOP decoder

**Files:**
- Create: `src/fogseg/synop.py`
- Test: `tests/test_synop.py`

**Interfaces:**
- Produces:
  - `CLOUD_BASE_LOWER_M = (0, 50, 100, 200, 300, 600, 1000, 1500, 2000, 2500)` — lower edge of each `h` class; using
    the lower edge makes `base < 1000` true exactly for `h ≤ 5`.
  - `visibility_m(vv: str) -> float` — WMO code table 4377: `00`–`50` → VV × 100 m; `51`–`55` → NaN; `56`–`80` →
    (VV − 50) km; `81`–`88` → ((VV − 80) × 5 + 30) km; `89` → 70 km; `90`–`99` → 0, 50, 200, 500, 1000, 2000, 4000,
    10000, 20000, 50000 m; `//` → NaN.
  - `@dataclass(frozen=True) class Synop: station: str; visibility_m: float; ceiling_m: float`
  - `decode_synop(message: str) -> Synop | None` — `None` for `NIL` or a message that is not `AAXX`.
- Ceiling rule (spec §4.3): `N = 9` → 0 m; cover = `Nh` from the first section-1 group starting with `8`, else `N`
  from `Nddff`; cover known and ≤ 5 → inf; cover > 5 (and ≠ 9) → `CLOUD_BASE_LOWER_M[h]`, NaN if `h` is `/`;
  cover unknown (`/`) → NaN. Section 1 ends at `333`, `555` or the end; a `Nddff` with `ff = 99` is followed by a
  `00fff` group, which must be skipped before looking for the `8` group.

- [ ] **Step 1: Write the failing tests** in `tests/test_synop.py`:
  - `test_visibility_code_table`: parametrized `("00", 0)`, `("09", 900)`, `("10", 1000)`, `("50", 5000)`,
    `("56", 6000)`, `("80", 30000)`, `("81", 35000)`, `("89", 70000)`, `("90", 0)`, `("94", 1000)`, `("97", 10000)`,
    `("52", nan)`, `("//", nan)`.
  - `test_missing_cover_gives_unknown_ceiling`: `"AAXX 28001 10004 46/60 /2209 10062 20046 30317 40317 333 55300=="` → station
    `"10004"`, visibility 10000, ceiling NaN (N is `/`).
  - `test_low_overcast_from_8_group`: `"AAXX 28031 10870 11408 82504 10012 20008 39950 40330 57010 70500 87500 333 ..."`
    (`h = 4`, `N = 8`, `Nh = 7`) → visibility 800, ceiling 300.
  - `test_cover_from_n_without_8_group`: `"AAXX 28031 10870 11560 32504 10012 20008=="` → ceiling inf (N = 3).
  - `test_sky_obscured`: `"AAXX 28031 10870 11/02 92502 10012 20008 70545=="` → visibility 200, ceiling 0.
  - `test_unknown_base_with_overcast_is_nan`: `"AAXX 28031 10870 11/60 82504 10012 87500=="` → ceiling NaN.
  - `test_high_wind_group_is_skipped`: Nddff with `ff = 99` followed by `00105`, then `87500`, `h = 4` → ceiling 300.
  - `test_nil_message`: `"AAXX 28031 10870 NIL=="` → `None`.

- [ ] **Step 2: Run** `pytest tests/test_synop.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement** `src/fogseg/synop.py`.
- [ ] **Step 4: Run** `pytest tests/test_synop.py -v` plus ruff — Expected: PASS.
- [ ] **Step 5: Commit** `Decode visibility and ceiling from SYNOP messages`.

### Task 3: Scores and bootstrap

**Files:**
- Create: `src/fogseg/scores.py`
- Test: `tests/test_scores.py`

**Interfaces:**
- Produces:
  - `COUNT_NAMES = ("hits", "misses", "false_alarms", "correct_negatives")`
  - `contingency(label: np.ndarray, truth: np.ndarray) -> np.ndarray` — int array of the 4 counts; only cases with
    `label in {0, 1}` and `truth in {0, 1}` count.
  - `skill(counts: np.ndarray) -> dict[str, float]` — keys `pod`, `far`, `csi`, `bias`; NaN where a denominator is 0.
    Works on the last axis, so `counts` of shape `(n, 4)` gives arrays of shape `(n,)`.
  - `counts_by_night(night: np.ndarray, label: np.ndarray, truth: np.ndarray, nights: np.ndarray) -> np.ndarray` —
    shape `(len(nights), 4)`, one row per night in the order of `nights`.
  - `bootstrap_skill(night_counts: np.ndarray, n_samples: int, seed: int) -> dict[str, tuple[float, float]]` —
    95 % interval per score; draws nights with replacement (`rng.integers`), sums the drawn rows.
  - `paired_csi_difference(counts_a: np.ndarray, counts_b: np.ndarray, n_samples: int, seed: int) -> tuple[float, float]`
    — 95 % interval of CSI(a) − CSI(b), both from the same drawn nights.

- [ ] **Step 1: Write the failing tests** in `tests/test_scores.py`:
  - `test_contingency_counts_and_skips_ignore`: labels `[1, 1, 0, 0, 255, 1]`, truth `[1, 0, 1, 0, 1, -1]` → `[1, 1, 1, 1]`.
  - `test_skill_values`: counts `[6, 2, 3, 89]` → pod 0.75, far 1/3, csi 6/11, bias 9/8.
  - `test_skill_without_fog_cases_is_nan`: counts `[0, 0, 0, 10]` → pod, far, csi, bias all NaN; no warning raised
    (`warnings.simplefilter("error")`).
  - `test_counts_by_night_keeps_night_order`: two nights, one empty → its row is zeros.
  - `test_bootstrap_is_reproducible_and_brackets_the_estimate`: same seed → same interval; interval contains the
    full-sample CSI.
  - `test_bootstrap_resamples_whole_nights`: one night with 3 hits, one with 3 misses → every drawn POD
    is 0, 0.5 or 1 (draw `n_samples = 200`, check the interval ends are in `{0, 0.5, 1}`).
  - `test_paired_difference_of_identical_counts_is_zero`: interval `(0.0, 0.0)`.

- [ ] **Step 2: Run** `pytest tests/test_scores.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement** `src/fogseg/scores.py`. Compute ratios with `np.divide(..., where=denominator > 0)` into a
  NaN-filled output, so empty groups give NaN without warnings.
- [ ] **Step 4: Run** tests plus ruff — Expected: PASS.
- [ ] **Step 5: Commit** `Add contingency scores and a bootstrap over nights`.

### Task 4: Truth rule, matching and case table

**Files:**
- Create: `src/fogseg/validation.py`
- Modify: `tests/helpers.py` (add `make_station_dataset`)
- Test: `tests/test_validation.py`

**Interfaces:**
- Consumes: `fogseg.labels.LabelParams`, `fogseg.label_store.LABEL_CHANNELS`, `fogseg.scores.*` (Task 3).
- Produces:
  - `TRUTH_YES, TRUTH_NO, TRUTH_UNDEFINED = 1, 0, -1`
  - `fls_truth(visibility_m: np.ndarray, ceiling_m: np.ndarray, limit_m: float) -> np.ndarray` (int8): yes if
    visibility < limit or ceiling < limit; no if both are known (inf counts as known) and ≥ limit; else undefined.
  - `fog_truth(visibility_m: np.ndarray, limit_m: float) -> np.ndarray` (int8).
  - `night_of(times: pd.Series | np.ndarray) -> np.ndarray` — date of the evening a scene belongs to
    (`time − 12 h`, as `datetime64[D]`).
  - `nearest_pixels(lon: np.ndarray, lat: np.ndarray, station_lon: np.ndarray, station_lat: np.ndarray) -> pd.DataFrame`
    — columns `iy`, `ix`, `distance_km` (haversine, earth radius 6371 km), one row per station.
  - `select_stations(stations: pd.DataFrame, lon, lat, max_distance_km: float, max_elevation_m: float) -> pd.DataFrame`
    — `stations` has `station, source, lat, lon, elevation_m`; adds `iy, ix, distance_km`; keeps stations within both
    limits (missing elevation is kept).
  - `match_observations(observations: pd.DataFrame, scene_times: np.ndarray, scan_delay_min: float, tolerance_min: float) -> pd.DataFrame`
    — one row per (station, scene) that has an observation within tolerance of `scene_time + scan_delay`; uses
    `pd.merge_asof(..., by="station", direction="nearest")`. Columns: all observation columns plus `scene_time`.
  - `deduplicate(cases: pd.DataFrame) -> pd.DataFrame` — one row per `(scene_time, iy, ix)`; METAR before SYNOP, then
    by station id.
  - `station_cases(data: xr.Dataset, labels: xr.Dataset, observations: pd.DataFrame, stations_cfg: dict, params: LabelParams) -> pd.DataFrame`
    — selects stations, matches observations to the label store's times, reads `label`, `IR_108`, `IR_120` at
    `(scene, iy, ix)` with vectorised `isel`, reads `threshold_source` per scene, and returns the case table with
    columns `scene_time, night, slot, station, source, lat, lon, iy, ix, visibility_m, ceiling_m, truth_fls,
    truth_fog, label, hidden, threshold_source`. `hidden = IR_108 < warm_min_bt108 or IR_108 − IR_120 ≥
    cirrus_max_split`; `slot` is `"HH:MM"` of the scene time.
- The observation table (Task 5 writes it) has columns `source, station, time, lat, lon, elevation_m, visibility_m,
  ceiling_m`.

- [ ] **Step 1: Write the failing tests** in `tests/test_validation.py`:
  - `test_fls_truth_edges`: visibility `[999, 1000, 5000, nan, 5000, nan]`, ceiling `[inf, inf, 999, 200, nan, inf]`
    → `[1, 0, 1, 1, -1, -1]`.
  - `test_fog_truth`: `[999, 1000, nan]` → `[1, 0, -1]`.
  - `test_night_of_morning_slot_belongs_to_evening`: `2023-11-21T03:00` → `2023-11-20`; `2023-11-20T21:00` → `2023-11-20`.
  - `test_nearest_pixel_and_distance`: a 3 × 3 grid with 0.1° spacing; station at a pixel centre → that pixel and
    distance 0; station 0.04° east → same pixel, distance ≈ 0.04 × 111.32 × cos(lat) km (rel. tol 1e-2).
  - `test_select_stations_drops_far_and_high`: one station 20 km outside the grid, one at 1500 m → both dropped.
  - `test_match_takes_closest_observation_after_scan_delay`: scene 05:00, observations 04:50, 05:20 → 05:20 is
    8 min from 05:12, chosen.
  - `test_match_outside_tolerance_gives_no_case`: only observation at 04:20 → no row.
  - `test_deduplicate_prefers_metar`: METAR and SYNOP on the same pixel and scene → METAR row kept; different pixels → both kept.
  - `test_station_cases_reads_labels_and_flags_hidden`: `make_station_dataset` (below) with labels written by
    `label_dataset(..., mode="fixed")`; one station on the fog patch, one on clear ground, one on a pixel made cold
    (IR_108 = 230 K) → labels 1, 0, 0 and `hidden` False, False, True.
- [ ] **Step 2: Add `make_station_dataset(path: Path) -> Path` to `tests/helpers.py`:** like `make_label_dataset`, plus
  `lon` / `lat` coordinates on `(y, x)` (0.1° spacing from 10 °E, 50 °N, north at the top) and one cold pixel
  (IR_108 230 K, IR_039 same as the background) at a fixed position, exported as `COLD_PIXEL = (iy, ix)`.
- [ ] **Step 3: Run** `pytest tests/test_validation.py -v` — Expected: FAIL.
- [ ] **Step 4: Implement** `src/fogseg/validation.py` (everything above except the score table, which is Task 6).
- [ ] **Step 5: Run** tests plus ruff — Expected: PASS.
- [ ] **Step 6: Commit** `Turn station observations into fog truth and match them to label pixels`.

### Task 5: Observation downloads and `fogseg stations`

**Files:**
- Create: `src/fogseg/observations.py`
- Modify: `configs/default.yaml` (new `stations:` section), `src/fogseg/cli.py` (`stations` command)
- Test: `tests/test_observations.py`

**Interfaces:**
- Consumes: `metar.prevailing_visibility_m`, `metar.ceiling_m` (Task 1), `synop.decode_synop` (Task 2),
  `validation.select_stations` (Task 4), `config.scene_times`.
- Produces:
  - `fetch_text(url: str, timeout: float = 120) -> str` — `urllib.request.urlopen`, decoded UTF-8.
  - `iem_stations(network: str) -> pd.DataFrame` — from `https://mesonet.agron.iastate.edu/geojson/network/{network}.geojson`;
    columns `station, source ("metar"), lat, lon, elevation_m`.
  - `wmo_stations(isd_history_csv: str) -> pd.DataFrame` — rows with `WBAN == "99999"`; WMO id = `USAF[:5]`; per id the
    row with the latest `END`; columns `station, source ("synop"), lat, lon, elevation_m` (from `ELEV(M)`).
  - `metar_url(station: str, start: dt.date, end: dt.date) -> str` — IEM `asos.py` with `data=metar`, `tz=Etc/UTC`,
    `format=onlycomma`, `latlon=no`, `missing=M`, `report_type=3&report_type=4`.
  - `synop_url(block: str, start: dt.datetime, end: dt.datetime) -> str` — `https://www.ogimet.com/cgi-bin/getsynop?begin=YYYYmmddHHMM&end=...&block=..`.
  - `download(url: str, path: Path, pause_s: float, max_tries: int, is_valid: Callable[[str], bool]) -> bool` — skips if
    `path` exists; otherwise tries up to `max_tries` times, sleeping `pause_s` before each try and `pause_s * 6` after
    an invalid answer; writes `path` only for a valid answer (temp name, then rename); returns whether `path` exists.
  - `parse_metar_csv(text: str) -> pd.DataFrame` and `parse_synop_text(text: str) -> pd.DataFrame` — observation
    columns `source, station, time, visibility_m, ceiling_m` (coordinates are joined later from the station table).
  - `night_windows(cfg: dict) -> list[tuple[dt.datetime, dt.datetime]]` — 20:00 on each dataset night to 06:00 next day.
  - `collect_observations(source: str, cfg: dict, lon: np.ndarray, lat: np.ndarray) -> pd.DataFrame` — station list,
    selection, downloads, parsing, join of `lat, lon, elevation_m`, filter to `±1 h` of a scene time; returns the
    observation table of Task 4.
- Config (`configs/default.yaml`), exact keys:

```yaml
# Station observations that check and tune the labels; design in docs/superpowers/specs/2026-10-28-station-validation-design.md
stations:
  data_dir: "/mnt/d/fogseg-data/stations"
  iem_networks: [AT__ASOS, BA__ASOS, BE__ASOS, BY__ASOS, CH__ASOS, CZ__ASOS, DE__ASOS, DK__ASOS, FR__ASOS, HR__ASOS,
                 HU__ASOS, IT__ASOS, LT__ASOS, LU__ASOS, NL__ASOS, PL__ASOS, RO__ASOS, RS__ASOS, SE__ASOS,
                 SI__ASOS, SK__ASOS, UA__ASOS]
  wmo_blocks: ["02", "06", "07", "10", "11", "12", "13", "14", "15", "16", "26", "33"]
  isd_history_url: "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
  iem_pause_s: 5
  ogimet_pause_s: 22   # Ogimet allows one query per remote IP every 20 s
  max_tries: 3
  max_distance_km: 5.0
  max_elevation_m: 1000.0
  truth_limit_m: 1000.0
  scan_delay_min: 12
  time_tolerance_min: 30
  lat_bands: [43, 47, 51, 56]
  tune_season: "2023-24"
  offset_grid: [0.5, 3.0, 0.25]      # start, stop (included), step in K
  btd_fixed_grid: [4.0, 7.0, 0.25]
  bootstrap_samples: 1000
  seed: 0
```

  The implementer checks once (by hand, not in tests) that every network name returns stations; a name that does not
  exist is removed from the list.
- CLI: `fogseg stations [--source metar|synop]` (both when omitted). Writes `<data_dir>/observations_<source>.csv` and
  logs stations used, observations kept, downloads failed. Raw files: `<data_dir>/raw/metar/<station>_<season>.csv`,
  `<data_dir>/raw/synop/<block>_<YYYYmmdd>.txt`. Ogimet answers are valid when they contain `AAXX`; IEM answers when
  their first line starts with `station,`.

- [ ] **Step 1: Write the failing tests** in `tests/test_observations.py` (monkeypatch `fogseg.observations.fetch_text`
  and `time.sleep`):
  - `test_parse_metar_csv`: two IEM rows (header `station,valid,metar`) → visibility and ceiling decoded, `time` is
    `datetime64`, `source == "metar"`.
  - `test_parse_synop_text`: two Ogimet lines, one `NIL` → one row, `station` and `time` from the leading columns.
  - `test_wmo_stations_takes_latest_record`: two rows for the same USAF with different `END` → the later one; a row with
    another WBAN is dropped.
  - `test_download_skips_existing_file`: `fetch_text` must not be called.
  - `test_download_does_not_save_empty_answer`: `fetch_text` returns `""` every time → returns False, no file, called
    `max_tries` times.
  - `test_download_retries_then_saves`: first `""`, then valid → file written, returns True.
  - `test_night_windows`: config with one season of two nights, `every_nth_night: 2` → one window per dataset night,
    20:00 to 06:00.
- [ ] **Step 2: Run** `pytest tests/test_observations.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement** `observations.py`, the config section and `cmd_stations` in `cli.py` (same pattern as
  `cmd_label`; heavy imports inside the function).
- [ ] **Step 4: Run** `pytest -q` plus ruff — Expected: all PASS (the existing label tests read `configs/default.yaml`;
  they must still pass with the new section).
- [ ] **Step 5: Commit** `Download METAR and SYNOP observations for the dataset nights`.

### Task 6: Score table and `fogseg validate`

**Files:**
- Modify: `src/fogseg/validation.py`, `src/fogseg/cli.py`
- Test: `tests/test_validation.py`

**Interfaces:**
- Consumes: Task 3 and Task 4 functions; `label_store.label_store_path`, `label_store.check_labels_match`,
  `label_store.stored_params`.
- Produces:
  - `season_bounds(cfg: dict, season: str) -> tuple[np.datetime64, np.datetime64]` — `[start 12:00, end + 1 day 12:00)`;
    unknown name → `ValueError` listing the names.
  - `check_season_allowed(cfg: dict, season: str, final: bool) -> None` — raises `PermissionError` when
    `season != cfg["stations"]["tune_season"]` and not `final`, with a message naming the spec.
  - `score_table(cases: pd.DataFrame, lat_bands: list[float], n_samples: int, seed: int) -> pd.DataFrame` — one row
    per view; columns `view, group, n_cases, hits, misses, false_alarms, correct_negatives, pod, pod_low, pod_high,
    far, far_low, far_high, csi, csi_low, csi_high, bias, n_fls_ignored, n_no_fls_ignored, n_fls_hidden`.
- Views (rows), all on deduplicated cases unless named per source:

| view | group | cases |
|---|---|---|
| `main` | `all` | pooled, FLS truth, hidden left out |
| `source` | `metar` / `synop` | each source deduplicated on its own, FLS, hidden left out |
| `threshold_source` | `dynamic` / `fallback` / `fixed` | pooled, FLS, hidden left out |
| `lat_band` | e.g. `43-47` | pooled, FLS, hidden left out |
| `slot` | `21:00` … | pooled, FLS, hidden left out |
| `fog_only` | `all` | pooled, fog truth, hidden left out |
| `with_hidden` | `all` | pooled, FLS, hidden kept |

  `n_fls_ignored` / `n_no_fls_ignored` count cases of the group with label 255; `n_fls_hidden` counts FLS cases of the
  group that are hidden (both before the label filter).
- CLI: `fogseg validate [--labels fixed|dynamic] --season NAME [--final]`. Reads `observations_metar.csv` and
  `observations_synop.csv` (missing file → `FileNotFoundError` naming `fogseg stations`; one source missing is allowed
  with a warning), checks the season, keeps the store's scenes inside the season, writes
  `<data_dir>/cases_<mode>_<season>.csv` and `<data_dir>/scores_<mode>_<season>.csv`, logs the `main` row.

- [ ] **Step 1: Write the failing tests:**
  - `test_check_season_refuses_test_season`: `"2024-25"` without final → `PermissionError`; with final → no error;
    `"2023-24"` → no error.
  - `test_season_bounds_unknown_name`: `"1999-00"` → `ValueError` mentioning `2023-24`.
  - `test_score_table_views`: a hand-made case table (2 nights, both sources, one hidden FLS case, one label-255 FLS
    case) → `main` row counts equal a hand count; `with_hidden` has one more miss; `n_fls_hidden == 1` and
    `n_fls_ignored == 1` in `main`.
  - `test_score_table_empty_group_is_nan`: a latitude band with no cases → row present, `n_cases == 0`, `csi` NaN.
- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement** the three functions and `cmd_validate`.
- [ ] **Step 4: Run** `pytest -q` plus ruff — Expected: PASS.
- [ ] **Step 5: Commit** `Score the label stores against station observations`.

### Task 7: Threshold tuning and `fogseg tune`

**Files:**
- Create: `src/fogseg/tuning.py`
- Modify: `src/fogseg/cli.py`
- Test: `tests/test_tuning.py`

**Interfaces:**
- Consumes: `labels.label_scene`, `labels.histogram_pixels`, `dynamic_threshold.clear_ground_peak`, Task 3 scores,
  Task 4 `station_cases` output (the case table).
- Produces:
  - `grid_values(spec: list[float]) -> np.ndarray` — `[start, stop, step]` with `stop` included, rounded to 1e-6.
  - `scene_threshold(peak: float, offset: float, fallback: float, clamp: tuple[float, float]) -> float` — clipped
    `peak + offset`, or `fallback` when `peak` is NaN.
  - `grid_labels(data: xr.Dataset, cases: pd.DataFrame, params: LabelParams, dynamic: DynamicParams, parameter: str, values: np.ndarray, mode: str) -> pd.DataFrame`
    — for every scene that has cases: read the three channels and `sza` once, compute the peak once
    (`clear_ground_peak` on `histogram_pixels`, NaN when fewer than `min_pixels` — the same rule as
    `dynamic_threshold`), then for each value label the whole scene with `label_scene` and read the labels at the case
    pixels. `parameter` is `"offset"` or `"btd_fixed"`; `mode` is `"dynamic"` or `"fixed"` (fixed: threshold = value).
    Returns the case rows repeated per value with columns `value, night, label, truth_fls`.
  - `@dataclass(frozen=True) class Decision: parameter: str; current: float; best: float; csi_current: float;
    csi_best: float; difference_low: float; difference_high: float` with property `changed -> bool`
    (`best != current and difference_low > 0`).
  - `decide(grid: pd.DataFrame, parameter: str, current: float, n_samples: int, seed: int) -> tuple[pd.DataFrame, Decision]`
    — per value: counts, scores and 95 % interval (the grid score table); best = highest CSI, ties → closest to
    `current`; paired interval of CSI(best) − CSI(current).
- Reuse the scene-peak code path of `dynamic_threshold` instead of copying it: if `dynamic_threshold` needs a small
  refactor to expose "peak or NaN" for this, do it there and keep its tests green.
- CLI: `fogseg tune`. Season = `stations.tune_season` (no option). Cases: pooled, deduplicated, FLS truth defined,
  hidden left out. Order: `offset` grid in dynamic mode (fallback = current `btd_fixed`); then `btd_fixed` grid in
  dynamic mode with the chosen offset (decided value, or current when unchanged); then `btd_fixed` grid in fixed mode,
  reported only. Writes `<data_dir>/tuning_offset.csv`, `tuning_btd_fixed.csv`, `tuning_btd_fixed_fixed_mode.csv` and
  logs each `Decision`. Does not edit the config.

- [ ] **Step 1: Write the failing tests** in `tests/test_tuning.py`:
  - `test_grid_values_includes_stop`: `[0.5, 1.0, 0.25]` → `[0.5, 0.75, 1.0]`.
  - `test_scene_threshold_clamps_and_falls_back`.
  - `test_grid_finds_planted_threshold`: synthetic dataset (helpers) with a clear background whose clear-ground peak
    gives a known value and a fog patch; station truth "yes" on the patch, "no" on clear ground and on a mid-BTD strip
    → the best `offset` is the one that labels the patch but not the strip; `Decision.best` equals the planted value.
    Use `min_pixels` small enough for the tiny grid (a `DynamicParams` built in the test).
  - `test_decide_keeps_current_without_evidence`: two values with identical counts → `changed` is False.
  - `test_decide_changes_with_clear_evidence`: 20 nights where the best value has hits and the current only misses →
    `changed` is True.
- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement** `tuning.py` and `cmd_tune`.
- [ ] **Step 4: Run** `pytest -q` plus ruff — Expected: PASS.
- [ ] **Step 5: Commit** `Tune the BTD offset and the fallback threshold on training stations`.

### Task 8: Real-data run (controller, not a subagent)

- [ ] `fogseg stations --source metar` then `--source synop`, both in the background (SYNOP takes hours); check the
  logs for failed downloads and rerun until none are left or the rest is explained.
- [ ] `fogseg validate --labels dynamic --season 2023-24` and `--labels fixed --season 2023-24`.
- [ ] `fogseg tune`; read the decisions. Change `configs/default.yaml` only where `Decision.changed` is True, with a
  comment pointing to the notes.
- [ ] If anything changed: `fogseg label --mode dynamic --overwrite` and `--mode fixed --overwrite`, then validate the
  training season again.
- [ ] Freeze. Then, once: `fogseg validate --labels dynamic --season 2024-25 --final` (and fixed).
- [ ] Look at numbers that seem off (for example very low POD at one station) before writing them up; sanity-check a
  few cases against their raw METAR.

### Task 9: Documentation and PR

**Files:**
- Create: `docs/notes/station_validation.md`
- Modify: `README.md`, `docs/DATASET_CARD.md`, `configs/default.yaml` (comments), `docs/notes/label_check.md`
  (the fallback limitation points to the new note)

- [ ] Notes page, in the style of `label_check.md`: data and stations used, truth rule with references, matching,
  the score tables (main, per source, per threshold source, per latitude band, per slot, fog only, with hidden), the
  tuning grids and each decision with its interval, the three failure analyses of spec §7 (fallback decks, north-south
  drift, fog under high cloud) plus ignored fog, benchmarks for comparison, and the final test-season scores marked as
  the one final evaluation.
- [ ] README: a short "Validation against stations" section with the headline scores and a link to the note.
- [ ] Dataset card: station validation and its limits.
- [ ] `pytest -q`, `ruff check .`, `ruff format --check .` — all clean.
- [ ] Commit, push the branch, open the PR into `develop` (PR #3 is merged) with `~/.local/bin/gh` in WSL.
