# Design: better fog labels from the station network

Status: draft for review · Date: 2026-10-31 · Branches: `feature/better-labels-evaluation` (phase A) →
`feature/better-labels-rule` (B) → `feature/better-labels-teacher` (C), each a PR into the one before; phase A goes
into `feature/station-validation` until PR #5 is merged into `develop`

## 1. Goal

Raise the number of station fog / low-stratus (FLS) reports that the labels also call fog, without buying the hits
with false alarms, before a U-Net is trained on the labels.

The station check (`docs/notes/station_validation.md`) scored the fixed 4.0 K labels at POD 0.50, FAR 0.28, CSI 0.42
on the visible 2023-24 cases. A first diagnostic of the misses (all 2023-24 cases, descriptive only; it moves into
`docs/notes/better_labels.md` in phase A) showed:

- Half of the 18,600 missed visible FLS reports have a BTD ≤ 3.5 K, like clear ground. BTD alone saturates near
  CSI 0.44 at any threshold, so new information is needed, not a new threshold.
- 27 % of the misses lie in 3.5–4.5 K (mostly the ±0.5 K ignore band), 17 % fail the smoothness test first.
- 85 % of the missed FLS are low-ceiling reports; 36 % of the misses are labelled fog at the same pixel in another
  slot of the same night.
- Counting ignored (255) FLS reports as misses, the labels score POD 0.40, FAR 0.28, CSI 0.35.

The 2024-25 test season stays sealed: no fit, choice or threshold uses it. It is opened once, at the end of phase C,
for the frozen baseline and the final labels, and only after the project owner agrees.

## 2. Decisions taken in the brainstorming

| Topic | Decision | Reason |
|---|---|---|
| Scope | Steps 1–8 of the research plan: evaluation protocol, matching fixes, diagnosis, rule repair, time, station-trained teacher, local correction, obscured class | Owner's choice, 2026-10-31 |
| Not in scope | Station point loss and the U-Net itself | They belong to the U-Net step |
| Split | Three phases, one plan and one PR each, stacked | Each phase is reviewable on its own; B and C build on A's protocol |
| Main score | **Full-count POD at FAR ≤ 0.28**: an ignored (255) FLS report counts as a miss | Abstaining cannot raise it; "more hits at today's false-alarm level" is the goal |
| Keep rule | A variant replaces the current one only if the paired bootstrap interval of its POD gain excludes 0 and its FAR is not higher | No change without evidence |
| ERA5 | Downloaded with the owner's own CDS key (`~/.cdsapirc`) | Temperature contrast and snow are the strongest candidates for the low-BTD misses |
| DEM | Copernicus GLO-90 from the public AWS bucket | No key needed |
| Model library | scikit-learn (logistic regression, `HistGradientBoostingClassifier`, isotonic calibration) | Enough for ~50,000 tabular cases; no GPU, one well-known dependency |
| Hidden fog | New label value 2 "obscured" instead of 0 | Fog under high cloud is unknown, not clear; the U-Net step decides whether to train or ignore it |

The source review behind these decisions is `docs/notes/label_improvement_research.md` and the research report
summarized in `docs/notes/better_labels.md`.

## 3. Evaluation protocol (phase A, step 1)

Everything is measured inside the 2023-24 season.

### 3.1 Folds

`fogseg split` writes `stations/splits_2023-24.csv` once; later runs read it and never redraw it.

- **Night blocks**: the season's nights in consecutive two-week blocks (about 11). Blocks are dealt to folds 1–5 in
  turn (block *i* → fold *i* mod 5 + 1), so every fold holds early and late winter.
- **Station blocks**: 2° × 2° cells over the region bbox. Cells are dealt to folds 1–5 at random with
  `stations.seed`. A station belongs to the cell of its coordinates.
- Columns: `kind` (`night_block` or `station_cell`), `key` (block start date or cell id), `fold`.

### 3.2 Held-out and selection sets

- **Held-out**: every case whose night or station is in fold 5 (about 36 % of cases). It is never used to fit,
  choose or calibrate anything. Final comparisons are reported on it in three parts: unseen nights only, unseen
  stations only, both.
- **Selection**: nights and stations both in folds 1–4. A blocked four-fold cross-validation runs on it: fold *k*
  trains on cases with night fold ≠ *k* and station fold ≠ *k* and tests on cases with night fold = *k* or station
  fold = *k*. Every threshold, variant and teacher setting is chosen here.

### 3.3 Scores

- Visible cases only (`hidden` false), truth defined, deduplicated as today.
- **Full count**: label 1 is "yes"; labels 0, 2 and 255 are "no". POD, FAR, CSI, bias and HSS come from this table.
  The current decided-only scores stay in the table as a secondary row.
- **Matched FAR**: a variant with a free threshold gets the threshold whose cross-validated FAR is closest to
  `stations.matched_far` (0.28) from below, on `stations.btd_fixed_grid` for the BTD and `stations.emissivity_grid`
  for the emissivity index.
- **Paired bootstrap** over night blocks (`stations.bootstrap_samples` draws): interval of the POD difference and of
  the FAR difference between a variant and its reference, on the same cases.
- **Limits**: at most 10 variants per phase are cross-validated; only the cross-validation winner of a phase is
  scored on the held-out set, once.
- **Baseline**: the fixed 4.0 K rule of PR #5, re-scored with this protocol and with the matching of Section 4.1.

### 3.4 Code

| Module | Job |
|---|---|
| `splits.py` | Night blocks, station cells, fold assignment, `held_out` / `selection` masks, CV fold iterator |
| `scores.py` | Gains `full_count` contingency and `paired_difference` for any score (generalizes `paired_csi_difference`) |

## 4. Phase A: matching, data, diagnosis, obscured class

### 4.1 Matching fixes (step 2)

Applied to the baseline and every variant alike, so they move all numbers together and are not label progress.

- **Parallax**: each station looks up the pixel where a cloud top at station elevation + `parallax_cloud_top_m`
  (default 500 m) appears from the satellite at 0° E. The shift is about 1–2 km north and sometimes changes the pixel.
  Implemented in `validation.nearest_pixels` via a shifted station position from `parallax.py`.
- **New case-table columns, used as fixed strata in the score table**:
  - `auto`: METAR with the `AUTO` group;
  - `ceiling_amount`: `BKN`, `OVC`, `VV` or empty, re-parsed from the cached raw METARs (`metar.py`);
  - `shallow_fog`: `MIFG` or `BCFG` present;
  - `elevation_diff_m`: station elevation minus the pixel-mean DEM elevation.
  SYNOP rows leave the METAR-only columns empty. Truth definition and the 1000 m limit do not change.

### 4.2 Data downloads

| Command | Source | Output | Notes |
|---|---|---|---|
| `fogseg era5` | Copernicus CDS, ERA5 single levels, with `~/.cdsapirc` | `processed/era5_ceu.zarr` on the SEVIRI grid, one field per scene time | 2 m temperature, skin temperature, 2 m dewpoint, snow depth; both seasons (C labels 2024-25 without fitting on it); bilinear to the grid; one request per month, resumable |
| `fogseg dem` | Copernicus GLO-90 DEM, public AWS bucket `copernicus-dem-90m` | `processed/dem_ceu.zarr`: pixel-mean elevation and std | Tiles cached under `raw/dem`, block-averaged onto the grid |

The code never reads or prints the CDS key; a missing `~/.cdsapirc` stops `fogseg era5` with a message that points
to the CDS setup page.

### 4.3 Diagnosis (step 3)

`fogseg diagnose` uses selection cases only and writes `stations/diagnosis_2023-24.csv` plus a log summary:

- visible FLS misses by first cause: twilight, BTD ≤ 3.5 K, BTD 3.5–4.5 K, fails texture, small blob;
- for the low-BTD misses and for the no-FLS cases at the same BTD, side by side: `ceiling_amount`, `auto`, slot, fog
  at the pixel in another slot of the night, IR10.8 − WV6.2, ERA5 T2m − IR10.8, `elevation_diff_m`.

It decides nothing; its numbers go into `docs/notes/better_labels.md` and size phases B and C.

### 4.4 Obscured class (step 8)

- `labels.OBSCURED = 2`: a trusted night pixel that fails the warm-top or the cirrus test. It is set before the fog
  tests, so it never overlaps 1 or 255.
- `label_store` fractions gain `obscured_fraction`; `validate` and the dataset card describe value 2.
- Scores: visible views drop hidden cases as today; the `with_hidden` view treats 2 as "no".

## 5. Phase B: rule repair and time (steps 4, 5)

### 5.1 New label options

New `LabelParams` fields; the defaults reproduce today's labels bit for bit.

| Field | Values | Meaning |
|---|---|---|
| `texture` | `ir108_std3` (default), `btd_std3`, `ir108_mad5`, `blob`, `none` | Spread of IR10.8 in 3×3; of the BTD in 3×3; median absolute deviation of IR10.8 in 5×5; median 3×3 IR10.8 spread per connected fog blob; no test |
| `fog_index` | `btd` (default), `emissivity` | BTD > threshold, or 3.9 µm pseudo-emissivity ε = B₃.₉(BT₃.₉) / B₃.₉(BT₁₀.₈) < threshold (Planck at 3.92 µm) |
| `split_window` | `fixed` (default), `kma` | IR10.8 − IR12.0 < `cirrus_max_split`, or within ±1 K of −37.48 + 0.1329 · IR10.8 |
| `temporal_relax` | K, default 0 | A pixel whose index misses the threshold by less than this becomes fog if the same pixel is fog at the strict threshold in an adjacent slot of the same night and IR10.8 changed by less than `temporal_max_dt108` |

Temporal hysteresis is applied once (no iteration) after the per-scene labels of a night are made. Adjacent slots
are the neighbours in the night's slot list (21:00, 00:00, 03:00, 05:00) that exist in the store.

### 5.2 Variants

1. baseline
2. `texture: btd_std3`
3. `texture: ir108_mad5`
4. `texture: blob`
5. `texture: none`
6. `uncertainty_band: 0.25`
7. `fog_index: emissivity`
8. `split_window: kma`
9. the best texture of 2–5 with the better index of 1 and 7
10. variant 9 with `temporal_relax` (value from a small grid inside the variant)

Each variant's threshold is chosen at matched FAR (Section 3.3). `fogseg experiment --phase B` labels the selection
scenes per variant, cross-validates and writes `stations/experiment_B.csv`; `--held-out` scores the winner and the
baseline once. If the winner passes the keep rule, its values go into `default.yaml` and the store is relabelled.

## 6. Phase C: station-trained teacher and local correction (steps 6, 7)

### 6.1 Features (`features.py`)

One function builds the feature stack for whole scenes, so training at station pixels and dense prediction share it:
the 8 brightness temperatures; 10.8−3.9, 10.8−12.0, 8.7−10.8, 12.0−13.4, 10.8−6.2 and ε₃.₉; the 3×3 spread of IR10.8
and of the BTD; satellite zenith and latitude; DEM mean and std; ERA5 T2m − IR10.8, skin − IR10.8, T2m − dewpoint and
snow depth; slot, the night maximum of the BTD at the pixel and the IR10.8 change from the previous slot; and the
phase B rule label.

### 6.2 Training and threshold (`teacher.py`)

- Training cases: visible selection cases with defined truth.
- Models: logistic regression on standardized features, then `HistGradientBoostingClassifier`; at most 10 settings in
  all, chosen on the blocked cross-validation.
- Out-of-fold probabilities are calibrated with isotonic regression. The threshold p* is where the out-of-fold FAR is
  0.28; pixels with |p − p*| < `teacher.ignore_halfwidth` (0.1) get 255 and count as misses in the main score.
- The final model is fitted on all selection cases and saved with its feature list to `processed/teacher.joblib`.
- It is scored once on the held-out set against the phase B rule.

### 6.3 Dense labels

`fogseg label --mode teacher` writes `labels_teacher.zarr`: `label` (0 / 1 / 2 / 255, untrusted and obscured pixels as
in the rule) and `fog_probability` (uint8, 0–250 for p × 250, 255 where undefined). If the teacher passes the keep rule
against the phase B rule, `labels.mode: teacher` becomes the default; otherwise it stays a documented benchmark.

### 6.4 Local correction

Per scene, the residuals truth − p at selection stations are spread with inverse-distance weights
w = exp(−d / L) within 3 L and added to p, which is then re-thresholded at p*. Three length scales L are tried on the
cross-validation. The correction is kept only if it beats the plain teacher on the held-out stations. Held-out
stations never feed it, so they stay clean for the U-Net evaluation later.

## 7. Configuration

New keys in `configs/default.yaml` (defaults keep today's behaviour until a phase changes them):

```yaml
labels:
  texture: ir108_std3
  fog_index: btd
  emissivity_max: null      # set by phase B if the emissivity index wins
  split_window: fixed
  temporal_relax: 0.0
  temporal_max_dt108: 2.0
stations:
  parallax_cloud_top_m: 500.0
  night_block_days: 14
  station_cell_deg: 2.0
  n_folds: 5
  matched_far: 0.28
  emissivity_grid: [0.80, 1.00, 0.01]
era5:
  out_path: "/mnt/d/fogseg-data/processed/era5_ceu.zarr"
  variables: [2m_temperature, skin_temperature, 2m_dewpoint_temperature, snow_depth]
dem:
  raw_dir: "/mnt/d/fogseg-data/raw/dem"
  out_path: "/mnt/d/fogseg-data/processed/dem_ceu.zarr"
teacher:
  model_path: "/mnt/d/fogseg-data/processed/teacher.joblib"
  ignore_halfwidth: 0.1
  idw_length_km: [25, 50, 100]
```

## 8. Error handling

- `fogseg split` refuses to overwrite an existing split file without `--overwrite`.
- Every command that fits or chooses something reads the split file and fails if it is missing.
- `check_season_allowed` keeps guarding 2024-25; `fogseg label --mode teacher` may label 2024-25 scenes (prediction
  only), but `experiment` and `diagnose` refuse any season but the tuning season.
- ERA5 and DEM downloads are resumable and skip finished months / tiles; a scene time missing in ERA5 makes the
  teacher label that scene 255.
- A store made with other label options than the config is detected through its attributes, as today.

## 9. Testing

Synthetic arrays and tables only, so CI needs no data or network:

- `splits`: block and cell assignment, disjoint held-out and selection, CV folds never train on their test blocks.
- `scores`: full-count table, matched-FAR threshold choice, paired difference on hand-made counts.
- `parallax`: shift direction and size against hand-computed geometry.
- `metar`: `AUTO`, cover amount and shallow-fog parsing.
- `labels`: each new option on small arrays, defaults reproduce the old labels, obscured class placement, temporal
  hysteresis.
- `features` and `teacher`: feature stack shape and values, threshold at the target FAR, ignore band, IDW correction.
- ERA5 and DEM: regridding and caching with fake downloaded files; no network call in tests.
- `ruff check` and `ruff format --check` clean; CI green on every PR.

## 10. Documentation

- `docs/notes/better_labels.md`: the miss diagnostic, the protocol, and per phase the variants, cross-validation and
  held-out tables. Every held-out table also shows the phase A baseline, so the total gain over the fixed 4.0 K rule
  is visible in one place.
- `docs/DATASET_CARD.md`: label value 2, label options, teacher labels and `fog_probability`, ERA5 and DEM sources and
  licences.
- `README.md`: new commands.
- The station notes link to the new page.

## 11. Order of work

- Phase A: split and scores → parallax and new case columns → obscured class → DEM → ERA5 → diagnose → re-scored
  baseline and notes → PR.
- Phase B: label options → experiment command → cross-validation run → held-out run → config and relabel → notes → PR.
- Phase C: features → teacher and calibration → dense labels → local correction → held-out run → notes → PR → ask the
  owner about the one final 2024-25 evaluation.

## 12. Risks

- **Small held-out set.** About 15 held-out nights; the bootstrap interval may be too wide to show a gain of 0.03.
  The interval is reported as it is; a gain that does not clear it is not claimed.
- **Low-BTD misses may be unrecoverable.** If the diagnosis shows they are mostly partial cover or valley stations,
  phases B and C will gain little; that result is reported, not hidden.
- **Teacher overfits stations.** Mitigated by the station-cell folds; the unseen-stations part of the held-out score
  is the one that counts.
- **ERA5 at 0.25° is coarse** for valley fog; it is a feature, not truth.
- **Selection on the held-out set.** Each phase opens the held-out set once for one winner; three phases mean three
  looks, which is stated in the notes.
