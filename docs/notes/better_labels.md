# Better fog labels from the station network

Phases A, B and C of [the design](../superpowers/specs/2026-10-31-better-labels-design.md). Phase A built an evaluation
protocol that can tell a real gain from noise, cleaner matching between stations and pixels, an `obscured` label, the
terrain and ERA5 data the later phases need, and a diagnosis of the missed fog. Phase B tried ten repairs of the label
rule against that protocol ([below](#phase-b-rule-repair-and-time)); none passed the keep rule, so the labels are still
the fixed 4.0 K baseline. Phase C trained a teacher on the station truth
([below](#phase-c-station-trained-teacher)): it finds far more fog, but the FAR clause of the keep rule stopped it, so
the configured labels stay as they were.

Everything is measured on the 2023-24 training season. The 2024-25 test season was not opened.

## Protocol

`fogseg split` writes `stations/splits_2023-24.csv` once:

- **Night blocks:** 14-day blocks of the season's nights, dealt to folds 1–5 in turn (block *i* → fold *i* mod 5 + 1),
  so every fold holds early and late winter.
- **Station cells:** 2° × 2° cells from the south-west corner of the region, dealt to folds 1–5 in a random order fixed
  by `stations.seed`.
- **Held-out:** every case whose night or station is in fold 5, 36 % of the cases. From now on nothing is chosen on
  it, and each phase opens it once, for its one winner (phase C was the last). It is reported in three parts: unseen nights only, unseen
  stations only, both.
- **Selection:** nights and stations both in folds 1–4. Every threshold and variant of phases B and C is chosen here,
  with a blocked four-fold cross-validation. `fogseg tune` and `fogseg diagnose` read only these cases and refuse to
  run without the split file.

One caveat: the baseline's 4.0 K was chosen in PR #5 on all 2023-24 cases, before the split existed, so the held-out
cases have seen that choice. This favours the baseline, which makes a later gain over it harder to show, not easier.

The main score is the **full count**: label `1` is "fog", and `0`, `2` and `255` are all "no fog". An ignored pixel
under a fog report is then a miss, which the old decided-only count did not see. Phases B and C compare at the same
FAR, 0.28 (`stations.matched_far`), and report the POD gain with a paired bootstrap over the 14-day night blocks.

## Matching fixes

These change how a station finds its pixel. They move the baseline and every later variant alike, so they are not label
progress.

- **Parallax:** a station now looks up the pixel where a cloud top 500 m above the station appears from the satellite
  at 0° E, about 1.6 km north of the station at 50° N.
- **Terrain:** `elevation_diff_m` is the station elevation minus the mean Copernicus GLO-90 elevation of its pixel.
- **New METAR columns**, re-parsed from the cached reports: `auto` (an automatic station), `ceiling_amount` (`BKN`,
  `OVC` or `VV` of the lowest ceiling) and `shallow_fog` (`MIFG` or `BCFG`).

The parallax shift barely moves the scores: POD 0.504, FAR 0.278, CSI 0.422 against 0.50, 0.28 and 0.42 before. The
shift is a fraction of a pixel.

## Baseline: fixed 4.0 K

The rule of PR #5, re-scored with this protocol. Visible cases only (the pixel passes the warm-top and cirrus tests),
one case per pixel and scene; 95 % intervals from a bootstrap over nights. "Cases" is the full count; the decided-only
POD, FAR and CSI leave out the cases labelled `255`.

| Set | Cases | POD | FAR | CSI | Full-count POD | Full-count FAR | Full-count CSI |
|---|---|---|---|---|---|---|---|
| all | 75,680 | 0.50 | 0.28 | 0.42 | 0.40 (0.35–0.46) | 0.28 (0.24–0.32) | 0.35 |
| selection | 48,669 | 0.49 | 0.29 | 0.41 | 0.39 (0.33–0.44) | 0.29 (0.24–0.36) | 0.33 |
| **held-out** | 27,011 | 0.52 | 0.25 | 0.44 | **0.43 (0.36–0.50)** | **0.25 (0.20–0.31)** | 0.38 |
| held-out, unseen nights only | 9,480 | 0.56 | 0.27 | 0.46 | 0.48 (0.38–0.57) | 0.27 (0.19–0.36) | 0.41 |
| held-out, unseen stations only | 14,603 | 0.46 | 0.25 | 0.40 | 0.37 (0.30–0.43) | 0.25 (0.19–0.31) | 0.33 |
| held-out, both | 2,928 | 0.61 | 0.23 | 0.52 | 0.54 (0.41–0.64) | 0.23 (0.17–0.32) | 0.46 |

The held-out full-count POD interval is 0.14 wide. A later phase has to gain more than a few hundredths before the
paired interval clears zero; a smaller gain is reported but not claimed.

Counted with the hidden cases (fog under a pixel that fails the high-cloud tests, now label `2`), POD falls to 0.20:
51 % of the fog and low-stratus reports lie under such a pixel. No IR rule can see them, so the visible cases remain
the main score.

### Strata

Full-count scores per fixed stratum, all cases:

| Stratum | Group | Cases | POD | FAR |
|---|---|---|---|---|
| METAR station | automatic | 18,564 | 0.40 | 0.29 |
| | no `AUTO` group | 11,547 | 0.45 | 0.36 |
| Lowest ceiling (FLS by ceiling only) | `BKN` | 5,637 | 0.37 | – |
| | `OVC` | 3,891 | 0.52 | – |
| | `VV` | 495 | 0.46 | – |
| Shallow or patchy fog reported | `MIFG` / `BCFG` | 455 | 0.36 | 0.51 |
| Station minus pixel terrain | below −100 m | 2,756 | 0.30 | 0.32 |
| | −100 to 100 m | 53,164 | 0.41 | 0.28 |
| | 100 m and more | 2,046 | 0.36 | 0.19 |

- **Broken decks are missed more than overcast ones:** POD 0.37 for `BKN` against 0.52 for `OVC`. A broken deck fills
  part of a 3 km pixel, so its BTD and texture mix cloud and ground.
- **Valley stations** (more than 100 m below their pixel's mean terrain) score worst; the pixel averages the valley
  with the slopes around it. They are 5 % of the cases.
- **Shallow fog** is half false alarms by the full count; a fog layer a few metres deep is a poor match for a pixel
  label either way.

## Why visible fog is missed

`fogseg diagnose` on the selection cases, one case per pixel and scene as in the score table (11,727 misses). A miss
is a visible fog or low-stratus report whose pixel is not labelled `1`; its first cause, in this order:

| First cause | Rule | Share of misses |
|---|---|---|
| twilight | solar zenith angle 95° or less | 5 % |
| low BTD | BTD ≤ 3.5 K | **49 %** |
| band | BTD below 4.5 K, smooth top | 28 % |
| texture | 3 × 3 standard deviation of IR10.8 at least 1.5 K | 15 % |
| small blob or other | | 2 % |

Half of the misses have a BTD of 3.5 K or less, where clear ground sits. For those, the missed fog and the cloud-free
reports at the same BTD (selection cases, night) compare as follows:

| Variable | Missed fog, BTD ≤ 3.5 K | No fog, BTD ≤ 3.5 K |
|---|---|---|
| lowest ceiling `BKN` | 26 % | 8 % |
| lowest ceiling `OVC` | 12 % | 2 % |
| lowest ceiling `VV` | 3 % | 0 % |
| lowest ceiling unknown (SYNOP, or no ceiling) | 59 % | 89 % |
| fog at the same pixel in another slot of the night | 26 % | 10 % |
| IR10.8 − WV6.2, median (middle half) | 33 K (30–37) | 37 K (32–42) |
| station minus pixel terrain, median (10th–90th percentile) | 4 m (−34 to 47) | 1 m (−75 to 36) |
| ERA5 T2m − IR10.8 | pending | pending |

- **Partial cover dominates.** A low-BTD miss is three times as likely as a cloud-free report to have a broken
  ceiling. These are decks that fill too little of the pixel to raise its BTD. A pixel rule cannot recover most of
  them; the teacher of phase C can at best learn their other signs.
- **Time helps.** A quarter of the low-BTD misses have fog at the same pixel in another slot of the same night,
  against a tenth of the cloud-free cases. This is the signal of phase B's time step.
- **The water-vapour contrast separates a little.** IR10.8 − WV6.2 is about 4 K smaller at the missed fog, but the
  middle halves overlap. It is a feature for the teacher, not a threshold.
- **Terrain is not the cause** of the low-BTD misses; the two terrain distributions are close.
- **The band** (28 %) is the 0.5 K ignore band and the BTD just above 4.0 K. Phase B's rule options (an emissivity
  index, a latitude-dependent threshold) aim at it; the first-cause shares cap what they can gain at about a quarter
  of the misses.

The ERA5 store (2 m temperature against the cloud-top temperature, a sign of a warm low cloud) was not built for this
run: it needs a Copernicus CDS key on the machine. `fogseg era5` then `fogseg diagnose` fill the last row.

## Phase B: rule repair and time

`fogseg experiment --phase B` relabels the case nights of each variant at every threshold of its grid and scores it
with a blocked four-fold cross-validation on the selection cases: each fold picks the threshold whose full-count FAR is
highest without passing 0.28 on its training part, and is scored on its test part. The pooled row adds up the four
test parts. The case set is the same for every variant, so a variant that calls a case obscured gets "no" for it.

The label options are new settings in the `labels:` section; their defaults give exactly the old labels.

| Option | Values |
|---|---|
| `texture` | `ir108_std3` (default), `btd_std3`, `ir108_mad5` (median deviation in 5 × 5), `blob` (median spread per fog patch), `none` |
| `fog_index` | `btd` (default), `emissivity`: the 3.9 µm pseudo-emissivity B(IR3.9) / B(IR10.8), turned into a BTD threshold per pixel |
| `split_window` | `fixed` (default), `kma`: IR10.8 − IR12.0 within ±1 K of −37.48 + 0.1329 · IR10.8 |
| `temporal_relax` | K, default 0: a near miss becomes fog if the same pixel is fog in the stored scene before or after and IR10.8 changed by less than 2 K |

### Cross-validation

| Variant | Threshold per fold | Pooled POD | Pooled FAR | Pooled CSI |
|---|---|---|---|---|
| 1 baseline | 4.25, 4.00, 4.50, 4.25 K | 0.33 | 0.27 | 0.29 |
| 2 texture `btd_std3` | same as 1 | 0.36 | 0.28 | 0.32 |
| 3 texture `ir108_mad5` | same as 1 | 0.36 | 0.28 | 0.32 |
| 4 texture `blob` | same as 1 | 0.36 | 0.28 | 0.32 |
| 5 texture `none` | same as 1 | 0.37 | 0.28 | 0.32 |
| 6 band 0.25 K | 4.50, 4.25, 4.75, 4.50 K | 0.33 | 0.27 | 0.29 |
| 7 emissivity | 0.81, 0.82, 0.80, 0.81 | 0.37 | 0.27 | 0.32 |
| 8 split window `kma` | 4.00, 4.25, 4.50, 5.50 K | 0.02 | 0.37 | 0.02 |
| 9 = 5 + 7 | 0.80, 0.82, 0.79, 0.81 | 0.38 | 0.28 | 0.33 |
| **10 = 9 + relax 0.5 K** | 0.80, 0.81, 0.79, 0.80 | **0.40** | 0.28 | **0.34** |
| 10 = 9 + relax 1.0 K | 0.80, 0.81, 0.78, 0.80 | 0.38 | 0.28 | 0.33 |
| 10 = 9 + relax 1.5 K | 0.79, 0.81, 0.77, 0.79 | 0.34 | 0.28 | 0.30 |

The baseline scores lower here than in the phase A table (0.39 on the selection cases) because each fold had to raise
its threshold to 4.25–4.5 K to keep the training FAR at 0.28; the phase A row is the configured 4.0 K.

- **The smoothness test costs POD.** All four texture options gain about 0.03 at the same FAR, and dropping the test
  gains the most. At matched FAR the threshold, not the texture, keeps the false alarms down.
- **A narrower uncertainty band changes nothing.** With 0.25 K each fold picks a threshold 0.25 K higher and lands on
  the same POD and FAR.
- **The emissivity index helps a little** (+0.04): the same BTD is a weaker signal on a warm top, which a fixed BTD
  threshold ignores.
- **The KMA split window fails on SEVIRI as written.** At the visible fog reports of the selection cases the median
  IR10.8 − IR12.0 is +0.37 K (10–90 %: −0.12 to 1.06 K), while the KMA line expects −1.9 K at their median IR10.8 of
  268 K; only 6 % lie within its ±1 K. Nearly all fog becomes obscured.
- **Time helps, in a small dose.** Relaxing the threshold by 0.5 K next to fog in the slot before or after adds 0.02;
  1.0 and 1.5 K add false alarms that force a stricter threshold and lose POD again.

The emissivity grid of the design, 0.80–1.00, ended too early: 0.80 is a BTD of about 4.4 K on a 268 K top, and
variants 9 and 10 found no threshold within it that kept the FAR at 0.28 on every training part. The grid is now
0.70–1.00 (`stations.emissivity_grid`), the same reach as the 7 K end of the BTD grid. Only the selection cases were
read for this change; the first run's table is kept as `stations/experiment_B_grid080.csv`.

### Held-out comparison

`fogseg experiment --phase B --held-out` chose the winner's threshold at FAR 0.28 on all selection cases (emissivity
0.80) and scored it once against the configured rule, fixed 4.0 K, on the held-out cases. Full count; the POD and FAR
differences come from a paired bootstrap over the 11 night blocks of the held-out cases.

| Part | Cases | Baseline POD | Winner POD | Baseline FAR | Winner FAR | Baseline CSI | Winner CSI |
|---|---|---|---|---|---|---|---|
| **held-out** | 27,011 | 0.43 | **0.47** | 0.25 | **0.24** | 0.38 | **0.41** |
| unseen nights only | 9,480 | 0.48 | 0.55 | 0.27 | 0.27 | 0.41 | 0.46 |
| unseen stations only | 14,603 | 0.37 | 0.38 | 0.25 | 0.21 | 0.33 | 0.34 |
| both | 2,928 | 0.54 | 0.61 | 0.23 | 0.23 | 0.46 | 0.52 |

- POD gain **+0.038**, 95 % interval **−0.004 to +0.069**.
- FAR difference −0.013, 95 % interval −0.046 to +0.002.

The keep rule needs the POD-gain interval above zero and a FAR that is not higher. The FAR is lower, but the interval
reaches −0.004, so **the winner is not kept** and the labels stay at fixed 4.0 K. The gain is probably real but too
small for eleven night blocks to confirm; it is reported, not claimed. It sits mostly on unseen nights (+0.06) and
hardly on unseen stations (+0.01), where the winner trades its gain for fewer false alarms.

The baseline rows repeat the phase A held-out table exactly, so the two phases score the same cases the same way. The
held-out set has now been opened twice: for the phase A baseline and for this one winner. Phase C took the last look.

## What phase B meant for phase C

- The target was the full-count POD at FAR 0.28 on the held-out cases: 0.43.
- A rule change buys about 0.04 at best. The low-BTD half of the misses is mostly partial cover, which only extra
  information (time, water vapour, ERA5, the stations themselves) can help with; the teacher should get the emissivity
  index, the neighbouring slots and no smoothness test as inputs rather than as fixed rules.
- Unseen stations score lower than unseen nights (POD 0.37 against 0.48), and phase B's rule barely moved them. The
  station-trained teacher of phase C has to be judged on that part, since it is the one where fitting to the stations
  shows.

## Phase C: station-trained teacher

`fogseg teacher` trains a per-pixel classifier on the station truth of the selection cases and scores it with the same
blocked four-fold cross-validation. The cases are the 48,669 selection cases (2,000 of them never usable: untrusted
pixel, no prediction), one per pixel and scene. A case that lies in a night fold and a station fold is tested twice, as
in phase B, so the pooled out-of-fold set has 85,521 rows.

**Features (27).** The groups `bt`, `differences`, `texture`, `geometry`, `terrain`, `time` and `rule` (the phase B rule
label). **ERA5 is left out.** The ERA5 store does not exist (there is
no CDS key yet), so `teacher.feature_groups` drops `era5`. Adding it later means rerunning `fogseg teacher`; the
held-out set is spent, so a new teacher could only be compared on the 2024-25 test season, once.

**Models and threshold.** Eight settings: logistic regression on standardized features (C 0.1 and 1) and
`HistGradientBoostingClassifier` (learning rate 0.05 and 0.1, 15, 31 and 63 leaves). The out-of-fold probabilities are
calibrated with isotonic regression, and p* is the probability at which the out-of-fold full-count FAR is 0.28. Pixels
within `teacher.ignore_halfwidth` (0.1) of p* get 255 and count as misses.

### Cross-validation

| Setting | p* | Pooled POD | Pooled FAR | Pooled CSI |
|---|---|---|---|---|
| 1 logistic C 0.1 | 0.49 | 0.563 | 0.275 | 0.464 |
| 2 logistic C 1 | 0.47 | 0.575 | 0.278 | 0.471 |
| 3 hgb lr 0.05, 15 leaves | 0.40 | 0.688 | 0.274 | 0.546 |
| **4 hgb lr 0.05, 31 leaves** | **0.41** | **0.696** | **0.276** | **0.550** |
| 5 hgb lr 0.05, 63 leaves | 0.42 | 0.696 | 0.276 | 0.550 |
| 6 hgb lr 0.1, 15 leaves | 0.41 | 0.696 | 0.278 | 0.549 |
| 7 hgb lr 0.1, 31 leaves | 0.42 | 0.690 | 0.277 | 0.546 |
| 8 hgb lr 0.1, 63 leaves | 0.45 | 0.661 | 0.271 | 0.530 |

**These scores are not comparable with phase B's.** p* is chosen on the same out-of-fold rows the table scores, so they
lean optimistic; phase B matched the threshold on the training part of each fold. For reference, phase B's
cross-validation baseline (fixed 4.0 K) was POD 0.33 at FAR 0.27. The table ranks the settings; the held-out table below
is the honest comparison.

- **Boosted trees beat logistic regression by 0.11 to 0.13 POD** at the same FAR.
- **Among the trees the settings are close.** Settings 4, 5 and 6 tie on POD; ties go to the lower number, so setting 4
  is the winner.

### Local correction

The correction spreads the residuals truth − p at the selection stations of the same scene: Δ = Σ wᵢ rᵢ / (1 + Σ wᵢ)
with w = exp(−d / L) within 3 L, added to p and re-thresholded at p* (0.41). The residual uses the out-of-fold p. In the
cross-validation the feeding stations exclude the test fold's and the held-out station folds, and any station within 5 km
of the target (a METAR and a SYNOP at the same site would otherwise feed each other).

| L | POD | FAR | CSI |
|---|---|---|---|
| none | 0.696 | 0.276 | 0.550 |
| 25 km | 0.724 | 0.250 | 0.583 |
| 50 km | 0.740 | 0.232 | 0.605 |
| **100 km (chosen)** | **0.747** | **0.223** | **0.615** |

100 km is the edge of the grid (25, 50, 100). The grid was not widened after seeing this table, so a longer scale was
never tried.

### Held-out comparison

`fogseg teacher --held-out` scored the winner (setting 4, p* 0.41) once on the held-out cases, with the 100 km correction
on the unseen-stations part. Full count; the rule row is the phase A baseline, the fixed 4.0 K labels, and the phase B
winner is shown for reference (it was not kept).

| Labels | Part | POD | FAR | CSI | Cases |
|---|---|---|---|---|---|
| Rule, fixed 4.0 K (phase A baseline) | held-out | 0.433 | 0.255 | 0.377 | 27,011 |
| Phase B winner (not kept) | held-out | 0.470 | 0.242 | 0.41 | 27,011 |
| **Teacher** | **held-out** | **0.769** | **0.266** | **0.601** | 27,011 |
| Teacher | unseen nights | 0.808 | 0.286 | 0.611 | 9,480 |
| Teacher | unseen stations | 0.722 | 0.250 | 0.582 | 14,603 |
| Teacher | both unseen | 0.834 | 0.262 | 0.643 | 2,928 |
| Rule | unseen nights | 0.485 | 0.272 | 0.410 | 9,480 |
| Rule | unseen stations | 0.366 | 0.246 | 0.327 | 14,603 |
| Rule | both unseen | 0.540 | 0.233 | 0.464 | 2,928 |
| Corrected teacher | unseen stations | 0.758 | 0.212 | 0.629 | 14,603 |

Only the unseen-stations part can be corrected: on unseen nights and on both-unseen, no selection station shares the
scene, so the corrected teacher there is the teacher. Intervals are from a paired bootstrap over the 14-day night
blocks.

- **Teacher against the rule:** POD gain **+0.336**, 95 % interval +0.289 to +0.400. FAR 0.266 against 0.255; the
  difference has an interval of −0.049 to +0.044.
- **Corrected teacher against teacher, unseen stations:** POD gain +0.035, 95 % interval −0.003 to +0.069. FAR 0.212
  against 0.250; the difference has an interval of −0.060 to −0.019.

**The teacher is not kept, because of the FAR clause alone.** The keep rule needs the POD-gain interval above zero,
which it clears by far, and a held-out FAR that is not higher than the rule's. The teacher's point estimate is 0.011
higher (not significant, and both are under the 0.28 target). The pre-registered rule is applied as written, so
`labels.mode` stays `fixed` at 4.0 K. Whether to switch to `labels.mode: teacher` despite the FAR clause is the owner's
decision; the store is already written (below).

**The correction is not kept.** Its FAR is significantly lower, but the POD interval reaches below zero, so the keep
rule fails. Dense labels carry no correction.

The held-out set has now been opened three times: for the phase A baseline, for the phase B winner and for this teacher.
This was the last look.

**A caveat about location.** The geometry and terrain features let the model learn the base rate of fog at each
location, which would help on stations it has seen and say nothing about new ones. The unseen-stations row is the test
of that: POD 0.722 against the rule's 0.366, at FAR 0.250 against 0.246. The gain holds there.

### Dense labels

`fogseg label --mode teacher` writes `labels_teacher.zarr`, a benchmark store, not the configured labels. 608 scenes took
293 s, with a peak memory of about 1.1 GB. `label` is `0`, `1`, `2` or `255`: untrusted pixels are `255` and obscured
pixels `2` as in the rule, and `255` also marks the band of ±0.1 around p* = 0.41. `fog_probability` is `uint8`,
round(p × 250), and 255 where p is undefined (untrusted, obscured, or no ERA5 when that group is used). No correction
is applied. Share of all pixels:

| Store | 0 | 1 | 2 | 255 | Fog among decided |
|---|---|---|---|---|---|
| `labels_fixed` (rule, configured) | 0.301 | 0.151 | 0.394 | 0.154 | 0.334 |
| `labels_teacher` (benchmark) | 0.279 | 0.215 | 0.394 | 0.112 | 0.436 |

7.9 % of all pixels fall in the teacher's ignore band. The obscured share is identical by construction.

## What this means next

- The teacher raises the held-out POD from 0.433 to 0.769 at a FAR within noise of the rule's. Only the FAR clause of
  the keep rule stops it; waiving it is the owner's call.
- ERA5 is still missing from the teacher. With the key, `fogseg era5` and a new `fogseg teacher` run would add its
  group; the result could be compared with this one only on the 2024-25 season, once.
- The correction's best length scale is the edge of its grid; nothing was tried beyond 100 km.
- A model trained on `labels_teacher.zarr` and scored on the held-out stations would start from labels that find most of
  the visible fog, with the location caveat above.

## Reproduce

```bash
fogseg label --overwrite                     # labels_fixed.zarr, now with the obscured class 2
fogseg stations --source metar               # re-parses the cached METARs into the new columns
fogseg dem                                   # dem_ceu.zarr, ~3 min, 405 GLO-90 tiles from AWS
fogseg era5                                  # era5_ceu.zarr, needs ~/.cdsapirc
fogseg split                                 # stations/splits_2023-24.csv, once
fogseg validate --season 2023-24             # cases and scores, with folds and strata
fogseg diagnose                              # stations/diagnosis_2023-24.csv, selection cases only
fogseg experiment --phase B                  # stations/experiment_B.csv, ~17 min, selection cases only
fogseg experiment --phase B --held-out       # stations/experiment_B_held_out.csv and .json, once per phase
fogseg teacher                               # phase C: cross-validation, correction grid, processed/teacher.joblib
fogseg teacher --held-out                    # the teacher against the rule on the held-out cases, once (the last look)
fogseg label --mode teacher                  # labels_teacher.zarr, 608 scenes in ~5 min
```
