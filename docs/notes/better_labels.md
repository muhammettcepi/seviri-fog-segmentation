# Better fog labels from the station network

Phase A of [the design](../superpowers/specs/2026-10-31-better-labels-design.md): an evaluation protocol that can tell
a real gain from noise, cleaner matching between stations and pixels, an `obscured` label, the terrain and ERA5 data the
later phases need, and a diagnosis of the missed fog. Nothing here changes which pixels are labelled fog; phases B
(rule repair and time) and C (a station-trained teacher) do that and are scored against the baseline below.

Everything is measured on the 2023-24 training season. The 2024-25 test season was not opened.

## Protocol

`fogseg split` writes `stations/splits_2023-24.csv` once:

- **Night blocks:** 14-day blocks of the season's nights, dealt to folds 1–5 in turn (block *i* → fold *i* mod 5 + 1),
  so every fold holds early and late winter.
- **Station cells:** 2° × 2° cells from the south-west corner of the region, dealt to folds 1–5 in a random order fixed
  by `stations.seed`.
- **Held-out:** every case whose night or station is in fold 5, 36 % of the cases. From now on nothing is chosen on
  it, and each phase opens it once, for its one winner. It is reported in three parts: unseen nights only, unseen
  stations only, both.
- **Selection:** nights and stations both in folds 1–4. Every threshold and variant of phases B and C is chosen here,
  with a blocked four-fold cross-validation. `fogseg tune` and `fogseg diagnose` read only these cases and refuse to
  run without the split file.

One caveat: the baseline's 4.0 K was chosen in PR #5 on all 2023-24 cases, before the split existed, so the held-out
cases have seen that choice. This favours the baseline, which makes a later gain over it harder to show, not easier.

The main score is the **full count**: label `1` is "fog", and `0`, `2` and `255` are all "no fog". An ignored pixel
under a fog report is then a miss, which the old decided-only count did not see. Phases B and C compare at the same
FAR, 0.28 (`stations.matched_far`), and report the POD gain with a paired bootstrap over nights.

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

## What this means for phases B and C

- The target is the full-count POD at FAR 0.28 on the held-out cases: 0.43 now.
- The band and the texture causes (43 % of the misses) are what a better rule can reach. The low-BTD half is mostly
  partial cover, which only extra information (time, water vapour, ERA5, the stations themselves) can help with.
- Unseen stations score lower than unseen nights (POD 0.37 against 0.48). The station-trained teacher of phase C has to
  be judged on that part, since it is the one where fitting to the stations shows.

## Reproduce

```bash
fogseg label --overwrite                     # labels_fixed.zarr, now with the obscured class 2
fogseg stations --source metar               # re-parses the cached METARs into the new columns
fogseg dem                                   # dem_ceu.zarr, ~3 min, 405 GLO-90 tiles from AWS
fogseg era5                                  # era5_ceu.zarr, needs ~/.cdsapirc
fogseg split                                 # stations/splits_2023-24.csv, once
fogseg validate --season 2023-24             # cases and scores, with folds and strata
fogseg diagnose                              # stations/diagnosis_2023-24.csv, selection cases only
```
