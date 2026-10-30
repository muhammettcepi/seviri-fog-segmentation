# Checking the fog labels against station observations

Design: [`2026-10-28-station-validation-design.md`](../superpowers/specs/2026-10-28-station-validation-design.md).
Code: `fogseg stations`, `fogseg validate`, `fogseg tune` ([`src/fogseg/cli.py`](../../src/fogseg/cli.py)).
Settings: `configs/default.yaml`, sections `labels` and `stations`. Earlier checks:
[`btd_exploration.md`](btd_exploration.md) (where the thresholds came from) and [`label_check.md`](label_check.md)
(the visual check).

## Purpose

The fog labels are a rule on satellite data. Until now they were only looked at by eye. This note scores them against
what weather stations on the ground reported at the same time, tunes the BTD threshold on the **training season
2023-24**, and scores the result once on the **test season 2024-25**.

The tuning used the training season only. The test season was scored once, after every value had been fixed (see the
last section).

## Data

| Source | Stations on the grid | Observations (night windows, both seasons) | Where from |
|---|---|---|---|
| METAR (airports) | 306 | 686,292 | Iowa Environmental Mesonet, raw METAR decoded by `fogseg` |
| SYNOP (weather stations) | 869 | 679,662 in the training season | Ogimet, raw `AAXX` messages decoded by `fogseg`; coordinates from NOAA `isd-history.csv` |

A station is used when its nearest pixel centre is at most 5 km away and it lies at most 1000 m above sea level (a
mountain station reports fog whenever it is inside a cloud). 1175 stations pass; 39 of them have no usable case in the
training season (no report near the scan time).

### Truth rule

Following Egli et al. (2017) and Jahani et al. (2025), a station report counts as **fog or low stratus (FLS)** when:

| | METAR | SYNOP |
|---|---|---|
| FLS | visibility below 1000 m, or a `BKN` / `OVC` / `VV` layer with its base below 1000 m | visibility below 1000 m, or more than 5 oktas of low cloud with a base below 1000 m (`Nh` > 5, `h` ≤ 5), or sky obscured (`N` = 9) |
| No FLS | visibility and cloud known, none of the above | the same |
| Undefined | visibility or cloud missing | the same |

**Fog only** (visibility below 1000 m) is scored as a second view. It is much rarer: 11.6 % of the FLS reports.

### Matching

- **Pixel:** the nearest pixel centre.
- **Time:** SEVIRI scans Central Europe about 12 minutes after the nominal slot time, so each scene takes, per
  station, the report closest to slot time + 12 min, within 30 minutes. No report in that window, no case.
- **One case per pixel and scene:** when an airport and a SYNOP station share a pixel, the pooled score keeps the
  airport (its cloud base is exact). Each source is also scored on its own.
- **Not scored:**
  - Cases whose label is `255` are not scored; their number is reported.
  - Cases where the station pixel fails the labels' own high-cloud tests (`IR_108` below 255 K, or
    `IR_108 − IR_120` at least 1.5 K) are flagged **hidden**. The satellite cannot see what lies below, so they are
    left out of the main score and the tuning, and scored in their own view.

The training season gives 330,400 station reports matched to a scene. Of these:
- 182,725 have a defined truth;
- 94,805 of those are not hidden;
- 58,581 are scored at 4.0 K, after keeping one case per pixel and scene and leaving out label `255`.

### Scores

| Score | Meaning | Best |
|---|---|---|
| POD | share of the FLS reports that the labels call fog | 1 |
| FAR | share of the labelled fog that the stations call no FLS | 0 |
| CSI | hits / (hits + misses + false alarms) | 1 |
| Bias | labelled fog / reported FLS | 1 |
| HSS | Heidke skill score: correct answers above chance, counting correct "no" too | 1 |
| PSS | Peirce skill score: POD minus the false-alarm rate on the "no" reports | 1 |

Intervals are 95 % bootstrap intervals over whole nights (1000 draws, seed 0), because cases on the same night are not
independent. A tuned value replaces the current one only when the 95 % interval of the paired difference excludes 0.

## The labels before tuning

The labels as they came out of the visual check: `dynamic`, scene clear-ground peak + 1.5 K, fallback 5.0 K.
Training season, pooled:

| View | Cases | POD | FAR | CSI (95 % interval) | Bias |
|---|---|---|---|---|---|
| all stations | 65,316 | 0.273 | 0.185 | 0.257 (0.200–0.321) | 0.335 |
| METAR | 34,312 | 0.294 | 0.215 | 0.272 (0.218–0.333) | 0.374 |
| SYNOP | 46,512 | 0.262 | 0.166 | 0.249 (0.192–0.312) | 0.314 |
| fallback scenes | 2,317 | 0.330 | 0.209 | 0.304 (0.197–0.363) | 0.417 |

They miss most of the fog (POD 0.27, bias 0.34) and are right when they do call fog (FAR 0.19). Airports and SYNOP
stations agree within their intervals.

## Tuning

`fogseg tune` relabels the training season for each value on a grid and scores it against the stations. Three grids:

1. the dynamic `offset`;
2. the fixed value `btd_fixed` used as the dynamic mode's fallback;
3. `btd_fixed` in fixed mode.

The full grids are written to `tuning_*.csv` in the stations folder. The rows that matter:

**Fixed mode, `btd_fixed`** (CSI interval in brackets; "ignored" counts all cases labelled `255`):

| Threshold (K) | POD | FAR | False fog on "no" reports | CSI | Bias | HSS | PSS | Ignored |
|---|---|---|---|---|---|---|---|---|
| 3.25 | 0.620 | 0.408 | 0.353 | 0.434 | 1.05 | 0.265 | 0.266 | 16,637 |
| 3.50 | 0.588 | 0.362 | 0.274 | **0.441** (0.393–0.489) | 0.92 | 0.316 | 0.314 | 17,692 |
| 3.75 | 0.552 | 0.320 | 0.206 | 0.438 | 0.81 | 0.353 | 0.346 | 18,256 |
| **4.00** | 0.502 | 0.280 | 0.146 | 0.420 (0.369–0.472) | 0.70 | **0.371** | **0.356** | 17,798 |
| 4.25 | 0.451 | 0.251 | 0.105 | 0.392 | 0.60 | 0.368 | 0.345 | 16,771 |
| 4.50 | 0.392 | 0.225 | 0.074 | 0.352 | 0.51 | 0.347 | 0.317 | 15,395 |
| 5.00 | 0.274 | 0.193 | 0.040 | 0.257 (0.207–0.311) | 0.34 | 0.269 | 0.234 | 12,410 |

**Dynamic mode, `offset`** (fallback 5.0 K):

| Offset (K) | POD | FAR | CSI | HSS |
|---|---|---|---|---|
| 0.00 | 0.526 | 0.318 | 0.422 (0.377–0.468) | 0.337 |
| 0.50 | 0.453 | 0.265 | 0.389 | 0.354 |
| 1.00 | 0.364 | 0.223 | 0.330 | 0.328 |
| 1.50 (before) | 0.273 | 0.185 | 0.257 | 0.269 |
| 2.00 | 0.199 | 0.155 | 0.192 | 0.207 |

The dynamic grid's best CSI lies at its lower edge, offset 0. That means the threshold sits right at the scene's
clear-ground peak, which is no longer a valley between ground and cloud. At that offset many scenes hit the lower clamp
of 3.5 K (42 % of the dynamic training scenes at offset 0), so the method ends up close to a fixed threshold anyway. The
fixed mode reaches the same CSI with one number and no edge optimum.

### Decisions

Decided with the project owner on 29 October 2026:

1. **`labels.mode` is now `fixed`, with `btd_fixed` = 4.0 K** (it was `dynamic`, offset 1.5 K, fallback 5.0 K).
   - CSI rises to 0.420 (0.369–0.472), from 0.257 (0.207–0.311) for the old fixed 5.0 K. For the CSI optimum, 3.5 K,
     the paired interval of the gain is +0.15 to +0.22, far from 0.
   - The design said to tune on CSI, and CSI peaks at 3.5 K (0.441). **4.0 K was chosen instead**, after the results
     were seen, for this reason:
     - CSI does not count correct "no" answers, so it rewards calling too much fog when fog is common (40 % of the
       visible reports here are FLS).
     - At 3.5 K, 27 % of all visible "no" reports and **48 % of the cloud-free reports in the north** are labelled fog
       (see the drift section).
     - HSS and PSS, which count the correct "no" answers, both peak at 4.0 K.
     - At 4.0 K the false fog on "no" reports halves (0.274 → 0.146) and CSI loses 0.02, well inside the interval of the
       3.5 K value.
   - A model trained on these labels learns the false fog along with the real fog, so the lower false-fog rate was
     judged worth more than the last 0.02 of CSI. **This is a change of objective after seeing the data**, and it is
     stated here so that the 4.0 K result is not read as a pre-registered CSI optimum.
2. **The dynamic mode stays in the code**, with offset 1.5 K, as the documented alternative. Its own fallback is
   `btd_fixed`, now 4.0 K. On the fallback scenes this alone raises CSI from 0.304 to 0.593 (2,575 cases): the
   suspicion of [`label_check.md`](label_check.md) that 5.0 K cuts through the deck on those nights was right.
3. **Nothing else was tuned.** The warm-top, cirrus, smoothness, band and patch tests keep their values. The tuning
   grids in `configs/default.yaml` were widened (offset from 0.0 K, `btd_fixed` from 3.0 K) so that the chosen value is
   not at an edge.

The training-season label statistics change accordingly. In fixed mode at 4.0 K:
- the mean fog fraction is 0.125 (0.053 at 5.0 K);
- the mean ignore fraction is 0.146 (0.096), because the 0.5 K band now lies where many pixels are.

## The chosen labels on the training season

Fixed 4.0 K, training season 2023-24, all views (the relabelled store reproduces the tuning row exactly):

| View | Group | Cases | POD | FAR | CSI (95 % interval) | Bias | FLS labelled `255` | No-FLS labelled `255` | FLS hidden |
|---|---|---|---|---|---|---|---|---|---|
| main | all | 58,581 | 0.502 | 0.280 | 0.420 (0.369–0.472) | 0.70 | 6,655 | 12,905 | 33,163 |
| source | METAR | 30,108 | 0.520 | 0.319 | 0.418 (0.369–0.469) | 0.76 | 2,786 | 7,244 | 13,145 |
| source | SYNOP | 42,105 | 0.492 | 0.257 | 0.421 (0.368–0.473) | 0.66 | 5,538 | 8,541 | 27,278 |
| latitude | 43–47° N | 16,404 | 0.464 | 0.434 | 0.342 (0.303–0.379) | 0.82 | 1,031 | 4,046 | 3,225 |
| latitude | 47–51° N | 22,291 | 0.474 | 0.247 | 0.410 (0.353–0.471) | 0.63 | 2,898 | 4,823 | 14,199 |
| latitude | 51–56° N | 19,449 | 0.538 | 0.255 | 0.455 (0.386–0.518) | 0.72 | 2,720 | 3,855 | 15,704 |
| slot | 21:00 | 14,357 | 0.529 | 0.335 | 0.418 (0.364–0.472) | 0.80 | 1,113 | 3,108 | 8,343 |
| slot | 00:00 | 14,919 | 0.463 | 0.284 | 0.391 (0.340–0.445) | 0.65 | 1,447 | 2,415 | 8,091 |
| slot | 03:00 | 15,483 | 0.507 | 0.265 | 0.429 (0.374–0.482) | 0.69 | 1,507 | 2,526 | 8,377 |
| slot | 05:00 | 13,822 | 0.509 | 0.237 | 0.440 (0.379–0.498) | 0.67 | 2,588 | 4,856 | 8,352 |
| fog only | all | 70,342 | 0.500 | 0.918 | 0.076 (0.061–0.091) | 6.10 | 1,249 | 22,576 | 3,050 |
| with hidden | all | 127,749 | 0.218 | 0.280 | 0.201 (0.164–0.243) | 0.30 | 6,655 | 12,905 | 33,163 |

How to read the views:
- **Source:** the METAR and SYNOP rows overlap, since a pixel can have both. The main row keeps one case per pixel
  and scene.
- **Fog only:** this view scores the same labels against visibility below 1 km. FAR 0.92 says what the labels are:
  they mark low water cloud, most of which does not touch the ground.

## Where the labels fail

### Fog under high cloud

40,669 of the 79,587 FLS reports (51 %) lie under a pixel that fails the high-cloud tests. The labels call these `0`,
and no BTD threshold can find them. Counted in, CSI falls from 0.420 to 0.201 and POD from 0.50 to 0.22. This is the
largest single limit of the labels. A model trained on them learns that there is no fog under high cloud.

### Ignored fog

10.5 % of the defined FLS reports (20 % of the visible ones) have label `255`:

| Reason | FLS | No FLS |
|---|---|---|
| within 0.5 K of the threshold | 6,511 | 11,940 |
| twilight (solar zenith angle 95° or less) | 1,408 | 3,419 |
| in a patch of fewer than 5 pixels | 466 | 489 |
| missing channel | 1 | 2 |

These are station reports, before one case per pixel and scene is kept. The band is the main reason: at 4.0 K it lies
where much of the fog is (see the next section). The scores above leave these cases out. A label that ignores fog is
not "right" about it, so the ignored counts are part of the result.

### Visible fog that looks like clear ground

For the visible reports, the share that is FLS rises with the BTD at the station pixel:
- below the threshold it stays between 19 and 34 %;
- it passes 50 % about 0.5 K above the threshold (at about 4.5 K);
- it reaches 90 % only 4 K above it.

So a large part of the reported FLS has the BTD of clear ground:
- the median visible FLS report lies only 0.4 K above the threshold;
- 10 % of them lie more than 2.4 K below it.

60 % of the visible FLS reports labelled `0` fail the smoothness test (`IR_108` standard deviation of 1.5 K or more in
3 × 3 pixels), so broken or patchy decks are a second reason for misses. Neither can be fixed by moving the threshold.

### North-south drift

[`btd_exploration.md`](btd_exploration.md) found that clear ground has a higher BTD in the north (about +1 K, mixed with
the viewing angle). The stations confirm it. The BTD medians at visible, night-time station pixels:

| Latitude | Cloud-free reports | FLS reports |
|---|---|---|
| 43–47° N | 3.0 K (METAR), 3.1 K (SYNOP) | 4.1 K |
| 47–51° N | 3.0 K, 3.2 K | 4.3 K |
| 51–56° N | 3.8 K, 3.9 K | 4.6 K |

Clear ground in the north lies 0.8 to 0.9 K higher, and fog only 0.5 K higher. So one threshold for the whole region
calls more clear ground "fog" in the north. The share of cloud-free reports (no cloud layer at all), visible and
decided, that are labelled fog:

| Latitude | at 3.5 K | at 4.0 K (chosen) |
|---|---|---|
| 43–47° N | 0.13 | 0.05 |
| 47–51° N | 0.24 | 0.09 |
| 51–56° N | 0.48 | 0.26 |

The FAR per latitude band hides this. It is *highest* in the south (0.43 against 0.26) because FLS is much more common
in the north (59 % of the scored cases in 51–56° N against 19 % in 43–47° N), so the same false fog is
diluted by more hits. Judged on the cloud-free reports, the north is clearly worse.

**One quarter of the cloud-free northern ground is labelled fog even at 4.0 K.**

**What drives it.** On 38,672 cloud-free reports, with each scene's mean removed so that night-to-night changes do not
mix in:

- **Viewing angle:** clear-ground BTD rises 0.07 K per degree (95 % interval 0.05–0.09).
  - Latitude and viewing angle correlate 0.96 at the stations, so the data cannot fully separate them.
  - Fitted together, viewing angle keeps its effect (+0.12 K per degree) and latitude does not, which fits the physics:
    a longer slant path absorbs more at 3.9 µm.
- **Ground height** matters little: −0.6 K per 1000 m, under 1 % of the variance, and mostly because high ground is
  colder.
- **Ground temperature** (`IR_108`) is the largest factor within a scene: +0.13 K per K, 19 % of the variance.
- **Unexplained:** most of the variance remains. After viewing angle and height, station means still differ by 0.38 K
  (standard deviation), most likely from the surface type.

**Would a regional threshold help?** A rough test at the station pixels used the BTD rule only, without the smoothness,
band and patch tests, so its numbers compare rules with each other, not with the label scores:

| Rule | HSS | Cloud-free reports called fog, 43–47° N / 51–56° N |
|---|---|---|
| one threshold | **0.351** | 0.06 / 0.27 |
| threshold rising 0.10 K per degree of viewing angle | 0.317 | 0.13 / 0.12 |
| clear-ground peak per scene + offset | 0.329 | 0.10 / 0.29 |
| clear-ground peak per latitude band and scene + offset | 0.278 | 0.13 / 0.12 |

Both regional rules even out the false fog between south and north, but they lose more than they gain:

- **Northern fog is lost.** Fog BTD rises only 0.5 K to the north and fog is common there, so a higher northern
  threshold throws away much real fog.
- **Band peaks are unstable.** The peak per band moves from night to night: the north-south difference has a standard
  deviation of 0.9 K, and the bands correlate only 0.2–0.4 with each other.
- **The band peak can be the fog deck.** In a band that is mostly fog, the histogram peak is the fog deck itself, which
  pushes the threshold too high.

The northern false fog is a limit of the BTD itself, where clear ground and fog overlap. Moving the threshold does not
remove it. The single threshold was kept.

### Fallback scenes

27 of the 608 scenes (11 in the training season) have no clear-ground peak inside the window. The dynamic mode then
uses `btd_fixed`. Measured on those scenes:

| Fallback value | Cases | POD | FAR | CSI |
|---|---|---|---|---|
| 5.0 K (before) | 2,317 | 0.330 | 0.209 | 0.304 |
| 4.0 K (now) | 2,575 | 0.708 | 0.215 | 0.593 |

The hump at the window edge on these nights is mostly a low-cloud deck, as the visual check suspected. In fixed mode
the question does not arise: every scene uses 4.0 K.

### Point against pixel

A station is a point, a pixel is 3 to 5 km across. Patchy fog gives misses and false alarms that are nobody's fault.
Airports avoid valleys, so valley fog is under-sampled; SYNOP stations help a little. None of this is corrected.

## Benchmarks

These are other studies' numbers, quoted for scale and not used for any decision. They come from different years,
stations and case selections, so they are not directly comparable:

- Jahani et al. (2025), SEVIRI FLS against METAR over Europe, test set: POD 0.82, FAR 0.31, CSI 0.60.
- FogPy / DWD against SYNOP visibility below 1 km: POD 71–74 %, FAR 15 % (low cloud) and 34 % (ground fog).

The labels here reach CSI 0.42 on the visible cases with a single BTD threshold and no cloud mask, well below the
published methods. They are weak labels for training, not a fog product.

## Test season: the one final evaluation

**Pending:** the test-season SYNOP download was still running when this was written. This section is filled in after
`fogseg validate --labels fixed --season 2024-25 --final` (and the same for `dynamic`) has been run once, with every
value above already fixed.

## Next step: better labels before the U-Net

The labels find half of the visible fog, and their errors are systematic (northern clear ground, patchy fog, fog under
high cloud). A U-Net trained only on them would most likely learn those errors: no published study was found in which
a network trained on a fog rule's output beat that rule against stations. The information that can correct them is
the station network itself.

A literature review ([`label_improvement_research.md`](label_improvement_research.md)) suggests these steps, cheapest
first. They are the next piece of work, in their own spec:

1. **Seal an evaluation inside the training season:** hold out blocks of nights and of stations, and judge every change
   only on them. The test season stays closed.
2. **Stop calling fog under high cloud "no fog":** an own class or `255` instead of `0`.
3. **Station-calibrated soft labels** instead of a hard 4.0 K, with a wider ignore band around it.
4. **Relax the smoothness test,** which rejects 60 % of the visible fog the labels miss.
5. **A station point loss** next to the dense labels when the U-Net is trained.
6. **A station-trained model** (in the style of Jahani et al. 2025) as a benchmark and, if it beats the rule on
   held-out stations, as the source of the dense labels.

## Caveats

- **The objective changed after seeing the data** (CSI → HSS / PSS, decision 1). Both optima are reported.
- **CSI favours over-detection** when the event is common, which is why it alone was not trusted.
- **More pixels are ignored** with 4.0 K than with 5.0 K (training mean ignore fraction 0.146 against 0.096), because
  the band now lies where most pixels are.
- **Truth is a report, not a measurement of the cloud deck:**
  - a METAR `BKN` deck at 900 m counts as FLS;
  - a `SCT` deck at 200 m does not;
  - SYNOP's `h` is the base of the lowest cloud of any amount.

## Reproduce

```bash
fogseg stations                                   # METAR (IEM) and SYNOP (Ogimet); resumable, many hours for SYNOP
fogseg validate --season 2023-24                  # scores of labels.mode on the training season
fogseg validate --labels dynamic --season 2023-24
fogseg tune                                       # tuning grids on the training season, tuning_*.csv
fogseg validate --season 2024-25 --final          # the one test-season run
```

Tables are written to `stations.data_dir`: `observations_*.csv`, `cases_<mode>_<season>.csv`,
`scores_<mode>_<season>.csv`, `tuning_*.csv`. Each `validate` run reads the label store as it is, so relabel first
(`fogseg label --overwrite`).

## References

- Egli, S., Thies, B., Drönner, J., Cermak, J., Bendix, J. (2017): A 10 year fog and low stratus climatology for
  Europe based on Meteosat Second Generation data. *Q. J. R. Meteorol. Soc.* 143, 530–541.
- Jahani, B., Karalus, S., Fuchs, J., Zech, T., Zara, M., Cermak, J. (2025): Algorithm for continual monitoring of
  fog based on geostationary satellite imagery. *Atmos. Meas. Tech.* 18, 1927:
  <https://amt.copernicus.org/articles/18/1927/2025/>
- Cermak, J. & Bendix, J. (2008): A novel approach to fog/low stratus detection using Meteosat 8 data.
  *Atmos. Res.* 87, 279–292.
- Iowa Environmental Mesonet ASOS/METAR archive: <https://mesonet.agron.iastate.edu/request/download.phtml>
- Ogimet SYNOP archive: <https://www.ogimet.com/>
- NOAA Integrated Surface Database station history: <https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv>
- WMO Manual on Codes (WMO-No. 306), FM 12 SYNOP and FM 15 METAR.

Contains modified EUMETSAT Meteosat SEVIRI data (2023-24).
