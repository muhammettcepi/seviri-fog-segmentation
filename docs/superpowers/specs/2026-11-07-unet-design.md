# Design: a U-Net trained on the teacher labels

## 1. Goal

Train a small U-Net that maps one SEVIRI night scene to a fog / low-stratus map, train it on the dense teacher labels
of phase C, track every run with MLflow, and score it against the station truth with the same full-count POD, FAR and
CSI and the same blocked folds as the label work.

The labels are a means; the U-Net is the product. Two questions are answered:

1. Does a convolutional model trained only on the teacher's dense labels keep the teacher's skill at the stations
   (held-out POD 0.769, FAR 0.266) without the teacher's hand-made features, neighbouring slots and location inputs?
2. Does a station point loss, which adds the raw station truth of the training stations to the loss, help over pure
   distillation of the teacher?

Everything is measured on the 2023-24 training season. The 2024-25 test season stays closed: no tuning and no
decision on it, including the `cases_*_2024-25.csv` files. It is opened once at the end, only with the owner's OK, and
that run is not part of this design.

## 2. Decisions

The owner chose the teacher labels as the training labels ("chose the best and move on") and asked not to be asked.
Every other open point is a ruling: the decision, why, and the cost if it is wrong.

| # | Ruling | Why | Cost if wrong |
|---|---|---|---|
| U1 | `labels.mode: teacher`. The rule stages (`fogseg diagnose`, `experiment`, `teacher`) keep working on the fixed rule: they read `cases_fixed_<season>.csv` and `labels_fixed.zarr` when the mode is `fixed` or `teacher`, and refuse `dynamic` as before | The teacher's rule is the fixed rule, and the teacher must stay refittable once ERA5 exists | A reader may expect `labels.mode` to steer those stages too; the notes say it does not |
| U2 | The U-Net trains only on scenes of the tuning season whose night is in night folds 1–4. Night fold 5 is never trained on; the test season is never read | Unseen nights stay unseen images; the held-out protocol is the label work's | About a fifth fewer training scenes |
| U3 | Inputs per pixel: the 8 brightness temperatures, 3 differences (IR10.8 − IR3.9, IR10.8 − IR12.0, IR10.8 − WV6.2), the mean terrain height, the solar zenith angle and a trusted flag: 14 channels. Each is standardized with the mean and standard deviation over the trusted pixels of the training scenes; untrusted pixels are 0 in every channel but the flag | The differences carry the fog signal at a scale the network sees at once; no latitude, longitude or time input, so the model has to learn the physics and the geography only through terrain | The teacher's time and location features are lost; if they matter, the U-Net scores below the teacher |
| U4 | Four settings are cross-validated: **1 soft** (binary cross-entropy against `fog_probability` / 250), **2 hard** (against `label` 0 / 1), **3 soft + points 0.25** and **4 soft + points 1.0** (the soft loss plus λ times the mean cross-entropy at the station cases of the training scenes). Untrusted, obscured and undefined pixels are left out of the loss | Distillation keeps the teacher's uncertainty and its ignore band; the hard target is the plain alternative; the point loss is the earlier idea of the label design | Four settings × four folds of training time (about 1.5 h on the GTX 1050 Ti: 0.17 s per batch of 4) |
| U5 | A plain U-Net: four 2 × 2 poolings, two 3 × 3 convolutions with batch norm and ReLU per level, bilinear upsampling, skip connections, one output logit; 16 channels at the top level, doubling per level (about 1.9 M weights) | Enough capacity for ~240 scenes of 241 × 462 pixels; fits the 4 GB card with whole scenes | A wider net might score a little higher |
| U6 | Training: whole scenes zero-padded to 256 × 464 (a multiple of 16) and cropped back, batch 4, AdamW (learning rate 1e-3, weight decay 1e-4), cosine decay, 40 epochs, seed `stations.seed` plus the fold; no augmentation, no early stopping | A geostationary view has one fixed geography: a flip would teach a mirrored map and break the parallax and terrain relations. A fixed epoch count needs no validation set, so no fold is spent on stopping | More overfitting than with augmentation; the CV shows it if it is large |
| U7 | Cross-validation over the four night folds: fold *k* trains on the scenes of the other three selection night folds and predicts the scenes of fold *k*; it is scored on the selection cases of night fold *k*. p* is the probability at which the pooled out-of-fold full-count FAR is highest without passing 0.28 (grid 0.10–0.90, step 0.01). The winner is the setting with the highest pooled POD; ties go to the lower number | The teacher's protocol, so the numbers line up | The CV leans optimistic twice over: p* is chosen on the scored rows, and the teacher that made the labels saw these stations' truth. It ranks settings, it does not measure skill |
| U8 | The final model trains on all scenes of night folds 1–4 with the winner's setting and epochs; the point loss, if the winner has one, uses the selection cases of those nights. Its labels: 1 where p ≥ p*, 0 below, 255 untrusted, 2 where the rule says obscured; no ignore band | A product map should decide every visible pixel; the full count would score an ignore band as misses | A narrow band of uncertain pixels is now called one way or the other |
| U9 | `fogseg evaluate` scores the U-Net, the teacher labels and the rule on the held-out cases once and writes a record that blocks a rerun. It reports, it does not decide: no setting, threshold or config value changes after it | The held-out set has been opened three times for label choices and is spent for choosing; a model evaluation on it is still the natural protocol, and the notes say so | None for the choice; the reader has to weigh a fourth look |
| U10 | MLflow with a SQLite store and an artifact folder on `/mnt/d/fogseg-data/mlflow`, experiment `fogseg-unet`: one parent run per `fogseg train`, a nested run per setting and fold and one for the final model; `fogseg evaluate` logs into the final model's run | A file a reviewer can open with `mlflow ui`; no server | SQLite on the Windows drive may lock badly; the URI is a config key |
| U11 | `torch` and `mlflow` are an optional extra, `pip install -e ".[model]"`; CI installs the CPU wheel of torch and runs the model tests | The label pipeline stays installable without a 2 GB dependency | Two install lines in the README |
| U12 | `fogseg predict` writes the tuning season only; the test season needs `--final`, as `fogseg validate` | No accidental look at 2024-25 | None |
| U13 | One input normalization, fitted on the trusted pixels of all scenes of night folds 1–4 (the final model's training scenes), is shared by the CV folds | Reading the scenes once keeps the code simple; it uses input statistics only, never labels or truth | The CV folds see the input mean and spread of their test nights, a negligible leak |

## 3. Data flow

```
seviri_ceu.zarr ─┐
dem_ceu.zarr ────┼─► inputs (14 channels) ─► U-Net ─► p ─► predictions_unet.zarr ─► fogseg evaluate
labels_teacher ──┘        targets ──────────► loss ◄── station cases (point loss)       ▲
splits_2023-24.csv ─► night folds of the scenes                       cases_fixed_2023-24.csv
```

### 3.1 Scenes and folds

A scene's night is `night_of(time)`; its night block and fold come from `night_block_start` and the split file, as for
the cases. Scenes outside the tuning season are dropped before anything is read; scenes in night fold 5 are dropped
from every fit.

### 3.2 Inputs and targets

`unet_data.py` reads the scenes it needs once into memory as float32 (about 1.6 GB for 240 scenes), computes the 14
channels, and keeps the normalization (mean and standard deviation per channel) with the model. A pixel is trusted
where `labels.trusted_mask` says so. Targets come from the store of `labels.mode`:

- soft: `fog_probability / 250` where it is not 255; weight 0 elsewhere;
- hard: `label` where it is 0 or 1; weight 0 elsewhere.

The hard target works with any label store; the soft one needs `fog_probability`, so it refuses a rule store.

The point loss uses the scored selection cases (not hidden, truth yes or no) whose scene is in the fit: full-count
truth, yes → 1, no → 0, at the case's pixel `(iy, ix)`.

### 3.3 Model, training, prediction

`unet.py` holds the network only. `unet_train.py` holds the loss, the training loop, the cross-validation and the
saved model: a `torch.save` file with the weights, the setting, p*, the normalization, the input channel names, the
label store's attributes and the versions of torch and fogseg. `unet_predict.py` runs a model over the tuning season
and writes `predictions_unet.zarr` in the layout of the teacher store (`label`, `fog_probability` = round(p × 250),
the scene fractions), with the model's sha256 in its attributes.

### 3.4 Evaluation

`unet_eval.py` reads the label of each held-out case from three sources: the rule (the `label` column of
`cases_fixed_2023-24.csv`), the teacher (`labels_teacher.zarr`) and the U-Net (`predictions_unet.zarr`). It scores
each on the whole held-out set and its `nights`, `stations` and `both` parts, and runs two paired bootstraps over
the 14-day night blocks: U-Net against teacher and U-Net against rule, POD gain and FAR difference with 95 %
intervals. Two consistency checks guard the reading: the teacher row must reproduce phase C's held-out teacher row
(POD 0.769, FAR 0.266) and the rule row phase A's (0.433, 0.255); the notes report both.

`evaluate` refuses to run if `predictions_unet.zarr` was not made by the current model file, if its record exists, or
without the split file.

## 4. Commands

```bash
pip install -e ".[model]"     # torch and mlflow
fogseg train                  # 4 settings × 4 folds, the winner, the final model, stations/unet_cv.csv, MLflow runs
fogseg predict                # predictions_unet.zarr for the tuning season
fogseg evaluate               # the held-out comparison, once: stations/unet_held_out.csv and .json
```

`fogseg train` refuses a season other than the tuning season and refuses to run once `unet_held_out.*` exist (a new
model after the held-out look would silently replace the one that was scored).

## 5. Configuration

```yaml
labels:
  mode: teacher

unet:
  model_path: "/mnt/d/fogseg-data/processed/unet.pt"
  predictions_path: "/mnt/d/fogseg-data/processed/predictions_unet.zarr"
  tracking_uri: "sqlite:////mnt/d/fogseg-data/mlflow/mlflow.db"
  artifact_dir: "/mnt/d/fogseg-data/mlflow/artifacts"
  experiment: "fogseg-unet"
  base_channels: 16
  depth: 4
  epochs: 40
  batch_size: 4
  learning_rate: 0.001
  weight_decay: 0.0001
  point_weights: [0.25, 1.0]
  probability_grid: [0.10, 0.90, 0.01]
  device: auto          # cuda if available, else cpu
```

## 6. Error handling

- No GPU: train on the CPU with a warning (slow, but correct).
- A label store without `fog_probability` and a soft setting: refuse, naming the store.
- A label store whose dataset attributes do not match the dataset: the existing `check_labels_match`.
- A fold without any station case or without fog: the fold is unusable and the pooled row stays empty, as in phase B.

## 7. Testing

Synthetic scenes from the existing test helpers, on the CPU, with a 2-level, 4-channel U-Net and 2 epochs:

- input channels: values, standardization, untrusted pixels at 0, flag;
- targets and weights for soft and hard, refusal without `fog_probability`;
- point-loss gathering at the right pixels and batch entries;
- the network's output shape for an odd scene size after padding and cropping;
- the training loss falls on a learnable synthetic task;
- fold assignment: no scene of the test fold or of night fold 5 in a fit, no test-season scene ever;
- the CV table: matched p*, pooled row, winner and ties;
- the saved model round-trips; prediction labels follow p*, untrusted and obscured;
- the prediction store's layout and attributes; `evaluate`'s refusals and the parts of its table;
- MLflow runs land in a temporary SQLite store.

## 8. Documentation

- `docs/notes/unet.md`: protocol, the CV table, the held-out table with intervals, learning curves, example maps
  against the teacher and the rule, the honest caveats (fourth look at the held-out set, optimistic CV, location).
- README: the model section, commands and install extra; the dataset card: the prediction store and the label switch.
- `docs/notes/better_labels.md`: one line that `labels.mode` is now `teacher`.

## 9. Risks

- **The U-Net copies the teacher's errors.** It can only be as good as its labels away from the stations; the point
  loss is the one channel of raw truth.
- **Small data.** About 240 training scenes; the night folds hold about 60 each. Fold scores will scatter.
- **Hardware.** A GTX 1050 Ti with 4 GB: whole scenes at batch 4 fit with 16 top-level channels; 32 would need a
  smaller batch.
