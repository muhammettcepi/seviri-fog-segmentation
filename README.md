# Night-time fog detection from Meteosat SEVIRI

[![CI](https://github.com/muhammettcepi/seviri-fog-segmentation/actions/workflows/ci.yml/badge.svg)](https://github.com/muhammettcepi/seviri-fog-segmentation/actions/workflows/ci.yml)

Segmenting fog and low stratus at night from Meteosat SEVIRI infrared imagery over Central Europe:
a physics-based baseline (the 10.8 − 3.9 µm brightness temperature difference), weak labels derived
from it, a small U-Net trained on them, and an evaluation of both against independent airport
visibility observations (METAR).

> **Status: work in progress.** Data access and preprocessing work; labels, model and evaluation are next.

## Why this works at night

Small water droplets emit less efficiently at 3.9 µm than at 10.8 µm. At night, the brightness
temperature difference BT(10.8) − BT(3.9) is therefore clearly positive over fog and low stratus
and close to zero over clear land. A warm cloud top (high BT 10.8) tells us the cloud is low.
During the day, reflected sunlight at 3.9 µm hides this signal, so the project is night-only.

## What exists so far

| Part | Where |
|---|---|
| Access a first scene with `eumdac` + `satpy` (Colab how-to) | [`notebooks/01_access_seviri_data.ipynb`](notebooks/01_access_seviri_data.ipynb) |
| Resumable bulk download → crop → small NetCDF per scene | `fogseg fetch` ([`src/fogseg/cli.py`](src/fogseg/cli.py)) |
| Quick access test (token, search, 1 KB download) | `fogseg check` |
| All settings (region, channels, seasons, night slots, paths) | [`configs/default.yaml`](configs/default.yaml) |

Dataset so far: 608 night scenes (two fog seasons, Oct–Feb 2023/24 and 2024/25, 4 slots per
night, every second night), 8 IR channels on SEVIRI's native grid, cropped to 2–20°E, 43–55°N
(241 × 462 pixels).

## Quickstart (Linux / WSL)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# EUMETSAT API consumer key + secret (https://api.eumetsat.int/api-key); never commit them
read -s EUMETSAT_KEY && read -s EUMETSAT_SECRET && export EUMETSAT_KEY EUMETSAT_SECRET

fogseg check                          # can we log in, search and download?
fogseg fetch --dry-run                # how many scenes would be fetched
fogseg fetch --workers 2              # download + crop all scenes (resumable)
```

Paths in `configs/default.yaml` point to `/mnt/d/...` (a Windows D: drive seen from WSL); change
them for your machine. Satellite data is never committed to this repository.

## CI and releases

Every push to `main` and every pull request runs `ruff`, `pytest` on Python 3.11–3.13, and a
package build ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)). To release, bump `version`
in `pyproject.toml`, commit, and push a matching tag:

```bash
git tag v0.2.0 && git push origin v0.2.0
```

[`release.yml`](.github/workflows/release.yml) then reruns CI, publishes the package to PyPI and
creates a GitHub Release with the built files.

## Data

Meteosat SEVIRI Level 1.5 data (`EO:EUM:DAT:MSG:HRSEVIRI`) from the EUMETSAT Data Store, used under
the EUMETSAT data licence. Downloading requires a free EUMETSAT account with the licences accepted.

## License

Code: [MIT](LICENSE).
