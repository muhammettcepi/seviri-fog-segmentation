"""Draw the explainer figure of the night fog signal: one fog pixel and one clear-ground pixel, both channels.

Run from the repository root: `python docs/explainers/make_fog_signal_figure.py`. It needs the dataset
and the dynamic label store named in `configs/default.yaml`.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from fogseg.config import load_config
from fogseg.inspect import night_microphysics_rgb
from fogseg.label_store import label_store_path
from fogseg.labels import FOG, NO_FOG, LabelParams

CONFIG_PATH = Path("configs/default.yaml")
OUTPUT = Path("docs/explainers/figures/fog_signal_real_pixels.png")
CAPTION = "Contains modified EUMETSAT Meteosat SEVIRI data (2023-24)"
SCENE_TIME = pd.Timestamp("2024-01-12 05:00")
# Bavaria to Czechia: land only, so the clear-ground pixel cannot fall on the sea
INLAND_LAT = (48.0, 50.5)
INLAND_LON = (9.0, 16.0)


def closest_pixel(values: np.ndarray, candidates: np.ndarray, target: float) -> tuple[int, int]:
    """Row and column of the candidate pixel whose value is closest to `target`."""
    distance = np.where(candidates, np.abs(values - target), np.inf)
    row, column = np.unravel_index(np.argmin(distance), values.shape)
    return int(row), int(column)


def main() -> None:
    config = load_config(CONFIG_PATH)
    dataset = xr.open_zarr(config["data"]["processed_path"], consolidated=False)
    store = label_store_path(Path(config["labels"]["output_dir"]), "dynamic")
    labels = xr.open_zarr(store, consolidated=False)

    scene = dataset.sel(time=SCENE_TIME)
    ir039 = scene["bt"].sel(channel="IR_039").values
    ir108 = scene["bt"].sel(channel="IR_108").values
    ir120 = scene["bt"].sel(channel="IR_120").values
    latitude = dataset["lat"].values
    longitude = dataset["lon"].values
    label = labels["label"].sel(time=SCENE_TIME).values
    threshold = float(labels["btd_threshold"].sel(time=SCENE_TIME))
    params = LabelParams.from_config(labels.attrs)
    btd = ir108 - ir039

    is_fog = label == FOG
    is_inland = (
        (latitude > INLAND_LAT[0])
        & (latitude < INLAND_LAT[1])
        & (longitude > INLAND_LON[0])
        & (longitude < INLAND_LON[1])
    )
    is_clear_ground = (
        (label == NO_FOG)
        & (ir108 >= params.warm_min_bt108)
        & (ir108 - ir120 < params.cirrus_max_split)
        & is_inland
    )
    ground_level = threshold - labels.attrs["dynamic_offset"]
    pixels = {
        "A": ("fog", closest_pixel(btd, is_fog, float(np.median(btd[is_fog])))),
        "B": ("clear ground", closest_pixel(btd, is_clear_ground, ground_level)),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.4), layout="constrained")
    rgb_ax, btd_ax, ir108_ax, ir039_ax = axes.ravel()

    rgb_ax.imshow(night_microphysics_rgb(ir039, ir108, ir120))
    rgb_ax.set_title("Night Microphysics RGB (fog = pale yellow-green)")

    btd_map = btd_ax.imshow(btd, cmap="viridis", vmin=0, vmax=8)
    fig.colorbar(btd_map, ax=btd_ax, shrink=0.85, label="BTD (K)")
    btd_ax.set_title(f"BTD = IR_108 − IR_039 (this night's threshold {threshold:.2f} K)")

    both_channels = np.concatenate([ir039.ravel(), ir108.ravel()])
    low, high = np.nanpercentile(both_channels, [2, 98])
    for ax, image, name in ((ir108_ax, ir108, "IR_108 (10.8 µm)"), (ir039_ax, ir039, "IR_039 (3.9 µm)")):
        shown = ax.imshow(image, cmap="inferno", vmin=low, vmax=high)
        fig.colorbar(shown, ax=ax, shrink=0.85, label="brightness temperature (K)")
        ax.set_title(f"{name}, same colour scale in both")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for letter, (_, (row, column)) in pixels.items():
            ax.plot(column, row, "o", markersize=11, markerfacecolor="none", markeredgecolor="white", mew=2)
            ax.annotate(
                letter,
                (column, row),
                xytext=(8, -8),
                textcoords="offset points",
                color="white",
                fontsize=13,
                weight="bold",
            )

    lines = [f"{SCENE_TIME:%Y-%m-%d %H:%M} UTC, real SEVIRI pixels"]
    for letter, (name, (row, column)) in pixels.items():
        lines.append(
            f"{letter} {name}:  IR_108 {ir108[row, column]:.1f} K   IR_039 {ir039[row, column]:.1f} K   "
            f"BTD {btd[row, column]:+.1f} K"
        )
    fig.suptitle("\n".join(lines), fontsize=12)
    fig.text(0.995, 0.004, CAPTION, ha="right", va="bottom", fontsize=7, color="0.3")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=100, facecolor="white")
    print(lines[1:], f"written to {OUTPUT}", sep="\n")


if __name__ == "__main__":
    main()
