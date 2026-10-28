"""Check figures for the labels: one PNG per scene, to judge the labels by eye.

The four panels are the Night Microphysics RGB, the labels over it, the histogram of the BTD
values the threshold was chosen from, and the solar zenith angle with the night limit.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.patches import Patch

from fogseg.dynamic_threshold import DynamicParams
from fogseg.label_store import check_labels_match
from fogseg.labels import FOG, IGNORE, LabelParams, histogram_pixels

# Set after the imports on purpose: figures are only written to files, never shown, so no GUI
# toolkit may be needed (the command also runs on a machine without a display).
matplotlib.use("Agg")

log = logging.getLogger("fogseg")

# EUMETSAT Night Microphysics recipe: (value in K that maps to 0, value in K that maps to 1)
RED_RANGE = (-4.0, 2.0)  # IR_120 - IR_108
GREEN_RANGE = (0.0, 10.0)  # IR_108 - IR_039
BLUE_RANGE = (243.0, 293.0)  # IR_108

NEAREST_TIMES_LISTED = 3
FOG_COLOUR = (0.0, 1.0, 1.0)
IGNORE_COLOUR = (0.5, 0.5, 0.5)
FOG_ALPHA = 0.6
IGNORE_ALPHA = 0.5
IGNORE_HATCH = "///"
TIME_FORMAT = "%Y-%m-%d %H:%M"
FILE_TIME_FORMAT = "%Y%m%dT%H%M"
FIGURE_SIZE = (14, 11)
FIGURE_DPI = 110


def scale_to_unit(values: np.ndarray, value_range: tuple[float, float]) -> np.ndarray:
    """Map `value_range` linearly to 0..1, clip outside it, and make missing values 0."""
    low, high = value_range
    scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
    return np.nan_to_num(scaled, nan=0.0)


def night_microphysics_rgb(ir039: np.ndarray, ir108: np.ndarray, ir120: np.ndarray) -> np.ndarray:
    """Night Microphysics RGB of one scene, shape (y, x, 3) with values 0..1; missing pixels are black."""
    red = scale_to_unit(ir120 - ir108, RED_RANGE)
    green = scale_to_unit(ir108 - ir039, GREEN_RANGE)
    blue = scale_to_unit(ir108, BLUE_RANGE)
    return np.stack([red, green, blue], axis=-1)


def find_scene(times: pd.DatetimeIndex, wanted: pd.Timestamp) -> int:
    """Index of the scene at time `wanted`; the error for a missing time lists the nearest scenes."""
    matches = np.flatnonzero(times == wanted)
    if matches.size > 0:
        return int(matches[0])

    distances = np.abs(times - wanted)
    nearest = np.argsort(distances, kind="stable")[:NEAREST_TIMES_LISTED]
    listed = ", ".join(times[i].strftime(TIME_FORMAT) for i in nearest)
    raise ValueError(f"No scene at {wanted.strftime(TIME_FORMAT)}. Nearest available times: {listed}")


def histogram_title(threshold: float, source: str) -> str:
    """Title of the histogram panel: the threshold used and where it came from."""
    return f"BTD histogram, threshold {threshold:.2f} K ({source})"


def figure_path(out_dir: Path, mode: str, time: pd.Timestamp) -> Path:
    """File of the check figure, e.g. `inspect_fixed_20231121T0000.png`."""
    return out_dir / f"inspect_{mode}_{time.strftime(FILE_TIME_FORMAT)}.png"


def class_overlay(mask: np.ndarray, colour: tuple[float, float, float], alpha: float) -> np.ndarray:
    """RGBA image that is `colour` where `mask` is true and fully transparent elsewhere."""
    overlay = np.zeros((*mask.shape, 4))
    overlay[mask, :3] = colour
    overlay[mask, 3] = alpha
    return overlay


def draw_rgb_panel(ax: Axes, rgb: np.ndarray) -> None:
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title("Night Microphysics RGB")


def draw_hatched_mask(ax: Axes, mask: np.ndarray) -> None:
    """Hatch the pixels of `mask`, so ignored pixels stay recognisable where the RGB is grey."""
    # A false border closes the contours at the image edge; contours of 0/1 values run exactly
    # halfway between pixel centres, i.e. along the pixel edges.
    padded = np.pad(mask.astype(float), 1)
    rows = np.arange(-1, mask.shape[0] + 1)
    columns = np.arange(-1, mask.shape[1] + 1)
    ax.contourf(columns, rows, padded, levels=[0.5, 1.5], colors="none", hatches=[IGNORE_HATCH])
    ax.set_xlim(-0.5, mask.shape[1] - 0.5)
    ax.set_ylim(mask.shape[0] - 0.5, -0.5)


def draw_labels_panel(ax: Axes, rgb: np.ndarray, labels: np.ndarray) -> None:
    ax.imshow(rgb, interpolation="nearest")
    is_fog = labels == FOG
    is_ignored = labels == IGNORE
    ax.imshow(class_overlay(is_ignored, IGNORE_COLOUR, IGNORE_ALPHA), interpolation="nearest")
    ax.imshow(class_overlay(is_fog, FOG_COLOUR, FOG_ALPHA), interpolation="nearest")
    with plt.rc_context({"hatch.color": "white", "hatch.linewidth": 0.8}):
        draw_hatched_mask(ax, is_ignored)

    legend_items = [
        Patch(facecolor=FOG_COLOUR, alpha=FOG_ALPHA, label="fog (1)"),
        Patch(facecolor=IGNORE_COLOUR, edgecolor="white", hatch=IGNORE_HATCH, label="ignore (255)"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_title("Labels over the RGB")


def histogram_edges(dynamic: DynamicParams) -> np.ndarray:
    """Bin edges of the BTD histogram, the same bins the dynamic threshold uses."""
    low, high = dynamic.bin_range
    return np.arange(low, high + dynamic.bin_width / 2, dynamic.bin_width)


def draw_histogram_panel(
    ax: Axes, btd: np.ndarray, threshold: float, source: str, dynamic: DynamicParams
) -> None:
    ax.hist(btd, bins=histogram_edges(dynamic), color="steelblue")
    ax.axvline(threshold, color="red", linewidth=2, label=f"threshold {threshold:.2f} K")
    if btd.size == 0:
        ax.text(0.5, 0.5, "no histogram pixels", transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("IR_108 - IR_039 (K)")
    ax.set_ylabel("pixels")
    ax.legend(fontsize=8)
    ax.set_title(histogram_title(threshold, source))


def draw_sza_panel(ax: Axes, sza: np.ndarray, night_min_sza: float) -> None:
    image = ax.imshow(sza, cmap="cividis", interpolation="nearest")
    ax.figure.colorbar(image, ax=ax, label="degrees", shrink=0.8)

    has_values = np.isfinite(sza).any()
    # A contour needs the level inside the data range; a scene fully in night or day has no line.
    if has_values and np.nanmin(sza) < night_min_sza < np.nanmax(sza):
        ax.contour(sza, levels=[night_min_sza], colors="red", linewidths=1.5)
    ax.set_title(f"Solar zenith angle (red: {night_min_sza:g} deg)")


def scene_channels(dataset: xr.Dataset, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IR_039, IR_108 and IR_120 of one scene as 2-D arrays."""
    bt = dataset["bt"].isel(time=index)
    ir039 = bt.sel(channel="IR_039").values
    ir108 = bt.sel(channel="IR_108").values
    ir120 = bt.sel(channel="IR_120").values
    return ir039, ir108, ir120


def inspect_figure(
    dataset: xr.Dataset,
    labels: xr.Dataset,
    wanted: pd.Timestamp,
    params: LabelParams,
    dynamic: DynamicParams,
    out_dir: Path,
) -> Path:
    """Write the four-panel check figure of the scene at `wanted` and return its path."""
    # Labels of another dataset build would be drawn over pixels they were not made from.
    check_labels_match(dataset, labels)
    scene = find_scene(pd.DatetimeIndex(dataset["time"].values), wanted)

    ir039, ir108, ir120 = scene_channels(dataset, scene)
    sza = dataset["sza"].isel(time=scene).values
    label_image = labels["label"].isel(time=scene).values
    threshold = float(labels["btd_threshold"].isel(time=scene).values)
    source = str(labels["threshold_source"].isel(time=scene).values)
    mode = str(labels.attrs["mode"])

    rgb = night_microphysics_rgb(ir039, ir108, ir120)
    btd = histogram_pixels(ir039, ir108, ir120, sza, params)

    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
    try:
        draw_rgb_panel(axes[0, 0], rgb)
        draw_labels_panel(axes[0, 1], rgb, label_image)
        draw_histogram_panel(axes[1, 0], btd, threshold, source, dynamic)
        draw_sza_panel(axes[1, 1], sza, params.night_min_sza)
        fig.suptitle(f"{wanted.strftime(TIME_FORMAT)} UTC, {mode} labels")
        fig.tight_layout()

        out_dir.mkdir(parents=True, exist_ok=True)
        path = figure_path(out_dir, mode, wanted)
        fig.savefig(path, dpi=FIGURE_DPI)
    finally:
        plt.close(fig)
    return path
