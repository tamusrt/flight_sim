"""Figures of a campaign's outputs.

Every public function draws on ``ax``, or on a new figure's axes without one,
and returns the axes; none shows the figure. ``limits`` draws a requirement's
bounds and ``reference`` a single value, such as the undispersed flight's,
flown on its own with ``flight_sim run``, to compare with the trials.
"""

import math
from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse
from numpy.typing import ArrayLike
from scipy import stats  # type: ignore[import-untyped]

from edith.statistics.outputs import Envelope, OutputDistribution

__all__ = ["plot_distribution", "plot_envelope", "plot_scatter", "plot_sensitivity"]

Limits = tuple[float | None, float | None]


def plot_distribution(
    output: OutputDistribution,
    label: str = "",
    *,
    limits: Limits | None = None,
    reference: float | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Draw a histogram of an output, as a density.

    Args:
        output (OutputDistribution): The output.
        label (str, optional): The x axis's label, such as "apogee (m)".
        limits (Limits | None, optional): A requirement's low and high, drawn
            as vertical lines. Defaults to none.
        reference (float | None, optional): A value drawn as a vertical line.
            Defaults to none.
        ax (Axes | None, optional): Where to draw. Defaults to new axes.

    Returns:
        Axes: The axes drawn on.
    """
    ax = _axes(ax)
    ax.hist(output.values, bins="auto", density=True, alpha=0.7)
    _lines(ax.axvline, limits, reference)
    ax.set_xlabel(label)
    ax.set_ylabel("density")
    return ax


def plot_envelope(
    envelope: Envelope,
    quantiles: Sequence[float] = (5, 50, 95),
    label: str = "",
    *,
    limits: Limits | None = None,
    reference: float | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Draw percentile curves of an output over time, the outer ones as a band.

    Args:
        envelope (Envelope): The output over time.
        quantiles (Sequence[float], optional): Percentiles, 0 to 100, in
            increasing order; the first and last bound the band. Defaults to
            the 5th, 50th and 95th.
        label (str, optional): The y axis's label, such as "altitude (m)".
        limits (Limits | None, optional): A requirement's low and high, drawn
            as horizontal lines. Defaults to none.
        reference (float | None, optional): A value drawn as a horizontal
            line. Defaults to none.
        ax (Axes | None, optional): Where to draw. Defaults to new axes.

    Returns:
        Axes: The axes drawn on.
    """
    ax = _axes(ax)
    curves = {q: envelope.percentile(q) for q in quantiles}
    first, last = quantiles[0], quantiles[-1]
    ax.fill_between(
        envelope.times,
        curves[first],
        curves[last],
        alpha=0.3,
        label=f"{first:g}th to {last:g}th percentile",
    )
    for q in quantiles[1:-1]:
        ax.plot(envelope.times, curves[q], label=f"{q:g}th percentile")
    _lines(ax.axhline, limits, reference)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(label)
    ax.legend()
    return ax


def plot_sensitivity(
    correlation: pd.DataFrame, output: str, *, ax: Axes | None = None
) -> Axes:
    """Draw a tornado chart of the inputs' rank correlations with one output.

    Args:
        correlation (pd.DataFrame): ``Sensitivity.correlation``, one row per
            input.
        output (str): Its column to draw.
        ax (Axes | None, optional): Where to draw. Defaults to new axes.

    Returns:
        Axes: The axes drawn on, the strongest input at the top.
    """
    ax = _axes(ax)
    column = correlation[output].dropna()
    ordered = column.reindex(column.abs().sort_values().index)
    colours = ["tab:blue" if value >= 0.0 else "tab:orange" for value in ordered]
    ax.barh([str(key) for key in ordered.index], ordered.to_numpy(), color=colours)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel(f"Spearman correlation with {output}")
    return ax


def plot_scatter(
    x: ArrayLike,
    y: ArrayLike,
    confidence: float = 0.95,
    labels: tuple[str, str] = ("", ""),
    *,
    reference: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Draw points, such as landing points, and their covariance ellipse.

    The ellipse holds ``confidence`` of a normal spread of the points' mean and
    covariance. Points with NaN are left out.

    Args:
        x (ArrayLike): Each point's x, such as east.
        y (ArrayLike): Each point's y, such as north.
        confidence (float, optional): Of the ellipse. Defaults to 0.95.
        labels (tuple[str, str], optional): The axes' labels.
        reference (tuple[float, float] | None, optional): A point drawn as a
            cross, such as the undispersed flight's. Defaults to none.
        ax (Axes | None, optional): Where to draw. Defaults to new axes.

    Returns:
        Axes: The axes drawn on, at equal scale.
    """
    ax = _axes(ax)
    points = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    points = points[np.isfinite(points).all(axis=1)]
    ax.scatter(points[:, 0], points[:, 1], s=8, alpha=0.6)
    if len(points) > 2:
        variances, axes = np.linalg.eigh(np.cov(points, rowvar=False))
        variances, axes = variances[::-1], axes[:, ::-1]  # The major axis first
        radius = math.sqrt(float(stats.chi2.ppf(confidence, 2)))
        width, height = 2.0 * radius * np.sqrt(np.maximum(variances, 0.0))
        angle = math.degrees(math.atan2(axes[1, 0], axes[0, 0]))
        ax.add_patch(
            Ellipse(
                (float(points[:, 0].mean()), float(points[:, 1].mean())),
                float(width),
                float(height),
                angle=angle,
                fill=False,
                color="tab:red",
                label=f"{confidence:.0%} ellipse",
            )
        )
        ax.legend()
    if reference is not None:
        ax.plot(*reference, marker="x", markersize=10, color="black", linestyle="")
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_aspect("equal", adjustable="datalim")
    return ax


def _axes(ax: Axes | None) -> Axes:
    """Return the axes given, or those of a new figure."""
    if ax is not None:
        return ax
    _, new = plt.subplots()
    return new


def _lines(
    line: Callable[..., object], limits: Limits | None, reference: float | None
) -> None:
    """Draw a requirement's limits, dashed red, and a reference value, dotted.

    ``line`` is ``ax.axvline`` or ``ax.axhline``.
    """
    for limit in limits or ():
        if limit is not None:
            line(limit, color="tab:red", linestyle="--", linewidth=1.0)
    if reference is not None:
        line(reference, color="black", linestyle=":", linewidth=1.5)
