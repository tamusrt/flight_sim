"""Smoke tests of the figures of a campaign's outputs."""

import math
from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from matplotlib.patches import Ellipse, Rectangle

from edith.statistics.outputs import Envelope, OutputDistribution
from edith.statistics.plotting import (
    plot_distribution,
    plot_envelope,
    plot_scatter,
    plot_sensitivity,
)


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    """Close the figures each test draws."""
    yield
    plt.close("all")


def test_a_distribution_draws_its_limits_and_reference() -> None:
    """A histogram, two limit lines and a reference line."""
    output = OutputDistribution(np.random.default_rng(0).normal(3500, 20, 100))

    ax = plot_distribution(
        output, "apogee (m)", limits=(3450.0, None), reference=3510.0
    )

    assert ax.patches  # The histogram's bars
    assert [np.asarray(line.get_xdata())[0] for line in ax.get_lines()] == [
        3450.0,
        3510.0,
    ]
    assert ax.get_xlabel() == "apogee (m)"


def test_an_envelope_draws_a_band_and_its_middle_curves() -> None:
    """The outer percentiles are a band; each inner one is a line."""
    envelope = Envelope.from_runs(
        [([0.0, 1.0, 2.0], [0.0, 1.0, 0.0]), ([0.0, 1.0], [0.0, 2.0])], 0.5
    )
    _, ax = plt.subplots()

    drawn = plot_envelope(
        envelope, (10, 50, 90), "altitude (m)", limits=(None, 3.0), ax=ax
    )

    assert drawn is ax
    assert len(ax.collections) == 1
    assert len(ax.get_lines()) == 2  # The median and the limit


def test_a_tornado_puts_the_strongest_input_on_top() -> None:
    """Bars are sorted by size, so the last drawn, at the top, is the largest."""
    correlation = pd.DataFrame(
        {"apogee": [-0.6, 0.1, 0.3, math.nan]}, index=["mass", "cg", "rail", "fixed"]
    )

    ax = plot_sensitivity(correlation, "apogee")

    widths = [p.get_width() for p in ax.patches if isinstance(p, Rectangle)]
    assert widths == [0.1, 0.3, -0.6]


def test_a_scatter_draws_its_covariance_ellipse() -> None:
    """The ellipse is centred on the points' mean; NaN points are left out."""
    rng = np.random.default_rng(1)
    east = np.append(rng.normal(100.0, 30.0, 200), math.nan)
    north = np.append(rng.normal(-50.0, 10.0, 200), 0.0)

    ax = plot_scatter(east, north, reference=(110.0, -40.0))

    (ellipse,) = [patch for patch in ax.patches if isinstance(patch, Ellipse)]
    assert ellipse.center == pytest.approx((np.nanmean(east), north[:-1].mean()))
    assert ellipse.width > ellipse.height  # Wider east to west, as drawn
