"""Figure layout tests: shared and right-hand y axes."""

# The panel helpers are private to keep them out of the plot names
# pylint: disable=protected-access

from collections.abc import Iterator
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest

from flight_sim import plots
from flight_sim.units import ureg


class _Recorded:
    """Three points of made-up channels, with one event."""

    def __init__(self) -> None:
        """Make up the channels."""
        self.events = {"middle": 1}
        self._channels = {
            "time": ureg.Quantity(np.array([0.0, 1.0, 2.0]), "s"),
            "aero.alpha": ureg.Quantity(np.array([0.0, 0.1, 0.2]), "rad"),
            "aero.phi": ureg.Quantity(np.array([0.0, 1.0, 3.0]), "rad"),
            "aero.mach": ureg.Quantity(np.array([0.0, 0.5, 0.9]), ""),
            "thrust": ureg.Quantity(np.array([100.0, 50.0, 0.0]), "N"),
            "mass.mass": ureg.Quantity(np.array([25.0, 22.0, 20.0]), "kg"),
            "aero.force": ureg.Quantity(np.ones((3, 3)), "N"),
        }

    def __getitem__(self, channel: str) -> Any:
        """Return a made-up channel."""
        return self._channels[channel]


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    """Closes every figure a test leaves open."""
    yield
    plt.close("all")


def _single(panel: plots._Panel) -> list[Any]:
    """Draw one panel and return its axes, the twin last if there is one."""
    figure = plots._figure(_Recorded(), "Test", [panel])
    return figure.get_axes()


def test_series_on_one_axis_share_its_units() -> None:
    """A second series on an axis is drawn in the first one's units."""
    (axis,) = _single(
        plots._Panel(
            "Flow angles",
            [plots._Series("aero.alpha", units="deg"), plots._Series("aero.phi")],
        )
    )
    alpha, phi = axis.get_lines()[:2]

    assert axis.get_ylabel() == "Flow angles (deg)"
    assert phi.get_ydata()[-1] == pytest.approx(np.degrees(3.0))
    assert alpha.get_ydata()[-1] == pytest.approx(np.degrees(0.2))
    # Several series with no legend are named by channel
    assert [t.get_text() for t in axis.get_legend().get_texts()] == [
        "aero.alpha",
        "aero.phi",
    ]


def test_right_series_get_their_own_axis_and_colors() -> None:
    """A right-hand series gets a twin axis, its own units and the next color."""
    left, right = _single(
        plots._Panel(
            "Thrust",
            [
                plots._Series("thrust", ["Thrust"]),
                plots._Series("mass.mass", ["Mass"], right=True),
            ],
            right_label="Mass",
        )
    )
    # Data lines come before the event markers
    thrust = left.get_lines()[0]
    (mass,) = right.get_lines()

    assert left.get_ylabel() == "Thrust (N)"
    assert right.get_ylabel() == "Mass (kg)"
    assert thrust.get_color() != mass.get_color()
    # One legend, on the top axis, lists both sides
    assert left.get_legend() is None
    assert [t.get_text() for t in right.get_legend().get_texts()] == ["Thrust", "Mass"]


def test_incompatible_series_cannot_share_an_axis() -> None:
    """Series of different dimensions on one axis are rejected."""
    panel = plots._Panel(
        "Thrust", [plots._Series("thrust"), plots._Series("mass.mass")]
    )

    with pytest.raises(ValueError, match=r"mass\.mass cannot share the left axis"):
        _single(panel)


def test_single_series_panels_need_no_legend() -> None:
    """A lone scalar series has no legend, and a dimensionless one no units."""
    (axis,) = _single(plots._Panel("Mach", [plots._Series("aero.mach")]))

    assert axis.get_legend() is None
    assert axis.get_ylabel() == "Mach"


def test_vector_series_label_each_component() -> None:
    """Each component of a vector series is its own line and legend entry."""
    (axis,) = _single(
        plots._Panel(
            "Loads",
            [plots._Series("aero.force"), plots._Series("thrust")],
        )
    )

    assert [t.get_text() for t in axis.get_legend().get_texts()] == [
        "aero.force_x",
        "aero.force_y",
        "aero.force_z",
        "thrust",
    ]
