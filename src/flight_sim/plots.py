"""Figures of a recorded flight. ``[output] plots`` names them by function name.

Every public function here is a plot: it takes the flight and a title and
returns a Figure. Helpers stay private.
"""

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, Protocol

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pint import DimensionalityError

_WORLD_AXES = ("X (up)", "Y (east)", "Z (north)")
_BODY_AXES = ("Body X (roll)", "Body Y", "Body Z")


class ChannelSource(Protocol):
    """Recorded flight a figure reads channels and events from."""

    @property
    def events(self) -> Mapping[str, int]:
        """Return the row of each event."""

    def __getitem__(self, channel: str) -> Any:
        """Return one channel at every point, as a quantity array."""


class _Series(NamedTuple):
    """One channel drawn in a panel."""

    channel: str

    # One entry per component; with none, the series is labelled by channel
    # only when the panel holds several
    legend: Sequence[str] = ()

    # Units to plot in; defaults to those of the axis's first series, else the
    # channel's own
    units: str | None = None
    right: bool = False  # Draw against the panel's right-hand y axis


class _Panel(NamedTuple):
    """One panel of a figure: one or more channels against time.

    Series on the same side share that y axis and are drawn in its units.
    """

    label: str  # Left y axis
    series: Sequence[_Series]
    right_label: str = ""  # Right y axis, for series drawn against it


def _axis_label(label: str, units: Any) -> str:
    """Return an axis label with its units, leaving out dimensionless ones."""
    text = f"{units:~P}" if units is not None else ""
    return f"{label} ({text})" if text else label


def _default_legend(series: _Series, components: int) -> list[str]:
    """Return legend entries naming a series by its channel."""
    if components == 1:
        return [series.channel]
    suffixes = "xyz" if components == 3 else [str(i) for i in range(components)]
    return [f"{series.channel}_{suffix}" for suffix in suffixes]


def _series_values(
    flight: ChannelSource, series: _Series, axis_units: Any, panel: _Panel
) -> Any:
    """Return a series' values in its own units, else in those of its axis.

    Raises:
        ValueError: If the values cannot be converted to the axis's units.
    """
    values = flight[series.channel]
    units = series.units if series.units is not None else axis_units
    if units is None:
        return values
    try:
        return values.to(units)
    except DimensionalityError as error:
        side = "right" if series.right else "left"
        raise ValueError(
            f"{series.channel} cannot share the {side} axis of panel "
            f"{panel.label!r}, which is in {units}"
        ) from error


def _draw_panel(
    axis: Axes, flight: ChannelSource, time: np.ndarray, panel: _Panel
) -> None:
    """Draw every series of a panel, adding a right-hand axis if one needs it.

    Raises:
        ValueError: If a series cannot be converted to the units of its axis.
    """
    right = axis.twinx() if any(series.right for series in panel.series) else None
    axis_units: dict[bool, Any] = {}
    lines: list[Any] = []
    labels: list[str] = []
    for series in panel.series:
        values = _series_values(flight, series, axis_units.get(series.right), panel)
        axis_units.setdefault(series.right, values.units)
        magnitudes = values.magnitude.reshape(len(time), -1)
        names = list(series.legend)
        if not names and len(panel.series) > 1:
            names = _default_legend(series, magnitudes.shape[1])
        target = right if series.right and right is not None else axis
        for index in range(magnitudes.shape[1]):
            # A twin axis restarts the color cycle, so continue it by hand
            (line,) = target.plot(
                time, magnitudes[:, index], linewidth=2, color=f"C{len(lines) % 10}"
            )
            lines.append(line)
            labels.append(names[index] if index < len(names) else "")
    axis.set_ylabel(_axis_label(panel.label, axis_units.get(False)))
    if right is not None:
        right.set_ylabel(_axis_label(panel.right_label, axis_units.get(True)))
    labelled = [
        (line, label) for line, label in zip(lines, labels, strict=True) if label
    ]
    if labelled:
        handles, texts = zip(*labelled, strict=True)
        # On the top axis, so no line is drawn over it
        (right if right is not None else axis).legend(handles, texts)
    axis.grid(True)


def _mark_events(
    panels: Sequence[Axes], flight: ChannelSource, time: np.ndarray
) -> None:
    """Draw a dotted line at each event on every panel, named on the top one."""
    for name, row in flight.events.items():
        for panel in panels:
            panel.axvline(time[row], color="gray", linestyle=":", linewidth=1)
        panels[0].annotate(
            name,
            (time[row], 1.0),
            xycoords=("data", "axes fraction"),
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
            color="gray",
        )


def _figure(flight: ChannelSource, title: str, panels: Sequence[_Panel]) -> Figure:
    """Build a figure of stacked panels sharing the time axis.

    Args:
        flight (ChannelSource): Recorded flight.
        title (str): Figure title.
        panels (Sequence[_Panel]): Panels, top to bottom.

    Returns:
        Figure: The figure, with the events marked.

    Raises:
        ValueError: If a series cannot be converted to the units of its axis.
    """
    figure, axes = plt.subplots(
        len(panels), 1, figsize=(10, 3 * len(panels)), sharex=True, squeeze=False
    )
    time = flight["time"].m_as("s")
    for axis, panel in zip(axes[:, 0], panels, strict=True):
        _draw_panel(axis, flight, time, panel)
    axes[-1, 0].set_xlabel("Time (s)")
    axes[0, 0].set_title(title)
    _mark_events(list(axes[:, 0]), flight, time)
    figure.tight_layout()
    return figure


def trajectory(flight: ChannelSource, title: str) -> Figure:
    """Return position and velocity in world axes."""
    return _figure(
        flight,
        f"{title} - Trajectory",
        [
            _Panel("Position", [_Series("state.position", _WORLD_AXES)]),
            _Panel("Velocity", [_Series("state.velocity", _WORLD_AXES)]),
        ],
    )


def attitude(flight: ChannelSource, title: str) -> Figure:
    """Return where the nose points, the body rates and the flow angles."""
    return _figure(
        flight,
        f"{title} - Attitude",
        [
            _Panel("Nose elevation", [_Series("derived.nose_elevation")]),
            _Panel("Nose azimuth", [_Series("derived.nose_azimuth")]),
            _Panel("Angular rate", [_Series("state.angular_velocity", _BODY_AXES)]),
            _Panel(
                "Total angle of attack",
                [
                    _Series("aero.alpha", ["Total angle of attack"], "deg"),
                    _Series("aero.phi", ["Aerodynamic roll"], "deg", right=True),
                ],
                right_label="Aerodynamic roll",
            ),
        ],
    )


def aero(flight: ChannelSource, title: str) -> Figure:
    """Return Mach number, dynamic pressure and the aerodynamic loads."""
    return _figure(
        flight,
        f"{title} - Aerodynamics",
        [
            _Panel("Mach", [_Series("aero.mach")]),
            _Panel("Dynamic pressure", [_Series("aero.dynamic_pressure")]),
            _Panel("Aero force", [_Series("aero.force", _BODY_AXES)]),
            _Panel("Aero torque about CG", [_Series("aero.torque", _BODY_AXES)]),
        ],
    )


def propulsion(flight: ChannelSource, title: str) -> Figure:
    """Return thrust, mass and the CG location."""
    return _figure(
        flight,
        f"{title} - Propulsion and mass",
        [
            _Panel("Thrust", [_Series("thrust")]),
            _Panel("Mass", [_Series("mass.mass")]),
            _Panel("CG from nose", [_Series("mass.cg_location", ("X", "Y", "Z"))]),
        ],
    )


def step_length(flight: ChannelSource, title: str) -> Figure:
    """Return the length of every step the integrator took."""
    figure, axes = plt.subplots()
    axes.plot(np.diff(flight["time"].m_as("s")), linewidth=2)
    axes.set_xlabel("Step (-)")
    axes.set_ylabel("dt taken (s)")
    axes.set_title(f"{title} - Step length")
    axes.grid(True)
    figure.tight_layout()
    return figure
