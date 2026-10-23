# Sim input files

Each TOML table builds one object. Its keys are the keyword arguments.

Run a sim with `uv run flight_sim examples/<name>/sim.toml`. A sim file builds a
`flight_sim.simulation.Simulation`: `properties`, `config`, `stop_event` and
`max_time`, with optional `events`, `initial_state`, `initial_time_step`, `name`
and `output`. A run whose stop event has not fired by `max_time` fails.

| Form | Meaning |
| --- | --- |
| `type = "flight_sim.x.y.Name"` | Call this class, classmethod or function with the table's other keys. Required when the field's type is abstract (`Engine`, `WindModel`, ...) or to use a factory such as `SolidEngine.from_rse`. Otherwise the field's type hint picks the class. |
| `ref = "flight_sim.x.y.NAME"` | Use this existing object as-is, without calling it. |
| `include = "path.toml"` | Insert that file's table here. The other keys in the table override it. |
| `extends = "path.toml"` (top level) | Start from that file and override with this one. A table whose `type` changes replaces the base table instead of merging into it. |

- **Units are required** on every quantity: `"17 ft"`, `"85 deg"`, or `{ value = [x, y, z], units = "m" }` for vectors. Matrices use `{ value = [[...], ...], units = ... }` or `{ diag = [...], units = ... }`.
- **Paths** are relative to the file they appear in.
- **Extra keys** on a factory pass through its `**kwargs` to the class it builds, e.g. `ignition_time` on `SolidEngine.from_rse`.
- **Motor files** (`SolidEngine.from_rse`, `SolidEngine.from_eng`) need `forward_end_location`, the motor's forward end from the nose tip, e.g. `{ value = [-2.08, 0, 0], units = "m" }`. Aft is -X. Every `SolidEngine` also needs `nozzle_exit_area` (`"0 m**2"` leaves out pressure thrust) and `reference_pressure`, the ambient pressure its thrust curve was measured at.
- **Launch site**: `config.truth` needs `launch_latitude`, `launch_elevation` and `launch_rail`. A rail needs `length` and `friction_coefficient`; `launch_rail = { type = "flight_sim.environment.launch_rail.NoRail" }` starts the rocket free instead.
- **Aero tables** need `frame` (`"body"` or `"missile"`) and `extrapolation` (`"linear"` or `"clamp"`) for points off the grid.
- **Ignition** is either a time, `ignition_time = "0 s"`, or an event, `ignition_time = { event = "rail exit", delay = "0.5 s" }`. The event is named, and must be one of the sim's `[[events]]` or the rail's `"rail exit"`. `delay` is optional and defaults to `"0 s"`.
- **Missing required arguments** are filled from already-built objects with the same name, so `peak_vertical_velocity` receives the sim's `properties` and `config`.
- **Checks**: unknown keys, missing required arguments, quantities without units or of the wrong dimension, and plain values of the wrong type (`friction_coefficient = "abc"`) are errors naming the key. A missing argument is reported against the callable used, so `SolidEngine.from_rse` asks for `forward_end_location` but never `grain`.
- **Defaults**: arguments left out take their defaults silently. `resolved_input.toml` lists each one, marked `# pulled from default`.

## Missions

A file with `[[segments]]` builds a `flight_sim.mission.Mission`: segments flown in
order, each a sim table that flies to its own `stop_event`, plus the mission's
`name` and `[output]`. Each later segment starts from the state, time and fired
events the one before stopped at, so each segment can have its own aero, motor or
mass. A later segment must have no rail (`NoRail`), and its stop event must still
be ahead of it, or the mission fails at once.

```toml
[[segments]]
include = "sim.toml"
stop_event = "rail exit"

[[segments]]
include = "sim.toml"
config.truth.launch_rail = { type = "flight_sim.environment.launch_rail.NoRail" }
properties.aero_table.filepath = "coast.csv"   # override only what changes
```

One clock runs through every segment, so `max_time`, thrust curves and `at_time`
events are mission times. A motor lit by an event that fired in an earlier segment
burns on. A later segment goes on with the step size the one before would have
taken, so its `initial_time_step` is unused, and segments that change nothing fly
exactly as one sim with the same events. Segment `name` and `output` are ignored.
See `examples/sol_invictus/mission.toml`.

### Phases with their own physics

`config.physics` picks the forces a segment flies, a `flight_sim.physics.RocketPhysics`
by default. A phase such as power-on drag or a parachute subclasses it and overrides
`thrust(time, conditions, properties)` or `aero(time, air_velocity, conditions, to_body,
cg_location, properties, thrust)`, which returns the body-axes `AeroPointSI` the
equations of motion and telemetry use. Numbers are SI.

```python
@dataclass(frozen=True)
class PowerOnDrag(RocketPhysics):
    factor: float = 0.8

    def aero(self, time, air_velocity, conditions, to_body, cg_location, properties, thrust):
        aero = super().aero(time, air_velocity, conditions, to_body, cg_location, properties, thrust)
        if thrust <= 0:
            return aero
        force = aero.force.copy()
        force[0] *= self.factor  # axial only: with the CG on the axis, the torque is unchanged
        c = replace(aero.coefficients, cx=aero.coefficients.cx * self.factor)
        return aero._replace(force=force, coefficients=c)
```

```toml
config.physics = { type = "my_models.PowerOnDrag", factor = 0.7 }
```

A phase that starts at an event, such as a parachute inflating after deployment,
overrides `after(event, point)`. The sim calls it with each event as it fires, and with
the events of earlier segments at the start, and flies the physics it returns. `point`
is the flight at the event (`time`, `state`, `aero`, ...), so the physics can depend on
the speed or attitude it fired at. `aero` gets the time, so a force can grow from then.

```python
@dataclass(frozen=True)
class Parachute(RocketPhysics):
    cd_area: float = 1.2  # m**2
    deployed_at: float = math.inf

    def after(self, event, point):
        return replace(self, deployed_at=point.time) if event == "deploy" else self
```

Change the coefficients with the loads, as above: a physics whose aero force or torque
is not what its `coefficients` give gets a warning, since telemetry would disagree.

A `type` from outside `flight_sim`, such as `my_models.PowerOnDrag`, is not pinned by
the commit in `run.toml`, so its module's SHA-256 is written there under `code`, and a
rerun from the run directory warns if the module has changed since.

`flight_sim.events.burnout` fires `delay` (default `"0 s"`) after the motor burns out;
give it a `name` to use it twice, e.g. a `"separation"` 1 s after burnout.

## Output

A sim's `[output]` table builds a `flight_sim.output.OutputConfiguration`. Without a
`directory` nothing is written.

```toml
[output]
directory = "out/{name}/{datetime}"   # sim name as a file name, and start time
formats = ["csv"]                # and/or "parquet"
units = "metric"                 # or "imperial": ft, lbf, psi, lb, degF; angles and times stay
channels = ["*"]                 # globs over channel names, e.g. "aero.*", "state.position"
sample_interval = "0.01 s"       # omit for one row per integrator step
plots = ["trajectory", "attitude", "aero", "propulsion", "step_length"]
```

`{datetime}` is the time the run starts, as `2026-10-22_140327`, or in a format of
your own after a colon: `{datetime:%Y%m%d}`. Preparing a directory that already
exists is an error unless `--overwrite` is passed.

`uv run flight_sim <sim>` first prepares the run directory: `resolved_input.toml`
(the merged input with every default filled in) and a copy of each data file it
reads under `inputs/`, which it names relative to itself. The sim then flies from
that file, so the directory holds exactly what ran and reruns anywhere with
`uv run flight_sim run <dir>`, writing back into itself. The run adds
`telemetry.csv` (or `.parquet`), `run.toml` (summary, event times, git commit) and a
PNG per plot under `plots/`. `--output-dir <dir> --no-show` overrides the directory and skips the
plot windows.

Each run prints, and `run.toml` keeps under `[diagnostics]`, the steps accepted and
rejected, RKF45 evaluations, event-search iterations, step length range and wall
times. `--profile` also times every function during the run, prints the slowest
and saves `profile.prof` (open it with `uvx snakeviz profile.prof`).

From a script, `Simulation.from_file(path).run()` returns a `Flight`:
`flight["aero.force"]` is a quantity array, `flight.at("apogee")` a typed
`FlightPoint`, `flight.to_dataframe(["aero.*"])` a table, and
`flight_sim.output.write(flight)` writes it as its `[output]` table says.
`flight_sim.output.prepare(simulation)` makes a run directory to load and fly
instead.
Channels, tables, plots and the summary are in `flight.units`, which the run takes
from `[output] units`; `dataclasses.replace(flight, units="imperial")` switches it.

A channel is named by its attribute path in `FlightPoint`, so `flight["aero.force"]`
and `flight.at("apogee").aero.force` are the same quantity. Values that are no field
of it, such as speed and nose angles, are under `derived.*`.

Each example is a folder holding everything its sim reads:

```
examples/sol_invictus/
├── sim.toml        the runnable sim, including the launch rail
├── vehicle.toml    RocketProperties, with the motor read from its .eng file
├── site.toml       TruthConfiguration minus the rail
├── aero.csv
└── Loki_M3464LB.eng
```
