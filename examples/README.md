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

Each segment can also fly different physics by picking its own aero model.
`properties.aero_table` takes any subclass of `flight_sim.utilities.data_loader.AeroModel`:
it sets `reference_area_m2`, `reference_length_m` and `reference_point_m`, and
implements `coefficients(flow)`, which gets the Mach number, angles and motor thrust
and returns body-axes `AeroCoefficients`. The CSV table (`aero_table_from_csv`) is
power-off and ignores thrust.

```toml
properties.aero_table = { type = "my_models.ParachuteDrag", cd_area = "1.2 m**2" }
```

One clock runs through every segment, so `max_time`, thrust curves and `at_time`
events are mission times. A motor lit by an event that fired in an earlier segment
burns on. Segment `name` and `output` are ignored. See
`examples/sol_invictus/mission.toml`.

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
