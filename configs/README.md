# Sim input files

Each TOML table builds one object. Its keys are the keyword arguments.

Run a sim with `uv run flight_sim configs/sims/<name>.toml`. A sim file builds a
`flight_sim.simulation.Simulation`: `properties`, `config`, `events`,
`initial_state`, `initial_time_step`, `stop_event` and `name`.

| Form | Meaning |
| --- | --- |
| `type = "flight_sim.x.y.Name"` | Call this class, classmethod or function with the table's other keys. Required when the field's type is abstract (`Engine`, `WindModel`, ...) or to use a factory such as `SolidEngine.from_rse`. Otherwise the field's type hint picks the class. |
| `ref = "flight_sim.x.y.NAME"` | Use this existing object as-is, without calling it. |
| `include = "path.toml"` | Insert that file's table here. The other keys in the table override it. |
| `extends = "path.toml"` (top level) | Start from that file and override with this one. A table whose `type` changes replaces the base table instead of merging into it. |

- **Units are required** on every quantity: `"17 ft"`, `"85 deg"`, or `{ value = [x, y, z], units = "m" }` for vectors. Matrices use `{ value = [[...], ...], units = ... }` or `{ diag = [...], units = ... }`.
- **Paths** are relative to the file they appear in.
- **Extra keys** on a factory pass through its `**kwargs` to the class it builds, e.g. `ignition_time` on `SolidEngine.from_rse`.
- **Motor files** (`SolidEngine.from_rse`, `SolidEngine.from_eng`) need `forward_end_location`, the motor's forward end from the nose tip, e.g. `{ value = [-2.08, 0, 0], units = "m" }`. Aft is -X.
- **Ignition** is either a time, `ignition_time = "0 s"`, or an event, `ignition_time = { event = "rail exit", delay = "0.5 s" }`. The event is named, and must be one of the sim's `[[events]]` or the rail's `"rail exit"`. `delay` is optional and defaults to `"0 s"`.
- **Missing required arguments** are filled from already-built objects with the same name, so `peak_vertical_velocity` receives the sim's `properties` and `config`.

## Output

A sim's `[output]` table builds a `flight_sim.output.OutputConfiguration`. Without a
`directory` nothing is written.

```toml
[output]
directory = "../../out/{name}"   # {name} = sim name as a file name
formats = ["csv"]                # and/or "parquet" (needs: uv add pyarrow)
channels = ["*"]                 # globs over channel names, e.g. "aero.*", "state.position"
sample_interval = "0.01 s"       # omit for one row per integrator step
plots = ["trajectory", "attitude", "aero", "propulsion", "step_length"]
```

The directory gets `telemetry.csv`, `run.toml` (summary, event times, git commit),
`resolved_input.toml` (the merged input with absolute paths, which reruns as-is) and a
PNG per plot. `uv run flight_sim <sim> --output-dir <dir> --no-show` overrides the
directory and skips the plot windows.

From a script, `Simulation.from_file(path).run()` returns a `Flight`:
`flight["aero.force"]` is a quantity array, `flight.at("apogee")` a typed
`FlightPoint`, and `flight.to_dataframe(["aero.*"])` a table.

A channel is named by its attribute path in `FlightPoint`, so `flight["aero.force"]`
and `flight.at("apogee").aero.force` are the same quantity. Values that are no field
of it, such as speed and nose angles, are under `derived.*`.

```
configs/
├── sims/       one runnable flight each, including the launch rail
├── vehicles/   RocketProperties
└── sites/      TruthConfiguration minus the rail
```
