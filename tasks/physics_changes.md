# Per-segment physics: what changed and why

Compared with `origin/hotfix/remove-defaults`. Includes the uncommitted `motion` change.

## 1. How a run works

1. `Simulation.run` loops: take one RKF45 step, record a flight point, repeat until the stop event.
2. Each step, the integrator asks `_state_rates` for the rates, which calls `_loads(time, values, inputs)`.
3. `_loads` works out the surroundings from the state: altitude, `to_world` (from the quaternion), `mass_properties(time)`, air `conditions`, `wind` and `gravity`.
4. It hands them to the physics:
   `physics.motion(time, velocity, angular_velocity, to_world, conditions, wind, gravity, mass, properties)`
5. The physics returns `Motion(acceleration, angular_acceleration, aero, thrust)`.
   - `acceleration`: world frame, m/s².
   - `angular_acceleration`: body axes, rad/s².
   - `aero`, `thrust`: for telemetry.
   - The default `motion` calls `thrust` and `aero`, then applies Newton and Euler.
6. On the rail, `_rail_acceleration` holds the rocket to the rail and angular acceleration is zero.
7. The rates go back to the integrator, which takes the step.
8. When an event fires (and is not the stop event), the sim calls `physics.after(event, point)` and flies on with the physics it returns.
9. `Mission.run` flies each segment in turn. Each one starts from the last one's state, time, step size and fired events.

## 2. Minor improvements

- A later segment starts with the step size the earlier one would have taken next, so a split flight is bit-identical to one run.
- The rocket no longer has to start on the rail. Only the first segment can have a rail. Later ones use `NoRail`.
- A motor lit by an event in an earlier segment keeps burning.
- Physics can change at events without changing `config`, so a sim can be run again.
- A warning when the aero loads don't match their coefficients, so telemetry can't silently disagree.
- Custom modules outside `flight_sim` are hashed in `run.toml`. A rerun warns if one changed.
- Aero math is shared through `coefficient_loads`, not copied.
- `AeroModel` was removed. `RocketPhysics` does its job.
- The ignition-event check moved to `run`, so an event fired in an earlier segment counts.
- A stop event that is already in the past fails at once, not at `max_time`.
- `events.burnout` event added.
- A badly wrapped torque line was fixed.

## 3. Adding a physics module

1. Create a file, e.g. `src/flight_sim/phases.py`.
2. Subclass `RocketPhysics` with `@dataclass(frozen=True)`. Override only what changes:
   - `thrust`: the motor's push.
   - `aero`: aero loads. Change the coefficients too, or you get a warning.
   - `motion`: replaces the equations of motion entirely.
   - `after`: switch behaviour when an event fires (e.g. a parachute).
3. If it should start at a point in the flight, add an event (e.g. `events.burnout`, `at_time`, or a `FlightEvent`).
4. Select it in TOML:
   `config.physics = { type = "flight_sim.phases.X", factor = 0.7 }`
5. Add a test, run the checks, and commit, so `run.toml` shows `dirty = false`.

## 4. Changes by file

### `src/flight_sim/physics.py` (new)

- Holds the force and motion code that used to be fixed inside `integration.py`. A segment can now swap it.
- `Motion`: what `motion` returns. A named tuple, so fields are clear.
- `RocketPhysics`: frozen dataclass so it can be built from TOML and copied with `replace`.
- `motion`: the old body of `_loads`, moved here so a phase can replace the equations of motion entirely. The math is unchanged.
- `thrust`: the old `engine.get_thrust` call, split out so a phase can change thrust alone.
- `after`: returns `self` by default. A phase overrides it to change at an event.
- `aero`: the old `_aero_loads`, moved here so a phase can change aero alone. It takes `time` and `thrust` so power-on or time-based effects are possible.
- `coefficient_loads`: coefficients → force and torque. Shared by `aero` and the mismatch check, so both use the same math.

### `src/flight_sim/integration.py`

- Import `RocketPhysics` and `coefficient_loads`: needed below.
- `aero_angles` and `cross` imports removed: their only user moved to `physics.py`.
- `IntegrationConfiguration.physics`: lets the config choose physics. Defaults to `RocketPhysics`, so old inputs still work.
- `_StepInputs.physics` and `physics=config.physics`: the step needs to know which physics to call.
- `_aero_loads` removed: moved to `RocketPhysics.aero`.
- `_loads`: the inline aero, thrust, Newton and Euler code was replaced by one `inputs.physics.motion(...)` call.
- `if inputs.rail is not None`: the rail now overrides whatever `motion` returns. Before, Euler only ran off the rail. Now it always runs, and the rail sets it to zero.
- `aero=motion.aero`, `thrust=motion.thrust`: telemetry records what the physics returned.
- `_check_coefficients`: warns when aero force or torque doesn't match its coefficients. Telemetry shows both, so they must agree.
- `flight_point_si` calls `_check_coefficients`: once per recorded point, not every integrator stage, to keep it cheap.

### `src/flight_sim/simulation.py`

- Import `replace` and `RocketPhysics`: used by `_after`.
- `__post_init__`: ignition check removed. A segment can be lit by an event that fired in an earlier segment, which isn't known until `run`.
- `run(start_time, fired_events)`: a mission segment starts later than 0 s and needs the events that already fired.
- `fired`, `names` and the ignition check: moved here, now counting fired events.
- `config = self.config`: a local copy, so physics changes during a run don't change the sim.
- `time = start_time`: start at the mission time.
- Lights the motor if its ignition event already fired, at that event's time plus the delay.
- `for name, point in fired.items(): config = _after(...)`: a segment's physics hears events from earlier segments, e.g. a chute deployed earlier stays deployed.
- `if fired: self._check_ahead(...)`: fails at once if the stop event is already behind.
- `flight_point_si(..., config)` and `step(..., config)`: use the local config.
- `config = _after(config, hit.name, points[-1])`: tell the physics an event fired.
- `next_time_step=dt`: pass the next step to `Flight` so a mission can carry it on.
- `_check_ahead`: a stop event already behind would never fire. Without this, it would run to `max_time`.
- `_after`: calls `physics.after`. Raises `TypeError` if it returns no physics, so a forgotten `return` fails clearly.
- `Flight.next_time_step`: keyword-only and required, so every `Flight` has it.

### `src/flight_sim/mission.py` (new)

- `Mission`: a list of segments flown in order.
- `from_file`: load from TOML, keeping the source for output.
- `__post_init__` checks:
  - At least one segment.
  - Only segment 0 can have a rail, because a later segment starts mid-flight.
  - Each motor's ignition event exists in its segment or an earlier one.
- `stop_event`: the last segment's. Output uses it.
- `run`:
  - Fails if a segment stopped while still on the rail. `NoRail` would wrongly release it.
  - `replace(segment, initial_state=..., initial_time_step=...)`: start from where the last segment stopped, with its step size, so a split matches one run exactly.
  - `segment.run(time, fired)`: carry the clock and fired events on.
  - `early` check: a later segment's stop event firing early means the segments are in the wrong order.
  - `skip`: a segment's first point repeats the last segment's end, so it is dropped.
  - Events, fired points and stats are merged across segments.
  - The final `Flight` carries the mission's name, output and source, so output is written as for one sim.

### `src/flight_sim/vehicle/engine.py`

- `Engine.burnout_time` (abstract): every engine must say when its burn ends, for `events.burnout`.
- `SolidEngine.burnout_time`: ignition time + last point of the thrust curve. `inf` until lit.

### `src/flight_sim/events.py`

- `burnout(properties, name, delay)`: event `delay` after burnout. A common place to switch phase.
- Reads the motor at each call, so a motor lit mid-flight is judged lit.
- `name` lets it be used twice, e.g. `"separation"`.
- Imports `Annotated` and `scalar`: for the `delay` unit and its default.

### `src/flight_sim/output.py`

- Imports `hashlib`, `sys`, `tomllib`, `warnings`, `Iterator`: used by hashing and the rerun check.
- `Mission` imported for type hints only. `prepare` accepts a `Mission`.
- `RUN_SUMMARY`: names `run.toml` once, as it is now both read and written.
- `write`: writes the input before `run.toml`, so a rerun is checked against the old summary before it is overwritten.
- `_run_table`: adds `code = {module: sha256}`. The git commit doesn't cover code outside `flight_sim`.
- `_write_input`: on a rerun, compares hashes and warns if a module changed.
- `endswith("output.directory")`: segments have their own output tables, which are skipped too.
- Drops segment `output` from the resolved input: it's ignored, and could hold machine paths.
- `_custom_modules`: finds module files named by `type`. Skips `flight_sim` and installed packages, which the commit and versions already pin.
- `_types`: collects every `type` string from the input.
- `_sha256`: hash a file.

### `src/flight_sim/__main__.py`

- `_load`: builds a `Mission` if the file has `segments`, otherwise a `Simulation`. Reads the file once and times the load.
- `fly`: accepts a `Mission`.

### `src/flight_sim/config/loading.py`

- `load_with_source(..., table=None)`: takes an already-read table, so `_load` doesn't read the file twice.

### `examples/sol_invictus/mission.toml` (new)

- `sim.toml` split into two segments: on the rail, then free. A working example that also tests the split.

### `examples/README.md`

- "Missions" section: how segments chain and what they inherit.
- "Phases with their own physics" section: `motion`, `thrust`, `aero` and `after`, with `PowerOnDrag` and `Parachute` examples.
- Notes on the coefficient warning, code hashing and `events.burnout`.

### Tests

- `conftest.py`: `_InertEngine.burnout_time` returns 0, as the new abstract method requires.
- `test_events.py`: `burnout` fires at curve end + delay.
- `test_examples.py`: every example mission flies each segment's stop event in order from its run directory, and writes no segment `output`.
- `test_output.py`: a custom module is hashed in `run.toml`, and a rerun warns once it changes.
- `test_simulation.py`:
  - The unknown-ignition test now expects the error from `run`, where the check moved.
  - New: a motor whose event fired before the start lights at that event's time.
- `test_mission.py` (new):
  - Split at rail exit / 3 s / apogee equals one run, exactly.
  - Motor lit in an earlier segment burns on.
  - A segment's physics (`_NoPowerOnDrag`) only affects that segment.
  - `after` (`_CutOff`) changes physics at an event and carries into later segments.
  - Torque that doesn't match its coefficients warns (`_Twisted`).
  - Replaced equations of motion (`_Vacuum`) give the vacuum parabola.
  - `after` returning nothing fails (`_Lost`).
  - Rejected: a rail on a later segment, an unknown ignition event, stopping on the rail, stop events out of order, a stop event already behind.
  - A thrust-based stop event sees a motor lit in an earlier segment.
