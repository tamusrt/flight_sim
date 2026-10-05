# flight_sim
FS2.0 . For real this time!

## Getting Started

Follow these steps to set up your local development environment for the first time.

### 1. Prerequisites

You only need **Git** installed on your machine. You do not need to pre-install Python; `uv` will automatically download the correct Python version for you.

### 2. Install uv

Install `uv` using the official installer for your operating system:

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

*Note: Restart your terminal after installation to ensure the `uv` command is available in your PATH.*

### 3. Clone and Set Up the Project

Clone the repository and let `uv` automatically create a virtual environment and install all required dependencies:

```bash
# Clone the repository
git clone git@github.com:tamusrt/flight_sim.git
cd flight_sim

# Sync dependencies and set up the local environment
uv sync
```

---

## Development Workflows

You do not need to manually activate virtual environments. Prefix your commands with `uv run` to safely execute scripts within the project environment.

### Run the Application

To start the simulation:
```bash
uv run flight_sim [--inputs]
```

To run something in particular:

```bash
uv run python python_script.py
```

or

```bash
uv run python -m "from module import package; package()"
```

### Recovery versions (RECO) and EDITH

**RECO** is the part of the simulator that flies the descent: the parachutes opening, the flight computer's
decisions and the fall to the ground. It comes in versions that can be swapped, because every version takes the same
request and gives back the same outcome (`flight_sim.reco`):

| Version | What it is | Use it for |
| --- | --- | --- |
| `full` | The model as it has always been: shock cord, the rocket swinging under its canopy, small steps | One flight you want to look at closely (VISION) |
| `fast` (`FastRECO`) | The same descent without the details that barely move the landing point | Thousands of flights (EDITH) |

`FastRECO` keeps the flight computer, the drogue stage (full 3D motion, including a tumble) and the wind at every
height. It leaves out the shock cord bounce (a canopy opening is solved in one go, with its peak load), the swinging
under the main canopy (the fall is straight down, drifting with the wind) and small steps once the rocket falls at a
steady speed (steps grow, but never cover more than 25 m of height, so no wind layer is skipped). It is 40 to 150 times
faster (by rocket). How close it lands to the full model is checked, not promised: every EDITH batch flies a few
flights both ways (three typical ones and the three closest to an IREC limit, with the gusts switched off in both so
only the model difference shows) and reports the largest landing offset, load difference and whether the pass or fail
of the IREC checks agreed. The tests found landings within about 1% of the drift on the test rockets, which says
nothing for certain about another rocket. To add a version, write a subclass of `RECO` and add it with
`reco.register`; `reco.get_reco("fast", scheme)` finds one by name.

**EDITH** runs many flights with the inputs spread around what is expected at the launch site, and reports the
chance of each outcome. Run it with:

```bash
uv run python -m flight_sim.edith --rocket morpheus --out report.json
uv run python -m flight_sim.edith --rocket ork --ork design.ork --aero aero.csv --motor motor.eng
uv run python -m flight_sim.edith --write-site site.json   # write the site settings to edit
uv run python -m flight_sim.edith --site site.json         # run with the edited settings
```

What it does, in short:

- **Inputs.** Wind speed follows a Weibull curve cut at the launch limit (winds above it are not flown); the site's
  mean wind is the mean of the days that are flown, and the curve's scale is solved for that. The wind changes with
  height in random layers on every flight. Temperature, pressure, thrust, mass, drag, the launch rail and the
  parachute inputs follow bell curves (the rail's tilt is folded at vertical, so a vertical rail does not put half
  the flights exactly on vertical). All spreads are in `SiteConfig` (`edith/inputs.py`).
- **Sampling.** Flights are drawn in scrambled Sobol sets, which cover the possibilities more evenly than random draws.
- **Speed.** The first round flies the full 6-DOF climb. Its results train a surrogate that predicts the climb in the
  later rounds. Flights close to a limit are re-flown with the full climb, and a few random ones are re-flown to
  measure the surrogate's error. Descents use `FastRECO`. Flights run on every core, and give the same results on any
  number of cores. With `Settings(surrogate=False)` every climb is flown in full instead; the dynamics site does this,
  since a full climb of SRT14 takes about a second.
- **Results.** The chance of any failure, of any warning, of reaching the target apogee, and of each IREC check, the
  apogee, speeds and loads, and the landing footprint. **Every number has a 90% interval** (two-sided: its high end
  alone is a "95% sure it is below" bound). A batch stops when the two chances it watches, **any IREC failure** and
  **apogee at least the target**, each have an interval no wider than plus or minus 4% (`target_half_width`, default
  0.04), after the most rounds, or at the time limit. The other chances are not watched and can be wider. About 512
  flights are needed for plus or minus 4% on a chance near 50%.
- **Failures** are the IREC checks on the JARVIS predictions page (rail exit speed, stability, drogue descent rate,
  main deployment altitude) plus what would end an IREC flight: apogee outside 21,000 to 39,000 ft, the charge not
  separating, a canopy not opening, a canopy opened harder than its rating, no apogee, not landing, a simulation
  error, or a number the checks need that is missing or not a number. The landing speed and the main's descent rate
  are not judged here (use the JARVIS page's). There is no landing zone. A flight that stops with an error is
  recorded, counted as a failure and left out of the apogee numbers; the batch carries on and the report says how many
  and the first error.
- **Every batch reports a FastRECO check**: a few flights flown with both versions (see above), how far apart they
  landed, and a "model check" alert when any pair disagrees on pass or fail.
- **Over time.** For every fully flown climb the report keeps the altitude, Mach number and stability against time
  (as the average, the standard deviation and the lowest and highest flight at each moment) and each flight's path
  from the pad to the ground.

On the dynamics site, `python -m flight_sim.whatif.edith_site` runs EDITH after the JARVIS pages are built and adds
the EDITH page (the chances, charts of altitude, Mach and stability over time with their spread, and a picture of every
flight path; EDITH's results are only on that page, not on the JARVIS page), and to VISION the apogee and landing spread, with landing circles
(25, 50, 75 and 90% of landings) that can be turned on with the landing spread. It keeps its result between builds; see `tools/whatif/README.md` in dynamics.

**All the launch-site numbers are placeholders** (marked in the report and in the site file) until the team replaces
them with Spaceport America's climate for the launch week, motor test spreads and the canopy's real strength. The
canopy rating (15 g) is assumed, because the recovery model has none. The intervals describe sampling and surrogate
error only; they cannot tell you whether the assumed inputs are right.

### Run Unit Tests

We use `pytest` for testing. Run the entire test suite with:
```bash
uv run pytest
```

Coverage is enforced per settings in `pyproject.toml`, causing a fail if total coverage
drops below the set threshold.

### Contributing to the Codebase

1. Create a new branch and make your commits.
2. Make a PR and add a reviewer.
3. At PR time, branches need to pass a few checks. All of them run in GitHub Actions against
   every PR targeting `main`, and you can run the identical commands locally first:

   ```bash
   uv run ruff check .                   # a. Linting
   uv run ruff format --check .          # b. Formatting
   uv run pylint $(git ls-files '*.py')  # c. Pylint
   uv run mypy .                         # d. Type checking
   uv run pytest                         # e. Tests and coverage
   ```

   These work as written in both bash and PowerShell.

   Ruff can fix most of what it reports:

   ```bash
   uv run ruff check --fix .
   uv run ruff format .
   ```

   Every function needs a docstring, written either as a one-line summary or a full
   Google-style block. See [the docstring skill](.github/skills/write-docstrings/SKILL.md).

4. Push the branch and open a PR against `main`. The **Lint** and **Tests** workflows start
   automatically. The lint job runs all four of its checks even when an early one fails, so a
   single run reports everything you need to fix.

### Managing Dependencies

If you need to add new tools or libraries to the project during development:

```bash
# Add a production dependency
uv add package-name

# Add a development-only dependency (like a linter or formatter)
uv add --dev package-name
```

Make sure you update `uv.lock` and `pyproject.toml` updates after adding dependencies. It's
needed for the CI actions.
