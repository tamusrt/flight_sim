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
faster (by rocket) and lands within about 1% of the full model. To add a version, write a subclass of `RECO` and add it with
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

- **Inputs.** Wind speed follows a Weibull curve (cut at the launch limit), and the wind changes with height in
  random layers on every flight. Temperature, pressure, thrust, mass, drag, the launch rail and the parachute inputs
  follow bell curves. All spreads are in `SiteConfig` (`edith/inputs.py`).
- **Sampling.** Flights are drawn in scrambled Sobol sets, which cover the possibilities more evenly than random draws.
- **Speed.** The first round flies the full 6-DOF climb. Its results train a surrogate that predicts the climb in the
  later rounds. Flights close to a limit are re-flown with the full climb, and a few random ones are re-flown to
  measure the surrogate's error. Descents use `FastRECO`. Flights run on every core, and give the same results on any
  number of cores.
- **Results.** The chance of any failure, of any warning, of reaching the target apogee, and of each IREC check, the
  apogee, speeds and loads, and the landing footprint. **Every number has a 90% interval.** A batch stops when the
  headline chances are within the target width (default plus or minus 3%), after the most rounds, or at the time limit.
- **Failures** are the IREC checks on the JARVIS predictions page (rail exit speed, stability, drogue and main
  descent rates, main altitude, landing speed) plus what would end an IREC flight: the charge not separating, a canopy
  not opening, a canopy opened harder than its rating, no apogee, not landing. There is no landing zone.
- **Every batch reports a FastRECO check**: a few flights flown with both versions, and how far apart they landed.

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
