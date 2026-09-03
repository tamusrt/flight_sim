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
