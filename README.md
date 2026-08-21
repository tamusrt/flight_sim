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
curl -LsSf https://astral.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh | iex"
```

Restart your terminal after installation to ensure the `uv` command is available in your PATH.*

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

To start the simulator using the built-in package entry point:
```bash
uv run python -m flight_sim
```

### Run Unit Tests

We use `pytest` for testing. Run the entire test suite with:
```bash
uv run pytest
```

### Managing Dependencies

If you need to add new tools or libraries to the project during development:

```bash
# Add a production dependency
uv add package-name

# Add a development-only dependency (like a linter or formatter)
uv add --dev package-name
```
