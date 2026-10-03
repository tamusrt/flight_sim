"""Turn RASAero II "Run Test" alpha files into a flight_sim aero CSV.

Same converter as ``python -m flight_sim.whatif.ras_csv`` (that is where it lives now,
so the dynamics repo's Update CSV helper can use it). Run from this folder:

    python ras_to_csv.py                      # files next to this script
    python ras_to_csv.py --in DIR --out FILE  # or choose the paths
"""

from pathlib import Path

from flight_sim.whatif.ras_csv import main

if __name__ == "__main__":
    main(here=Path(__file__).resolve().parent)
