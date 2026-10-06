"""Headless replica of flight_sim.__main__.main's integration loop."""
import sys, time as _t
from flight_sim.__main__ import get_default_properties, get_default_state
from flight_sim.environment.launch_rail import LaunchRail
from flight_sim.events import APOGEE, IMPACT, peak_vertical_velocity
from flight_sim.integration import IntegrationConfiguration, TruthConfiguration, adaptive_step
from flight_sim import integration
from flight_sim.units import scalar

calls = {"rates": 0}
_orig = integration._state_rates
def counted(*a):
    calls["rates"] += 1
    return _orig(*a)
if "--count" in sys.argv:
    integration._state_rates = counted

def run():
    properties = get_default_properties()
    rail = LaunchRail(length=scalar(17.0, "ft"), elevation=scalar(85.0, "deg"), azimuth=scalar(0.0, "deg"))
    state = rail.mount(get_default_state())
    config = IntegrationConfiguration(truth=TruthConfiguration(launch_rail=rail))
    events = (peak_vertical_velocity(properties, config), APOGEE, IMPACT)
    dt = scalar(0.01, "s"); t = 0.0; hit = None; steps = 0
    while hit is not IMPACT:
        state, dt_taken, dt, hit = adaptive_step(t, state, properties, config, dt, events=events)
        t += float(dt_taken.m_as("s")); steps += 1
    return t, steps, state

if __name__ == "__main__":
    t0 = _t.perf_counter()
    t, steps, state = run()
    el = _t.perf_counter() - t0
    print(f"flight {t:.3f}s, {steps} steps, wall {el:.2f}s, rates calls {calls['rates']}")
    print("final pos", state.position)
