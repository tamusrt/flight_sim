"""Unprofiled wall-clock breakdown of one flight."""
import time, timeit, statistics
import numpy as np
import bench
from flight_sim import integration as I
from flight_sim import flight_event as FE
from flight_sim.__main__ import get_default_properties, get_default_state
from flight_sim.environment.launch_rail import LaunchRail
from flight_sim.units import scalar, vector

acc = {"rates": 0.0, "crossed": 0.0, "n_rates": 0}
orig_rates = I._state_rates
def timed_rates(*a):
    t = time.perf_counter(); r = orig_rates(*a); acc["rates"] += time.perf_counter() - t; acc["n_rates"] += 1; return r
orig_crossed = FE.FlightEvent.crossed
def timed_crossed(self, *a):
    t = time.perf_counter(); r = orig_crossed(self, *a); acc["crossed"] += time.perf_counter() - t; return r

def total():
    ts = []
    for _ in range(7):
        t = time.perf_counter(); bench.run(); ts.append(time.perf_counter() - t)
    return statistics.median(ts)

base = total()
I._state_rates = timed_rates; FE.FlightEvent.crossed = timed_crossed
acc.update(rates=0.0, crossed=0.0, n_rates=0); bench.run()
print(f"median flight wall time: {base*1000:.0f} ms")
print(f"  _state_rates: {acc['rates']*1000:.0f} ms over {acc['n_rates']} calls "
      f"(note: includes calls made from inside event checks)")
print(f"  event.crossed (incl. peak-vel derivative_computation): {acc['crossed']*1000:.0f} ms")
I._state_rates = orig_rates; FE.FlightEvent.crossed = orig_crossed

# Per-call micro timings
props = get_default_properties()
rail = LaunchRail(length=scalar(17.0, "ft"))
cfg = I.IntegrationConfiguration()
st = get_default_state()
st.velocity = vector((150.0, 3.0, 1.0), "m/s"); st.angular_velocity = vector((0.1, 0.2, 0.0), "rad/s")
inputs = I._step_inputs(st, props, cfg)
vals = I._pack(st)
def t(fn, n=3000):
    return min(timeit.repeat(fn, number=n, repeat=5)) / n * 1e6
print("\nper-call (us):")
print(f"  _state_rates             {t(lambda: I._state_rates(1.0, vals, inputs)):7.1f}")
print(f"  rkf45_step (6 rates)     {t(lambda: I.rkf45_step(1.0, vals, 0.01, inputs), 500):7.1f}")
print(f"  aero table lookup        {t(lambda: props.aero_table(0.5, 3.0, 45.0)):7.1f}")
print(f"  mass_properties(t)       {t(lambda: props.mass_properties(1.0)):7.1f}")
print(f"  atmosphere.conditions    {t(lambda: inputs.atmosphere.conditions(500.0)):7.1f}")
print(f"  body_to_world (r3f)      {t(lambda: I.body_to_world(st.orientation)):7.1f}")
M = props.mass_properties(1.0).inertia
print(f"  np.linalg.solve 3x3      {t(lambda: np.linalg.solve(M, vals[6:9])):7.1f}")
print(f"  _step_inputs (Pint)      {t(lambda: I._step_inputs(st, props, cfg)):7.1f}")
print(f"  _pack (Pint)             {t(lambda: I._pack(st)):7.1f}")
print(f"  _unpack (Pint)           {t(lambda: I._unpack(vals, st)):7.1f}")
print(f"  derivative_computation   {t(lambda: I.derivative_computation(1.0, st, props, cfg)):7.1f}")
print(f"  scalar(1.0,'s').m_as('s'){t(lambda: scalar(1.0,'s').m_as('s')):7.1f}")
