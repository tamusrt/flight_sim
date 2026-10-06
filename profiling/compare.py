import time, timeit, statistics
import numpy as np, numba
import bench, fast_rhs
from flight_sim import integration as I
from flight_sim.__main__ import get_default_properties, get_default_state
from flight_sim.environment.launch_rail import LaunchRail
from flight_sim.units import scalar

props = get_default_properties()
rail = LaunchRail(length=scalar(17.0, "ft"), elevation=scalar(85.0, "deg"), friction_coefficient=0.2)
rng = np.random.default_rng(0)
import importlib.util
spec = importlib.util.spec_from_file_location("fast_jit", fast_rhs.__file__)
fj = importlib.util.module_from_spec(spec); spec.loader.exec_module(fj)
for name in ("cell", "atmosphere", "gravity", "rates"):
    setattr(fj, name, numba.njit(getattr(fj, name)))
jit_rates = fj.rates
py = fast_rhs

worst = 0.0
for on_rail in (False, True):
    st = rail.mount(get_default_state()) if on_rail else get_default_state()
    cfg = I.IntegrationConfiguration(truth=I.TruthConfiguration(launch_rail=rail if on_rail else None))
    inputs = I._step_inputs(st, props, cfg)
    P = py.params(inputs)
    for _ in range(300):
        y = I._pack(st).copy()
        y[:3] += rng.normal(0, 50, 3); y[0] = abs(y[0]) * 20
        y[3:6] = rng.normal(0, 150, 3); y[6:9] = rng.normal(0, 1, 3)
        q = rng.normal(size=4); y[9:13] = q / np.linalg.norm(q)
        t = rng.uniform(-0.5, 8.0)
        ref = I._state_rates(t, y, inputs)
        for fn in (py.rates, jit_rates):
            got = fn(t, y, np.empty(13), *P)
            worst = max(worst, float(np.max(np.abs(got - ref) / (np.abs(ref) + 1e-9))))
print(f"max relative difference vs _state_rates over 600 random states: {worst:.2e}")

st = get_default_state(); cfg = I.IntegrationConfiguration()
inputs = I._step_inputs(st, props, cfg); P = py.params(inputs)
y = I._pack(st); y[3:6] = (150, 3, 1); y[6:9] = (0.1, 0.2, 0); out = np.empty(13)
def us(fn, n=20000): return min(timeit.repeat(fn, number=n, repeat=5)) / n * 1e6
print(f"_state_rates (current)      {us(lambda: I._state_rates(1.0, y, inputs), 3000):7.2f} us")
print(f"flattened, pure Python      {us(lambda: py.rates(1.0, y, out, *P)):7.2f} us")
print(f"flattened, numba (from Py)  {us(lambda: jit_rates(1.0, y, out, *P)):7.2f} us")

# Whole flight with each RHS swapped in (everything else unchanged)
def flight_ms(rhs):
    cache = {}
    def wrapped(t, values, inputs):
        P = cache.get(id(inputs))
        if P is None: P = cache[id(inputs)] = (inputs, py.params(inputs))
        return rhs(t, values, np.empty(13), *P[1])
    orig = I._state_rates
    if rhs is not None: I._state_rates = wrapped
    try:
        ts = []
        for _ in range(7):
            t0 = time.perf_counter(); r = bench.run(); ts.append(time.perf_counter() - t0)
        return statistics.median(ts) * 1000, r
    finally:
        I._state_rates = orig
base, r0 = flight_ms(None); fpy, r1 = flight_ms(py.rates); fjit, r2 = flight_ms(jit_rates)
print(f"\nwhole flight: current {base:.0f} ms | pure-Python RHS {fpy:.0f} ms | numba RHS {fjit:.0f} ms")
print("impact time / downrange:", [f"{r[0]:.4f}s {r[2].position.m_as('m')[2]:.3f}m" for r in (r0, r1, r2)])
