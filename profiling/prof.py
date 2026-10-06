import cProfile, pstats, time, sys, io
t0 = time.perf_counter()
import bench
t1 = time.perf_counter()
from flight_sim.__main__ import get_default_properties
get_default_properties()
t2 = time.perf_counter()
bench.run()  # warm
N = 10
t3 = time.perf_counter()
for _ in range(N): bench.run()
t4 = time.perf_counter()
print(f"imports {t1-t0:.3f}s | build properties {t2-t1:.3f}s | full run (warm) {(t4-t3)/N*1000:.1f} ms")
pr = cProfile.Profile(); pr.enable()
for _ in range(N): bench.run()
pr.disable()
s = io.StringIO()
p = pstats.Stats(pr, stream=s); p.sort_stats("tottime").print_stats(30)
p.sort_stats("cumulative").print_stats(40)
print(s.getvalue())
