import time
import statistics
import torch

torch.manual_seed(0)


def median_time(fn, runs=5, iters=50):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        times.append((time.perf_counter() - t0) / iters)
    return statistics.median(times)


A = torch.randn(2048, 2048)
B = torch.randn(2048, 2048)
At = A.t()

t1 = median_time(lambda: A + B)
t2 = median_time(lambda: At + B)
Atc = At.contiguous()
t3 = median_time(lambda: Atc + B)

print(f"A  + B (both contiguous):          {t1 * 1e3:.3f} ms")
print(f"At + B (mixed layout):             {t2 * 1e3:.3f} ms  ({t2 / t1:.2f}x)")
print(f"At.contiguous() + B (pre-copied):  {t3 * 1e3:.3f} ms")