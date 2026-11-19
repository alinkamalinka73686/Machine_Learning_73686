import time
import statistics
import torch
import torch.nn as nn

torch.manual_seed(0)


def median_time(fn, runs=5, iters=200):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        times.append((time.perf_counter() - t0) / iters)
    return statistics.median(times)


print("=== 1. Computation graph ===")
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid())
x = torch.randn(4, 2)
out = model(x)
print("output shape:", tuple(out.shape), "dtype:", out.dtype)


def walk(fn, depth=0, seen=None):
    seen = set() if seen is None else seen
    if fn is None or fn in seen:
        return
    seen.add(fn)
    print("  " * depth + type(fn).__name__)
    for nxt, _ in fn.next_functions:
        walk(nxt, depth + 1, seen)


walk(out.grad_fn)
print()

print("=== 2. Eager vs compiled ===")
big = nn.Sequential(
    nn.Linear(256, 512), nn.ReLU(),
    nn.Linear(512, 512), nn.ReLU(),
    nn.Linear(512, 1), nn.Sigmoid(),
)
xb = torch.randn(1024, 256)

with torch.no_grad():
    t_eager = median_time(lambda: big(xb))
print(f"eager:            {t_eager * 1e3:.3f} ms/iter")

compiled = None
label = None
try:
    compiled = torch.compile(big)
    with torch.no_grad():
        compiled(xb)  # ilk çağrı derlemeyi tetikler
    label = "torch.compile"
except Exception as e:
    print(f"torch.compile failed ({type(e).__name__}); falling back to torch.jit.trace")
    compiled = None

if compiled is None:
    with torch.no_grad():
        compiled = torch.jit.trace(big, xb)
        for _ in range(5):  # warm-up
            compiled(xb)
    label = "torch.jit.trace"

with torch.no_grad():
    for _ in range(10):  # warm-up
        compiled(xb)
    t_comp = median_time(lambda: compiled(xb))
print(f"{label}: {t_comp * 1e3:.3f} ms/iter")
print(f"speedup (eager / compiled): {t_eager / t_comp:.2f}x")
print()


print("=== 3. Tensor properties and contiguity ===")
A = torch.randn(2048, 2048)
At = A.t()  # kopyalamadan view: non-contiguous
print("A : shape", tuple(A.shape), "dtype", A.dtype, "stride", A.stride(), "contiguous", A.is_contiguous())
print("At: shape", tuple(At.shape), "dtype", At.dtype, "stride", At.stride(), "contiguous", At.is_contiguous())

t_contig = median_time(lambda: A + 1.0, iters=50)
t_noncontig = median_time(lambda: At + 1.0, iters=50)
t_sum_c = median_time(lambda: A.sum(dim=0), iters=50)
t_sum_nc = median_time(lambda: At.sum(dim=0), iters=50)
t_copy = median_time(lambda: At.contiguous(), iters=50)
print(f"A + 1       (contiguous):     {t_contig * 1e3:.3f} ms")
print(f"At + 1      (non-contiguous): {t_noncontig * 1e3:.3f} ms  ({t_noncontig / t_contig:.2f}x)")
print(f"A.sum(0)    (contiguous):     {t_sum_c * 1e3:.3f} ms")
print(f"At.sum(0)   (non-contiguous): {t_sum_nc * 1e3:.3f} ms  ({t_sum_nc / t_sum_c:.2f}x)")
print(f"At.contiguous() copy cost:    {t_copy * 1e3:.3f} ms")