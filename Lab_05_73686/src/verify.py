import numpy as np
import torch
from engine import Value

np.random.seed(0)
N = 6
a_np = np.random.randn(N)
b_np = np.random.randn(N)
c_np = np.random.randn(N)

a_v = [Value(x) for x in a_np]
b_v = [Value(x) for x in b_np]
c_v = [Value(x) for x in c_np]

terms = [((a_v[i] * b_v[i] + c_v[i]) ** 2).relu() for i in range(N)]
out = sum(terms)          # sum(): 0 + Value, __radd__ sayesinde çalışır
out.backward()

mine_a = np.array([v.grad for v in a_v])
mine_b = np.array([v.grad for v in b_v])
mine_c = np.array([v.grad for v in c_v])

a_t = torch.tensor(a_np, dtype=torch.float64, requires_grad=True)
b_t = torch.tensor(b_np, dtype=torch.float64, requires_grad=True)
c_t = torch.tensor(c_np, dtype=torch.float64, requires_grad=True)

loss = ((a_t * b_t + c_t) ** 2).relu().sum()
loss.backward()

torch_a = a_t.grad.numpy()
torch_b = b_t.grad.numpy()
torch_c = c_t.grad.numpy()

def f(a, b, c):
    return np.maximum((a * b + c) ** 2, 0.0).sum()

def numeric_grad(which, eps=1e-5):
    args = [a_np.copy(), b_np.copy(), c_np.copy()]
    g = np.zeros(N)
    for i in range(N):
        orig = args[which][i]
        args[which][i] = orig + eps
        fp = f(*args)
        args[which][i] = orig - eps
        fm = f(*args)
        args[which][i] = orig
        g[i] = (fp - fm) / (2 * eps)
    return g

num_a = numeric_grad(0)
num_b = numeric_grad(1)
num_c = numeric_grad(2)

def maxdiff(x, y):
    return float(np.max(np.abs(x - y)))

print("Forward: mine =", out.data, " torch =", loss.item())
print()
print("Max |mine - torch|")
print("  a:", maxdiff(mine_a, torch_a))
print("  b:", maxdiff(mine_b, torch_b))
print("  c:", maxdiff(mine_c, torch_c))
print()
print("Max |mine - numeric|")
print("  a:", maxdiff(mine_a, num_a))
print("  b:", maxdiff(mine_b, num_b))
print("  c:", maxdiff(mine_c, num_c))

worst = max(
    maxdiff(mine_a, torch_a), maxdiff(mine_b, torch_b), maxdiff(mine_c, torch_c),
    maxdiff(mine_a, num_a), maxdiff(mine_b, num_b), maxdiff(mine_c, num_c),
)
print()
print("Worst difference:", worst)
print("PASS" if worst < 1e-6 else "FAIL")