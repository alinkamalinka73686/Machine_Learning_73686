import sys
import time
import statistics
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import make_moons
from engine import Value

sys.setrecursionlimit(20000)

N_SAMPLES = 200
EPOCHS = 100
LR = 0.5
RUNS = 5
HIDDEN = 8

X, y = make_moons(n_samples=N_SAMPLES, noise=0.1, random_state=0)


rng = np.random.RandomState(42)
W1_0 = rng.randn(HIDDEN, 2) * 0.5
b1_0 = np.zeros(HIDDEN)
W2_0 = rng.randn(HIDDEN) * 0.5
b2_0 = 0.0


def train_mine():
    W1 = [[Value(W1_0[i, j]) for j in range(2)] for i in range(HIDDEN)]
    b1 = [Value(b1_0[i]) for i in range(HIDDEN)]
    W2 = [Value(W2_0[i]) for i in range(HIDDEN)]
    b2 = Value(b2_0)
    params = [p for row in W1 for p in row] + b1 + W2 + [b2]

    def forward(x):
        hidden = []
        for i in range(HIDDEN):
            h = W1[i][0] * x[0] + W1[i][1] * x[1] + b1[i]
            hidden.append(h.relu())
        z = b2
        for i in range(HIDDEN):
            z = z + W2[i] * hidden[i]
        return z.sigmoid()

    eps = 1e-9
    for epoch in range(EPOCHS):
        total = Value(0.0)
        for xi, yi in zip(X, y):
            p = forward(xi)
            loss_i = -(yi * (p + eps).log() + (1 - yi) * ((1 - p) + eps).log())
            total = total + loss_i
        loss = total * (1.0 / N_SAMPLES)

        for p_ in params:
            p_.grad = 0.0
        loss.backward()
        for p_ in params:
            p_.data -= LR * p_.grad

    preds = [1 if forward(xi).data > 0.5 else 0 for xi in X]
    acc = float(np.mean(np.array(preds) == y))
    return loss.data, acc


def train_torch():
    Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    W1 = torch.tensor(W1_0, dtype=torch.float64, requires_grad=True)
    b1 = torch.tensor(b1_0, dtype=torch.float64, requires_grad=True)
    W2 = torch.tensor(W2_0, dtype=torch.float64, requires_grad=True)
    b2 = torch.tensor(b2_0, dtype=torch.float64, requires_grad=True)
    params = [W1, b1, W2, b2]

    def forward(x):
        h = torch.relu(x @ W1.T + b1)
        return torch.sigmoid(h @ W2 + b2)

    for epoch in range(EPOCHS):
        p = forward(Xt)
        loss = F.binary_cross_entropy(p, yt)
        for q in params:
            q.grad = None
        loss.backward()
        with torch.no_grad():
            for q in params:
                q -= LR * q.grad

    with torch.no_grad():
        preds = (forward(Xt) > 0.5).double()
        acc = float((preds == yt).double().mean())
    return loss.item(), acc


def benchmark(fn, name):
    times = []
    result = None
    for r in range(RUNS):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
        print(f"  {name} run {r + 1}: {times[-1]:.4f} s")
    med = statistics.median(times)
    print(f"{name}: median {med:.4f} s | final loss {result[0]:.4f} | accuracy {result[1]:.3f}\n")
    return med


print(f"Data: {N_SAMPLES} samples, {EPOCHS} epochs, lr={LR}, {RUNS} runs\n")
t_mine = benchmark(train_mine, "My engine")
t_torch = benchmark(train_torch, "PyTorch  ")
print(f"Speed ratio (mine / torch): {t_mine / t_torch:.1f}x slower")