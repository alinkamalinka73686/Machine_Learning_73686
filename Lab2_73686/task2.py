
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


digits = load_digits()

X = digits.data.astype(np.float64) / 16.0
y = digits.target
Y = np.eye(10)[y]

X_tr, X_te, Y_tr, Y_te, y_tr, y_te = train_test_split(
    X, Y, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

print("Training set:", X_tr.shape)
print("Test set:", X_te.shape)


def init_params(W, std=0.01, seed=0):
    rng = np.random.default_rng(seed)

    return {
        "W1": rng.normal(0, std, (64, W)),
        "b1": np.zeros(W),
        "W2": rng.normal(0, std, (W, W)),
        "b2": np.zeros(W),
        "W3": rng.normal(0, std, (W, 10)),
        "b3": np.zeros(10),
    }


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def forward(X, p):
    Z1 = X @ p["W1"] + p["b1"]
    A1 = np.maximum(0, Z1)

    Z2 = A1 @ p["W2"] + p["b2"]
    A2 = np.maximum(0, Z2)

    Z3 = A2 @ p["W3"] + p["b3"]
    A3 = softmax(Z3)

    return A3, (X, Z1, A1, Z2, A2, A3)


def cross_entropy(A3, Y):
    return -np.mean(
        np.sum(Y * np.log(A3 + 1e-12), axis=1)
    )


def loss_fn(X, Y, p):
    return cross_entropy(forward(X, p)[0], Y)


def accuracy(X, y, p):
    predictions = np.argmax(forward(X, p)[0], axis=1)
    return np.mean(predictions == y)


def backward(Y, p, cache):
    X, Z1, A1, Z2, A2, A3 = cache
    m = X.shape[0]

    dZ3 = (A3 - Y) / m
    dW3 = A2.T @ dZ3
    db3 = dZ3.sum(axis=0)

    dZ2 = (dZ3 @ p["W3"].T) * (Z2 > 0)
    dW2 = A1.T @ dZ2
    db2 = dZ2.sum(axis=0)

    dZ1 = (dZ2 @ p["W2"].T) * (Z1 > 0)
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0)

    return {
        "W1": dW1, "b1": db1,
        "W2": dW2, "b2": db2,
        "W3": dW3, "b3": db3
    }


p = init_params(16)
A3, _ = forward(X_tr[:32], p)

print("Output shape:", A3.shape)
print(
    "Probability rows sum to 1:",
    np.allclose(A3.sum(axis=1), 1, atol=1e-6)
)
print(
    "Untrained loss: %.4f (ln(10) = %.4f)"
    % (loss_fn(X_tr, Y_tr, p), np.log(10))
)


def gradient_check(p, Xb, Yb, n=10, eps=1e-5, seed=1):
    rng = np.random.default_rng(seed)

    _, cache = forward(Xb, p)
    grads = backward(Yb, p, cache)

    keys = ["W1", "W2", "W3"]
    ok = 0

    for i in range(n):
        k = keys[rng.integers(len(keys))]
        idx = tuple(
            int(rng.integers(0, s)) for s in p[k].shape
        )

        old = p[k][idx]

        p[k][idx] = old + eps
        Lp = loss_fn(Xb, Yb, p)

        p[k][idx] = old - eps
        Lm = loss_fn(Xb, Yb, p)

        p[k][idx] = old

        numerical = (Lp - Lm) / (2 * eps)
        analytical = grads[k][idx]

        rel_error = abs(analytical - numerical) / max(
            abs(analytical) + abs(numerical), 1e-12
        )

        passed = rel_error < 1e-6
        ok += passed

        print(
            f"{k}{idx}: analytical={analytical:.3e}, "
            f"numerical={numerical:.3e}, "
            f"relative_error={rel_error:.2e}, "
            f"{'OK' if passed else 'FAIL'}"
        )

    print(f"Gradient check: {ok}/{n} passed")
    print("PASSED" if ok >= 9 else "FAILED")


print("\nGradient checking:")
pc = init_params(16, std=0.5, seed=3)
gradient_check(pc, X_tr[:64], Y_tr[:64])


def train(W, lr, epochs=50, bs=32, seed=0):
    p = init_params(W, seed=seed)
    rng = np.random.default_rng(seed)

    n = X_tr.shape[0]
    losses = []

    for ep in range(epochs):
        indices = rng.permutation(n)

        for s in range(0, n, bs):
            batch = indices[s:s + bs]

            _, cache = forward(X_tr[batch], p)
            gradients = backward(Y_tr[batch], p, cache)

            for k in p:
                p[k] -= lr * gradients[k]

        epoch_loss = loss_fn(X_tr, Y_tr, p)
        losses.append(epoch_loss)

        print(
            f"W={W}, lr={lr}, "
            f"epoch={ep + 1}/{epochs}, "
            f"loss={epoch_loss:.4f}"
        )

    return p, losses


os.makedirs("results", exist_ok=True)

results = {}
fig, ax = plt.subplots(figsize=(7, 5))

print("\nTraining:")

for W in [16, 64]:
    for lr in [0.001, 0.01, 0.1]:
        p, losses = train(W, lr)

        test_acc = accuracy(X_te, y_te, p)
        results[(W, lr)] = test_acc

        ax.plot(
            range(1, len(losses) + 1),
            losses,
            label=f"W={W}, lr={lr}"
        )

ax.set_xlabel("Epoch")
ax.set_ylabel("Training cross-entropy loss")
ax.set_title("Training Loss Curves")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/loss_curves.png", dpi=150)
plt.close()

print("\nTest accuracy:")
print("W    Learning rate    Accuracy")

for (W, lr), test_acc in results.items():
    print(f"{W:<4} {lr:<16} {test_acc:.4f}")

print("\nPlot saved to results/loss_curves.png")


print("\nTiming results (median of 100 runs):")

for W in [16, 64]:
    p = init_params(W)

    batches = [
        ("batch=32", X_tr[:32], Y_tr[:32]),
        (f"batch={len(X_tr)}", X_tr, Y_tr)
    ]

    for name, Xb, Yb in batches:
        forward_times = []
        backward_times = []

        for _ in range(100):
            start = time.perf_counter()
            _, cache = forward(Xb, p)
            forward_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            backward(Yb, p, cache)
            backward_times.append(time.perf_counter() - start)

        median_forward = np.median(forward_times) * 1e6
        median_backward = np.median(backward_times) * 1e6

        print(
            f"W={W:<3} {name:<11} "
            f"forward={median_forward:.1f} us, "
            f"backward={median_backward:.1f} us, "
            f"ratio={median_backward / median_forward:.2f}x"
        )

print(
    "\nThe backward pass can take longer because it performs "
    "additional matrix multiplications to calculate gradients."
)
