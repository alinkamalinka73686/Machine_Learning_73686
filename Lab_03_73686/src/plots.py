import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from models import get_models
from cost import measure_latency

with open("results/results.json") as f:
    results = json.load(f)

models = get_models()
lat1, lat128 = {}, {}
for name, m in models.items():
    lat1[name] = measure_latency(m, 1, 100)
    lat128[name] = measure_latency(m, 128, 20)
    print(f"{name:10s} batch=1: {lat1[name]:.3f} ms   batch=128: {lat128[name]:.3f} ms")

colors = {"MLP": "tab:blue", "CNN": "tab:orange", "Attention": "tab:green"}
names = list(results.keys())


def scatter(ax, xs, xlabel, title, logx=False):
    for n in names:
        acc = results[n]["test_acc"] * 100
        ax.scatter(xs[n], acc, s=120, color=colors[n], label=n)
        ax.annotate(n, (xs[n], acc), textcoords="offset points", xytext=(8, 6))
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Plot 1: accuracy vs parameters
params = {n: results[n]["params"] for n in names}
scatter(axes[0], params, "Parameters", "Accuracy vs parameters", logx=True)

# Plot 2: accuracy vs latency (both batch sizes)
ax = axes[1]
for n in names:
    acc = results[n]["test_acc"] * 100
    ax.scatter(lat1[n], acc, s=120, color=colors[n], marker="o")
    ax.scatter(lat128[n], acc, s=120, color=colors[n], marker="s")
    ax.annotate(n, (lat128[n], acc), textcoords="offset points", xytext=(8, 6))
ax.set_xscale("log")
ax.set_xlabel("Latency (ms, log scale)")
ax.set_ylabel("Test accuracy (%)")
ax.set_title("Accuracy vs latency")
ax.grid(True, alpha=0.3)
ax.legend(handles=[
    Line2D([], [], marker="o", color="gray", linestyle="", label="batch 1"),
    Line2D([], [], marker="s", color="gray", linestyle="", label="batch 128"),
])

plt.tight_layout()
plt.savefig("results/tradeoff.png", dpi=150)
print("Saved tradeoff.png")
