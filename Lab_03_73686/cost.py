import time
import statistics
import torch
from torchinfo import summary
from thop import profile
from models import get_models

def hand_flops():
    mlp_linear = 2 * 784 * 256
    cnn_conv = 2 * 3 * 3 * 16 * 32 * 14 * 14
    seq, d = 16, 64
    qkv = 2 * seq * d * 3 * d
    scores = 2 * seq**2 * d
    attn_v = 2 * seq**2 * d
    out_proj = 2 * seq * d * d
    attn_total = qkv + scores + attn_v + out_proj
    return {
        "MLP (Linear 784->256)": mlp_linear,
        "CNN (Conv 16->32, 14x14)": cnn_conv,
        "Attention (1 MHA blogu)": attn_total,
    }

def measure_latency(model, batch, runs, warmup=10):
    model.eval()
    x = torch.randn(batch, 1, 28, 28)
    times = []
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)

if __name__ == "__main__":
    models = get_models()

    print("\n===== torchinfo =====")
    for name, m in models.items():
        print(f"\n--- {name} ---")
        summary(m, input_size=(1, 1, 28, 28))

    print("\n===== Elle hesaplanan FLOPs (tek katman) =====")
    for k, v in hand_flops().items():
        print(f"{k:30s} {v:,}")

    print("\n===== thop (tum model, batch=1) =====")
    for name, m in models.items():
        m.eval()
        macs, params = profile(m, inputs=(torch.randn(1, 1, 28, 28),), verbose=False)
        print(f"{name:10s} MACs: {int(macs):,}   FLOPs(=2xMACs): {int(2*macs):,}   params: {int(params):,}")

    print("\n===== Latency (ms, median, CPU) =====")
    for name, m in models.items():
        l1 = measure_latency(m, 1, 100)
        l128 = measure_latency(m, 128, 20)
        print(f"{name:10s} batch=1: {l1:.3f} ms   batch=128: {l128:.3f} ms")