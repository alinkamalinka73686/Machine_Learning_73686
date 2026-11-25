   

import argparse
import copy
import json
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32 * 7 * 7, 10))

    def forward(self, x):
        return self.classifier(self.features(x))


class AttentionNet(nn.Module):
    def __init__(self, d_model=64, nhead=4, dim_ff=128, num_layers=2):
        super().__init__()
        self.num_patches = 16
        self.proj = nn.Linear(49, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 10)

    def forward(self, x):
        B = x.shape[0]
        x = x.unfold(2, 7, 7).unfold(3, 7, 7).reshape(B, self.num_patches, 49)
        x = self.encoder(self.proj(x))
        return self.head(x.mean(dim=1))


MODELS = {"MLP": MLP, "CNN": CNN, "Attention": AttentionNet}


def load_cfg(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = path
    return cfg


def override(cfg, changes):
    c = copy.deepcopy(cfg)
    for k, v in changes.items():
        a, b = k.split(".")
        c[a][b] = v
    return c


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_loaders(cfg):
    d = cfg["data"]
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.2860,), (0.3530,))])
    full = datasets.FashionMNIST(d["root"], train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(d["root"], train=False, download=True, transform=tf)
    g = torch.Generator().manual_seed(cfg["seed"])
    train, val = random_split(full, [55000, 5000], generator=g)
    kw = dict(batch_size=d["batch_size"], num_workers=d["num_workers"],
              persistent_workers=d["num_workers"] > 0, pin_memory=(device == "cuda"))
    return (DataLoader(train, shuffle=True, **kw),
            DataLoader(val, shuffle=False, **kw),
            DataLoader(test, shuffle=False, **kw))


def build_opt(cfg, model):
    t = cfg["train"]
    if t["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=t["lr"],
                                weight_decay=t.get("weight_decay", 0.0))
    if t["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=t["lr"],
                               momentum=t.get("momentum", 0.9),
                               weight_decay=t.get("weight_decay", 0.0))
    raise ValueError(t["optimizer"])


def build_sched(cfg, opt):
    t = cfg["train"]
    n = t["scheduler"]
    if n == "none":
        return None
    if n == "step":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, t["epochs"] // 3), gamma=0.1)
    if n == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"])
    if n == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)
    raise ValueError(n)


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def mem_mb():
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 2 ** 20
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 20
    except ImportError:
        return float("nan")


def save_table(rows, path):
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(
            f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    text = "\n".join(lines)
    with open(path, "w") as f:
        f.write(text)
    print(text)


def run_training(cfg, run_name, accum=1, stop_after=None, resume=False):
    """Tek egitim run'i. YAML'dan okur, MLflow'a loglar, checkpoint kaydeder."""
    set_seed(cfg["seed"])
    tr, va, te = get_loaders(cfg)
    model = MODELS[cfg["model"]["name"]]().to(device)
    opt = build_opt(cfg, model)
    sched = build_sched(cfg, opt)
    loss_fn = nn.CrossEntropyLoss()
    epochs = cfg["train"]["epochs"]

    ckpt_dir = os.path.join(cfg["checkpoint"]["dir"], run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    last_path, best_path = f"{ckpt_dir}/last.pt", f"{ckpt_dir}/best.pt"

    start, history, run_id = 1, [], None
    if resume:
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        if sched and ck["scheduler"]:
            sched.load_state_dict(ck["scheduler"])
        start, history, run_id = ck["epoch"] + 1, ck["history"], ck["run_id"]
        print(f"[resume] {last_path} -> epoch {start}'den devam")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    mlflow.set_experiment(cfg["experiment_name"])
    peak, best_val = 0.0, max([h["val_acc"] for h in history], default=0.0)

    with mlflow.start_run(run_id=run_id, run_name=None if run_id else run_name) as run:
        if not resume:
            mlflow.log_params({
                "seed": cfg["seed"], "model": cfg["model"]["name"],
                "batch_size": cfg["data"]["batch_size"], "accum_steps": accum,
                "effective_batch": cfg["data"]["batch_size"] * accum,
                "num_workers": cfg["data"]["num_workers"], "epochs": epochs,
                "optimizer": cfg["train"]["optimizer"], "lr": cfg["train"]["lr"],
                "scheduler": cfg["train"]["scheduler"], "device": device,
                "n_params": sum(p.numel() for p in model.parameters())})
            cfg_copy = {k: v for k, v in cfg.items() if k != "_path"}
            cfg_file = f"{ckpt_dir}/config_used.yaml"
            with open(cfg_file, "w") as f:
                yaml.safe_dump(cfg_copy, f)
            mlflow.log_artifact(cfg_file)

        for epoch in range(start, epochs + 1):
            model.train()
            t0 = time.time()
            running = 0.0
            opt.zero_grad(set_to_none=True)
            for i, (x, y) in enumerate(tr):
                x, y = x.to(device), y.to(device)
                loss = loss_fn(model(x), y)
                (loss / accum).backward()
                if (i + 1) % accum == 0 or (i + 1) == len(tr):
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                running += loss.item() * y.size(0)
            if device == "cuda":
                torch.cuda.synchronize()
            train_time = time.time() - t0
            train_loss = running / len(tr.dataset)
            val_acc = evaluate(model, va)
            peak = max(peak, mem_mb())

            if sched is not None:
                sched.step(val_acc) if cfg["train"]["scheduler"] == "plateau" else sched.step()

            h = {"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc,
                 "epoch_time": train_time, "lr": opt.param_groups[0]["lr"]}
            history.append(h)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)
            mlflow.log_metric("epoch_time_s", train_time, step=epoch)
            mlflow.log_metric("throughput_samples_s", len(tr.dataset) / train_time, step=epoch)
            mlflow.log_metric("lr", h["lr"], step=epoch)
            print(f"[{run_name}] epoch {epoch}/{epochs} loss {train_loss:.4f} "
                  f"val_acc {val_acc:.4f} ({train_time:.1f}s)")

            def ckpt(path):
                torch.save({"epoch": epoch, "model": model.state_dict(),
                            "optimizer": opt.state_dict(),
                            "scheduler": sched.state_dict() if sched else None,
                            "history": history, "run_id": run.info.run_id}, path)

            if epoch % cfg["checkpoint"]["save_every"] == 0:
                ckpt(last_path)
            if val_acc > best_val:
                best_val = val_acc
                ckpt(best_path)
            if stop_after is not None and epoch == stop_after:
                print(f"[{run_name}] epoch {epoch}'de DURDURULDU (kasitli kesinti)")
                return {"history": history}

        test_acc = evaluate(model, te)
        thr = float(np.mean([len(tr.dataset) / h["epoch_time"] for h in history]))
        mlflow.log_metric("test_acc", test_acc)
        mlflow.log_metric("peak_memory_mb", peak)
        if os.path.exists(best_path):
            mlflow.log_artifact(best_path)
        print(f"[{run_name}] TEST ACC {test_acc:.4f}")
        return {"history": history, "final_val_acc": history[-1]["val_acc"],
                "test_acc": test_acc, "throughput": thr, "peak_mem_mb": peak}


def cmd_train(cfg):
    run_training(cfg, cfg["run_name"])


def cmd_profile(cfg):
    from torch.profiler import ProfilerActivity, profile, record_function
    set_seed(cfg["seed"])
    tr, _, _ = get_loaders(cfg)
    model = MODELS[cfg["model"]["name"]]().to(device)
    opt = build_opt(cfg, model)
    loss_fn = nn.CrossEntropyLoss()
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device == "cuda" else [])
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)

    model.train()
    it = iter(tr)
    with profile(activities=acts) as prof:
        for _ in range(len(tr)):
            with record_function("data_loading"):
                x, y = next(it)
                x, y = x.to(device), y.to(device)
                sync()
            with record_function("forward"):
                loss = loss_fn(model(x), y)
                sync()
            with record_function("backward"):
                opt.zero_grad(set_to_none=True)
                loss.backward()
                sync()
            with record_function("optimizer_step"):
                opt.step()
                sync()

    ev = {e.key: e for e in prof.key_averages()}
    names = ["data_loading", "forward", "backward", "optimizer_step"]
    secs = {n: ev[n].cpu_time_total / 1e6 for n in names}
    total = sum(secs.values())
    rows = [{"stage": n, "seconds": secs[n], "share_%": 100 * secs[n] / total} for n in names]
    save_table(rows, os.path.join(cfg["results_dir"], "profile_table.md"))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=12))

    mlflow.set_experiment(cfg["experiment_name"])
    with mlflow.start_run(run_name="profile_1_epoch"):
        mlflow.log_params({"num_workers": cfg["data"]["num_workers"],
                           "batch_size": cfg["data"]["batch_size"], "device": device})
        for r in rows:
            mlflow.log_metric(f"share_{r['stage']}", r["share_%"])
        mlflow.log_artifact(os.path.join(cfg["results_dir"], "profile_table.md"))


def cmd_workers(cfg):
    rows = []
    for nw in [0, 1, 2, 4, 8]:
        c = override(cfg, {"data.num_workers": nw, "train.epochs": 1})
        tr, _, _ = get_loaders(c)
        t0 = time.time()
        for _ in tr:                      
            pass
        t_data = time.time() - t0
        r = run_training(c, f"workers_{nw}")
        t_epoch = r["history"][0]["epoch_time"]
        rows.append({"num_workers": nw, "data_only_s": t_data, "epoch_s": t_epoch,
                     "data_share": t_data / t_epoch,
                     "samples_per_s": len(tr.dataset) / t_epoch})
    save_table(rows, os.path.join(cfg["results_dir"], "workers_table.md"))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    x = [r["num_workers"] for r in rows]
    ax[0].plot(x, [r["epoch_s"] for r in rows], "o-", label="epoch time")
    ax[0].plot(x, [r["data_only_s"] for r in rows], "s--", label="data loading only")
    ax[0].set_xlabel("num_workers"); ax[0].set_ylabel("seconds"); ax[0].legend()
    ax[1].plot(x, [r["samples_per_s"] for r in rows], "o-")
    ax[1].set_xlabel("num_workers"); ax[1].set_ylabel("samples/s")
    fig.suptitle("num_workers study"); fig.tight_layout()
    fig.savefig(os.path.join(cfg["results_dir"], "workers_plot.png"), dpi=150)


def cmd_batch(cfg):
    rows = []
    for bs in [32, 64, 128, 256, 512]:
        c = override(cfg, {"data.batch_size": bs, "train.epochs": 5})
        r = run_training(c, f"batch_{bs}")
        rows.append({"batch_size": bs, "samples_per_s": r["throughput"],
                     "peak_mem_mb": r["peak_mem_mb"], "val_acc_5ep": r["final_val_acc"],
                     "test_acc": r["test_acc"]})
    save_table(rows, os.path.join(cfg["results_dir"], "batch_table.md"))


def cmd_sched(cfg):
    lrs = {"sgd": 0.05, "adam": 0.001}
    rows, curves = [], {}
    for opt_name in ["sgd", "adam"]:
        for sch in ["step", "cosine", "plateau"]:
            c = override(cfg, {"train.epochs": 10, "train.optimizer": opt_name,
                               "train.scheduler": sch, "train.lr": lrs[opt_name],
                               "train.momentum": 0.9})
            r = run_training(c, f"sched_{opt_name}_{sch}")
            rows.append({"optimizer": opt_name, "scheduler": sch,
                         "val_acc_10ep": r["final_val_acc"], "test_acc": r["test_acc"],
                         "final_train_loss": r["history"][-1]["train_loss"]})
            curves[f"{opt_name}+{sch}"] = [h["val_acc"] for h in r["history"]]
    save_table(rows, os.path.join(cfg["results_dir"], "sched_table.md"))
    plt.figure(figsize=(8, 5))
    for k, v in curves.items():
        plt.plot(range(1, len(v) + 1), v, marker="o", label=k)
    plt.xlabel("epoch"); plt.ylabel("val acc"); plt.legend(); plt.grid(alpha=.3)
    plt.title("Optimizer x scheduler (10 epochs)")
    plt.savefig(os.path.join(cfg["results_dir"], "sched_plot.png"), dpi=150)


def cmd_accum(cfg):
    a = run_training(override(cfg, {"data.batch_size": 256, "train.epochs": 10}),
                     "accum_bs256")
    b = run_training(override(cfg, {"data.batch_size": 64, "train.epochs": 10}),
                     "accum_bs64x4", accum=4)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for r, lbl in [(a, "batch 256"), (b, "batch 64 x accum 4")]:
        ax[0].plot([h["train_loss"] for h in r["history"]], marker="o", label=lbl)
        ax[1].plot([h["val_acc"] for h in r["history"]], marker="o", label=lbl)
    ax[0].set_title("train loss"); ax[1].set_title("val acc")
    for x in ax:
        x.set_xlabel("epoch"); x.legend(); x.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg["results_dir"], "accum_plot.png"), dpi=150)
    print(f"test acc  bs256: {a['test_acc']:.4f}   64x4: {b['test_acc']:.4f}")


def cmd_resume(cfg):
    c = override(cfg, {"train.epochs": 10})
    run_training(c, "resume_demo", stop_after=3)
    r = run_training(c, "resume_demo", resume=True)
    h = r["history"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot([x["epoch"] for x in h], [x["train_loss"] for x in h], marker="o")
    ax[1].plot([x["epoch"] for x in h], [x["val_acc"] for x in h], marker="o")
    ax[0].set_title("train loss"); ax[1].set_title("val acc")
    for x in ax:
        x.axvline(3, color="red", ls="--", label="interrupt/resume (epoch 3)")
        x.set_xlabel("epoch"); x.legend(); x.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg["results_dir"], "resume_plot.png"), dpi=150)


COMMANDS = {"train": cmd_train, "profile": cmd_profile, "workers": cmd_workers,
            "batch": cmd_batch, "sched": cmd_sched, "accum": cmd_accum,
            "resume": cmd_resume}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=list(COMMANDS) + ["all"])
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    cfg = load_cfg(args.config)
    os.makedirs(cfg["results_dir"], exist_ok=True)
    print("device:", device)
    for name in (COMMANDS if args.cmd == "all" else [args.cmd]):
        print(f"\n===== {name} =====")
        COMMANDS[name](cfg)
