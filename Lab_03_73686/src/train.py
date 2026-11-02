import json
import time
import torch
import torch.nn as nn
from data import get_loaders
from models import get_models

EPOCHS = 5
LR = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def train_model(name, model, train_loader, val_loader, test_loader):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * y.size(0)
        train_loss = running / len(train_loader.dataset)
        val_acc = evaluate(model, val_loader)
        print(f"[{name}] epoch {epoch}/{EPOCHS}  loss {train_loss:.4f}  "
              f"val_acc {val_acc:.4f}  ({time.time() - t0:.1f}s)")

    test_acc = evaluate(model, test_loader)
    print(f"[{name}] TEST ACC: {test_acc:.4f}\n")
    return test_acc


if __name__ == "__main__":
    print("device:", device)
    train_loader, val_loader, test_loader = get_loaders()

    torch.manual_seed(0)
    models = get_models()

    results = {}
    for name, model in models.items():
        results[name] = {
            "test_acc": train_model(name, model, train_loader, val_loader, test_loader),
            "params": sum(p.numel() for p in model.parameters()),
        }

    with open("results/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results.json")
    print(json.dumps(results, indent=2))
