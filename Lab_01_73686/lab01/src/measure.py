import random
import statistics
import time
import os

import joblib
import numpy as np
import pandas as pd
from memory_profiler import memory_usage
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def peak(func):
    m = memory_usage((func, (), {}), interval=0.01, max_usage=True)
    return float(m[0] if isinstance(m, (list, tuple)) else m)


def main():
    random.seed(42)
    np.random.seed(42)

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    rows = []
    for name, base in models.items():
        # Training time: 1 warm-up, then median of 5
        clone(base).fit(X_train, y_train)
        times = []
        for _ in range(5):
            m = clone(base)
            t0 = time.perf_counter()
            m.fit(X_train, y_train)
            times.append(time.perf_counter() - t0)
        train_median = statistics.median(times)

        # Final model for inference and size
        model = clone(base).fit(X_train, y_train)
        sample = X_test[:1]

        # Single-sample inference: warm-up, then 100 repeats, median
        model.predict(sample)
        lat = []
        for _ in range(100):
            t0 = time.perf_counter()
            model.predict(sample)
            lat.append((time.perf_counter() - t0) * 1000)
        latency_median_ms = statistics.median(lat)

        # Model size
        path = f"lab01/results/model_{name}.joblib"
        joblib.dump(model, path)
        size_bytes = os.path.getsize(path)

        # Peak memory (MiB) during training and inference
        train_peak = peak(lambda: clone(base).fit(X_train, y_train))
        infer_peak = peak(lambda: [model.predict(sample) for _ in range(100)])

        rows.append({
            "model": name,
            "train_time_median_s": round(train_median, 4),
            "inference_median_ms": round(latency_median_ms, 4),
            "model_size_bytes": size_bytes,
            "model_size_KB": round(size_bytes / 1024, 2),
            "peak_mem_train_MiB": round(train_peak, 1),
            "peak_mem_inference_MiB": round(infer_peak, 1),
        })
        print(rows[-1])

    pd.DataFrame(rows).to_csv("lab01/results/system_cost.csv", index=False)


if __name__ == "__main__":
    main()