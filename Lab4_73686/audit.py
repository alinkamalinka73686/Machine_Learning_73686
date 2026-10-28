import logging
from pathlib import Path

import pandas as pd

from ingest import load_raw

log = logging.getLogger("audit")

SEED = 42


def conflict_audit(df):
    feats = [c for c in df.columns if c != "income"]
    key = df[feats].astype(str)
    grouped = df.assign(_key=key.agg("|".join, axis=1)).groupby("_key")["income"]
    n_labels = grouped.nunique()
    conflicting = n_labels[n_labels > 1]
    rows_involved = int(grouped.size()[conflicting.index].sum())
    return len(conflicting), rows_involved


def sample_check(df, n=50):
    sample = df.sample(n=n, random_state=SEED)
    problems = []
    for idx, r in sample.iterrows():
        if not 17 <= r["age"] <= 90:
            problems.append((idx, "age out of range"))
        if not 1 <= r["hours-per-week"] <= 99:
            problems.append((idx, "hours-per-week out of range"))
        if r["capital-gain"] < 0 or r["capital-gain"] > 99999:
            problems.append((idx, "capital-gain out of range"))
        if r["capital-loss"] < 0 or r["capital-loss"] > 4356:
            problems.append((idx, "capital-loss out of range"))
        if r["capital-gain"] > 0 and r["capital-loss"] > 0:
            problems.append((idx, "both capital-gain and capital-loss > 0"))
        if r["education-num"] < 1 or r["education-num"] > 16:
            problems.append((idx, "education-num out of range"))
    return sample, problems


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    df = load_raw()

    n_groups, n_rows = conflict_audit(df)
    sample, problems = sample_check(df)

    lines = [
        "== Duplicate-conflict audit ==",
        f"Feature combinations with conflicting income labels: {n_groups}",
        f"Rows involved in those conflicts: {n_rows}",
        "",
        "== 50 random rows (seed=42), impossible-value check ==",
        f"Impossible values found: {len(problems)}",
    ]
    for idx, msg in problems:
        lines.append(f"  row {idx}: {msg}")
    report = "\n".join(lines)
    print(report)

    Path("results").mkdir(exist_ok=True)
    Path("results/label_audit.txt").write_text(report)
    sample.to_csv("results/audit_sample_50.csv")
