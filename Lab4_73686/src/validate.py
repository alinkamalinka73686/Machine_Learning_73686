import logging
from pathlib import Path

import pandas as pd

from ingest import load_raw

log = logging.getLogger("validate")

INT_COLS = ["age", "fnlwgt", "education-num", "capital-gain",
            "capital-loss", "hours-per-week"]
CAT_COLS = ["workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country", "income"]
RANGES = {
    "age": (17, 90),
    "hours-per-week": (1, 99),
    "capital-gain": (0, 99999),
    "capital-loss": (0, 4356),
}


def validate(df):
    lines = []
    total = 0

    lines.append("== Schema check ==")
    for col in INT_COLS:
        if col not in df.columns:
            lines.append(f"VIOLATION: column '{col}' is missing")
            total += 1
        elif pd.api.types.is_integer_dtype(df[col]):
            lines.append(f"ok: {col} is int")
        else:
            lines.append(f"VIOLATION: {col} has dtype {df[col].dtype}, expected int")
            total += 1
    for col in CAT_COLS:
        if col not in df.columns:
            lines.append(f"VIOLATION: column '{col}' is missing")
            total += 1
        elif isinstance(df[col].dtype, pd.CategoricalDtype) or df[col].dtype == object:
            lines.append(f"ok: {col} is category")
        else:
            lines.append(f"VIOLATION: {col} has dtype {df[col].dtype}, expected category")
            total += 1

    lines.append("")
    lines.append("== Range check ==")
    for col, (lo, hi) in RANGES.items():
        bad = df[(df[col] < lo) | (df[col] > hi)]
        lines.append(f"{col} must be in [{lo}, {hi}]: {len(bad)} violations")
        total += len(bad)

    lines.append("")
    lines.append("== Missing values ('?' or NaN counts as missing) ==")
    for col in CAT_COLS + INT_COLS:
        n_q = int(df[col].astype(str).eq("?").sum())
        n_nan = int(df[col].isna().sum())
        if n_q or n_nan:
            lines.append(f"{col}: {n_q} '?' and {n_nan} NaN")

    lines.append("")
    lines.append("== Label check ==")
    lines.append(f"income values: {sorted(df['income'].astype(str).unique())}")

    lines.append("")
    lines.append(f"TOTAL violations: {total} (no rows were dropped)")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    report = validate(load_raw())
    print(report)
    Path("results").mkdir(exist_ok=True)
    Path("results/validation_report.txt").write_text(report)
