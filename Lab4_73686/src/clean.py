import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ingest import load_raw

log = logging.getLogger("clean")

NUM_COLS = ["age", "fnlwgt", "education-num", "capital-gain",
            "capital-loss", "hours-per-week"]
CAT_COLS = ["workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country"]


class Preprocessor:
    """Imputes (median / mode) and clips to the 1st/99th percentile.
    fit() sees train only; transform() is applied to train and test."""

    def fit(self, X):
        X = self._mark_missing(X)
        self.medians_ = {c: float(X[c].median()) for c in NUM_COLS}
        self.modes_ = {c: X[c].mode().iloc[0] for c in CAT_COLS}
        self.bounds_ = {c: (float(X[c].quantile(0.01)),
                            float(X[c].quantile(0.99))) for c in NUM_COLS}
        return self

    def transform(self, X):
        X = self._mark_missing(X)
        for c in NUM_COLS:
            X[c] = X[c].fillna(self.medians_[c])
            lo, hi = self.bounds_[c]
            X[c] = X[c].clip(lo, hi)
        for c in CAT_COLS:
            X[c] = X[c].fillna(self.modes_[c])
        return X

    @staticmethod
    def _mark_missing(X):
        X = X.copy()
        for c in CAT_COLS:
            X[c] = X[c].astype(object).replace("?", np.nan)
        for c in NUM_COLS:
            X[c] = X[c].astype(float)
        return X


def split_and_clean(df, seed=42, test_size=0.3):
    train, test = train_test_split(
        df, test_size=test_size, stratify=df["income"], random_state=seed
    )
    log.info("Split: train=%d, test=%d", len(train), len(test))

    before = len(train)
    train = train.drop_duplicates()
    log.info("Dropped %d exact duplicate rows from train", before - len(train))

    X_train, y_train = train.drop(columns="income"), train["income"]
    X_test, y_test = test.drop(columns="income"), test["income"]
    pre = Preprocessor().fit(X_train)

    X_train = pre.transform(X_train)
    X_test = pre.transform(X_test)

    Path("results").mkdir(exist_ok=True)
    joblib.dump(pre, "results/preprocessor.joblib")
    log.info("Saved results/preprocessor.joblib")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    split_and_clean(load_raw())
