import logging
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from clean import CAT_COLS, NUM_COLS, split_and_clean
from ingest import load_raw

log = logging.getLogger("features")

NEW_COLS = ["capital_net", "hours_per_age"]


def add_features(X):
    X = X.copy()
    X["capital_net"] = X["capital-gain"] - X["capital-loss"]
    X["hours_per_age"] = X["hours-per-week"] / (X["age"] - 16)
    return X


def evaluate(X_train, X_test, y_train, y_test, num_cols):
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])
    model = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return (accuracy_score(y_test, pred),
            f1_score(y_test, pred, pos_label=">50K"))


def compare(X_train, X_test, y_train, y_test):
    acc0, f10 = evaluate(X_train, X_test, y_train, y_test, NUM_COLS)

    Xtr, Xte = add_features(X_train), add_features(X_test)
    acc1, f11 = evaluate(Xtr, Xte, y_train, y_test, NUM_COLS + NEW_COLS)

    text = "\n".join([
        "LogisticRegression baseline (test set, 30%)",
        f"Without derived features: accuracy={acc0:.4f}  F1={f10:.4f}",
        f"With derived features:    accuracy={acc1:.4f}  F1={f11:.4f}",
        f"Difference:               accuracy={acc1 - acc0:+.4f}  F1={f11 - f10:+.4f}",
    ])
    Path("results").mkdir(exist_ok=True)
    Path("results/feature_comparison.txt").write_text(text)
    return text, Xtr, Xte


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    X_train, X_test, y_train, y_test = split_and_clean(load_raw())
    text, _, _ = compare(X_train, X_test, y_train, y_test)
    print(text)
