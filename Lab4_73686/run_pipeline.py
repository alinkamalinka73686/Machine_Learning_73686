import hashlib
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)  

from audit import conflict_audit, sample_check  
from clean import split_and_clean  
from features import compare  
from ingest import load_raw  
from validate import validate 

log = logging.getLogger("pipeline")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = yaml.safe_load(open("config.yaml"))
    out = Path(cfg["output_dir"])
    out.mkdir(exist_ok=True)

    df = load_raw()

    (out / "validation_report.txt").write_text(validate(df))

    X_train, X_test, y_train, y_test = split_and_clean(
        df, seed=cfg["seed"], test_size=cfg["test_size"]
    )

    text, Xtr, Xte = compare(X_train, X_test, y_train, y_test)
    print(text)

    n_groups, n_rows = conflict_audit(df)
    sample, problems = sample_check(df, n=cfg["audit_sample_size"])
    sample.to_csv(out / "audit_sample_50.csv")
    lines = [
        f"Conflicting duplicate groups: {n_groups} ({n_rows} rows)",
        f"Impossible values in {cfg['audit_sample_size']} random rows: {len(problems)}",
    ] + [f"  row {i}: {m}" for i, m in problems]
    (out / "label_audit.txt").write_text("\n".join(lines))

    train = Xtr.assign(income=y_train.values, split="train")
    test = Xte.assign(income=y_test.values, split="test")
    processed = pd.concat([train, test], ignore_index=True)
    proc_path = out / "processed.csv"
    processed.to_csv(proc_path, index=False)
    digest = sha256_of(proc_path)
    log.info("SHA256 of processed.csv: %s", digest)

    manifest = dict(cfg)
    manifest["processed_file"] = str(proc_path)
    manifest["processed_rows"] = len(processed)
    manifest["processed_sha256"] = digest
    (out / "run_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    log.info("Done. Results are in %s/", out)


if __name__ == "__main__":
    main()
