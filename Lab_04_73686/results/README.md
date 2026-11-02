# Lab 4: Data pipeline and quality (UCI Adult)

## Run
pip install -r requirements.txt
python run_pipeline.py

One command runs ingestion, validation, split, cleaning, feature
engineering, audits and hashing. Outputs go to results/.

## Results
- Validation: 0 schema/range violations, no rows dropped (results/validation_report.txt).
- Split: 70/30 stratified, seed 42. Imputer and clipping bounds fitted on train only. 24 exact duplicates dropped from train only.
- LogisticRegression baseline (test set):
  - Without derived features: accuracy 0.8540, F1 0.6601
  - With capital_net and hours_per_age: accuracy 0.8578, F1 0.6696
- Label audit: 5 conflicting duplicate groups (10 rows); 0 impossible values in 50 random rows (results/label_audit.txt, manual_check.md).
- SHA256 of processed.csv: c107b49e630cb60cae706c1e52d99da9755ae9b005f854fbd8c9ff5e16d76ead

## Files
- problem_definition.md, data_card.md, manual_check.md
- config.yaml, requirements.txt, run_pipeline.py
- src/: ingest, validate, clean, features, audit
- results/: all generated outputs, including preprocessor.joblib
