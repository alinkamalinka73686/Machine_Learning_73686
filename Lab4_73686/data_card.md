# Data Card: UCI Adult (Census Income)

**Source:** UCI Machine Learning Repository, loaded through OpenML with
`fetch_openml(name='adult', version=2, as_frame=True)`. It was extracted from
the 1994 US Census database.

**License:** CC BY 4.0 (as listed by UCI). Please check the OpenML page again
before submitting.

**Size:** 48,842 rows, 15 columns (14 features + `income`). In this lab the
data is split 70/30 (stratified, seed 42). Exact duplicates are removed from
train only.

**Task:** predict whether income is above $50K per year.

## Known biases
- The data is from 1994, so it does not describe today's population or wages.
- The label is imbalanced (about 76% `<=50K`).
- Men and White people are the majority, so the model can learn gender and
  race patterns. Income gaps between groups in the data are real historical
  gaps, and a model trained on them can reproduce them.
- Missing values ('?') are concentrated in workclass, occupation and
  native-country, and are probably not missing at random.
- `fnlwgt` is a census sampling weight, not a personal characteristic.
- Duplicate-conflict result: FILL IN from results/label_audit.txt.

## Privacy concerns
There are no names or addresses, but age, sex, race, country and occupation
together can still act as quasi-identifiers. The data is public and old, but
it should not be linked with other datasets to identify people. Sensitive
attributes (race, sex) should not be used to make real decisions about people.

## Pipeline and reproducibility
Run `python run_pipeline.py`. The SHA256 hash of `results/processed.csv` and
the config used are saved in `results/run_manifest.yaml`.
SHA256: FILL IN after running.
