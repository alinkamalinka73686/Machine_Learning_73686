# Data Card: UCI Adult (Census Income)

**Source:** UCI Machine Learning Repository, loaded through OpenML with
`fetch_openml(name='adult', version=2, as_frame=True)`. The data was extracted
from the 1994 US Census database.

**License:** CC BY 4.0 (as listed by UCI).

**Size:** 48,842 rows and 15 columns (14 features + `income`). The data is
split 70/30 with a stratified split (seed 42): 34,189 train rows and 14,653
test rows. 24 exact duplicates were removed from train only, so the final
train set has 34,165 rows. The test set is untouched.

**Task:** predict whether a person earns more than $50K a year.

## Validation summary
- Schema and range checks: 0 violations, no rows were dropped.
- Missing values: workclass (2,799), occupation (2,809) and native-country
  (857). In this OpenML version they appear as NaN instead of '?'.
- Label audit: 5 groups of rows (10 rows in total) have identical features
  but different income labels. In a manual check of 50 random rows, no
  impossible values were found.

## Known biases
- The data is from 1994, so it does not describe today's population or wages.
- The label is imbalanced (about 76% `<=50K`).
- Most people in the data are men and White. A model can learn gender and race
  patterns, and the income gaps in the data are real historical gaps that a
  model can reproduce.
- Missing values are concentrated in a few columns and are probably not
  missing at random.
- `fnlwgt` is a census sampling weight, not a personal characteristic.
- Label noise: some identical rows have conflicting labels (see above).

## Privacy concerns
There are no names or addresses, but age, sex, race, country and occupation
together can still work as quasi-identifiers. The data is public and old, but
it should not be linked with other datasets to identify people. Sensitive
attributes (race, sex) should not be used to make real decisions about people.

## Pipeline and reproducibility
Run `python run_pipeline.py`. The settings are in `config.yaml`. The processed
dataset is saved as `results/processed.csv`, and its hash and the config used
are saved in `results/run_manifest.yaml`.

SHA256 of processed.csv: c107b49e630cb60cae706c1e52d99da9755ae9b005f854fbd8c9ff5e16d76ead

## Manual check and environment
The 50-row manual check is written up in `manual_check.md` (sample in
`results/audit_sample_50.csv`). It found no impossible values.

Tested with Python 3.11.9, pandas 3.1.0, numpy 2.4.6, scikit-learn 1.9.1,
joblib 1.6.0 and PyYAML 6.0.3 (pinned in `requirements.txt`). The hash above
was reproduced after deleting `results/` and running `python run_pipeline.py`
again. A different library version may change the hash.