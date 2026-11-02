# Lab 4: Problem Definition

In this lab I work with the UCI Adult (Census Income) dataset, loaded with
`fetch_openml(name='adult', version=2, as_frame=True)`. It has 48,842 rows and
15 columns. The goal is to predict whether a person earns more than $50K a
year, so it is a binary classification problem.

## Inputs
The 14 feature columns (age, workclass, fnlwgt, education, education-num,
marital-status, occupation, relationship, race, sex, capital-gain,
capital-loss, hours-per-week, native-country). Some of them contain '?' for
missing values.

## Output
`income`, either `<=50K` or `>50K`. About three quarters of the rows are
`<=50K`, so I use a stratified split.

## Success metric
F1 score and accuracy on a 30% test set. I care more about F1 because accuracy
looks good even for a model that always predicts `<=50K`.

## Constraints
- No leakage. I split the data into train and test before any cleaning, and
  everything (imputer, clipping bounds, encoders) is fitted on train only.
- The pipeline must be reproducible: fixed random seed, settings in a config
  file, and the whole thing runs with one command.
