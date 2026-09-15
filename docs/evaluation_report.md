# VetDx model evaluation report

*Generated from `evaluation_report.json`, training run 2026-09-12T10:33:22+00:00. Regenerate with `python -m ml.report`.*

## Data

- Source: `data/data.csv`
- Columns: `AnimalName`, `symptoms1`, `symptoms2`, `symptoms3`, `symptoms4`, `symptoms5`, `Dangerous`
- Rows: 871 raw -> 841 after dropping null targets and 28 exact duplicates
- Target column: `Dangerous`
- Class counts: Yes = 821, No = 20 (41.05x imbalance)
- **Absent from this CSV** (described in the proposal): breed, age, weight, symptom_duration
- Distinct canonical symptoms: 826; mean 4.94 per case
- Split (stratified): train 537 / validation 135 / test 169
- Test class counts: No = 4, Yes = 165
- Engineered features: 196

## Model comparison

Scores on the validation split; the CV column is 5-fold stratified macro F1 over train+validation. Selection metric: **macro_f1 (validation split)**.

| Model | Accuracy | Macro F1 | Balanced acc. | ROC AUC | CV macro F1 | Fit (s) | Tuned |
|---|---|---|---|---|---|---|---|
| `majority_baseline` | 0.978 | 0.494 | 0.500 | 0.500 | 0.494 | 0.0 | — |
| `logistic_regression` **(deployed)** | 0.993 | 0.927 | 0.996 | 0.998 | 0.822 | 4.88 | C=0.003 |
| `naive_bayes` | 0.978 | 0.494 | 0.500 | 0.896 | 0.743 | 0.1 | alpha=0.5 |
| `random_forest` | 0.970 | 0.492 | 0.496 | 0.977 | 0.745 | 12.1 | max_depth=6, min_samples_leaf=2 |
| `neural_net` | 0.963 | 0.491 | 0.492 | 0.318 | 0.564 | 1.01 | alpha=0.01, hidden_layer_sizes=(32,) |
| `xgboost` | 0.985 | 0.746 | 0.667 | 0.990 | 0.614 | 4.81 | learning_rate=0.05, max_depth=3, n_estimators=400 |

## Chosen model: `logistic_regression`

Selected on validation macro F1, then refit on train+validation and scored once on the untouched test split.

| Metric | Test score |
|---|---|
| accuracy | 0.970 |
| macro f1 | 0.715 |
| weighted f1 | 0.972 |
| balanced accuracy | 0.741 |
| roc auc | 0.976 |

### Per-class performance (test split)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No | 0.400 | 0.500 | 0.444 | 4 |
| Yes | 0.988 | 0.982 | 0.985 | 165 |

### Confusion matrix (test split)

| actual \ predicted | No | Yes |
|---|---|---|
| **No** | 2 | 2 |
| **Yes** | 3 | 162 |

## What the model leans on

Global importance, method: `shap`.

| Feature | Mean absolute contribution |
|---|---|
| Species: Buffaloes | 0.1896 |
| Species: Chicken | 0.0821 |
| Symptom: Lethargy | 0.0806 |
| Symptom pair: Diarrhea + Loss Of Appetite | 0.0530 |
| Symptom: Fever | 0.0487 |
| Symptom: Epistaxis | 0.0449 |
| Symptom pair: Diarrhea + Lethargy | 0.0408 |
| Symptom: Nasal Discharge | 0.0403 |
| Symptom: Vomiting | 0.0346 |
| Species: Monkey | 0.0329 |
| Species: Sheep | 0.0304 |
| Symptom pair: Fever + Pain | 0.0301 |
| Symptom pair: Fever + Weight Loss | 0.0287 |
| Symptom: Ruffled Feathers | 0.0282 |
| Symptom pair: Pain + Weight Loss | 0.0281 |

## Limitations

These matter more than the headline numbers.

1. **The dataset carries no disease label.** The project proposal targets a
   multi-class disease diagnosis. The CSV at `data/data.csv` has only
   `AnimalName`, `symptoms1`-`symptoms5` and a binary `Dangerous` flag, so the
   deployed model answers *"does this case look serious?"*, not *"which disease
   is it?"*. The pipeline picks a `disease`/`diagnosis` column automatically if
   one is ever added (see `TARGET_CANDIDATES` in `ml/config.py`); nothing else
   has to change.
2. **Severe class imbalance.** 41.05x between the largest and smallest
   class. A model that always answers "Yes" scores
   97.6% accuracy, which is why macro F1 and balanced accuracy are
   the selection and reporting metrics here. Accuracy alone is meaningless on
   this data.
3. **The minority class is tiny.** The held-out test split contains only
   4 minority-class cases, so per-class precision/recall for it move
   in large steps and carry very wide confidence intervals. The 5-fold CV macro
   F1 on train+validation is the more stable estimate.
4. **Missing clinical inputs.** No breed, age, weight or symptom-duration
   columns exist in this CSV, so genuinely predictive signals the proposal
   assumed are simply unavailable. Cleaning, parsing and feature code for all
   four is implemented and unit-tested, dormant until the columns appear.
5. **Label and text noise.** Symptoms are free text with inconsistent spelling,
   casing and encoding artefacts. A synonym table merges the obvious variants
   (935 raw strings collapse to 826 canonical terms), but
   spelling errors and clinically ambiguous phrasings remain.
6. **Confounded species signal.** Species is one of the strongest features,
   which reflects how the data was collected rather than biology: minority-class
   cases are concentrated in a couple of species. Treat per-species behaviour as
   a sampling artefact.
7. **Unknown provenance.** The geographic coverage, breed coverage and clinical
   review process behind the source records are unknown, so nothing here should
   be assumed to generalise to a new population.
8. **Not a diagnostic tool.** VetDx is a triage and decision-support aid, not a diagnostic tool. Its predictions are statistical associations learned from a small, noisy, publicly sourced dataset and are not a substitute for examination and diagnosis by a qualified veterinarian. Always seek professional care for a sick animal.

## Reproducing this report

```bash
python -m ml.inspect_data   # confirm the schema of your CSV
python -m ml.eda            # figures + reports/eda_summary.json
python -m ml.train          # comparison, winner, artifacts, SHAP
python -m ml.report         # regenerate this document
```

Environment: Python 3.11.9 on Windows-10-10.0.26200-SP0.
