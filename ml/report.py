"""Render docs/evaluation_report.md from the JSON artifacts.

    python -m ml.report

Keeps the written-up results in lockstep with the last training run instead of
being hand-maintained prose that drifts.
"""
from __future__ import annotations

import json

from ml import config

LIMITATIONS = """\
## Limitations

These matter more than the headline numbers.

1. **The dataset carries no disease label.** The project proposal targets a
   multi-class disease diagnosis. The CSV at `data/data.csv` has only
   `AnimalName`, `symptoms1`-`symptoms5` and a binary `Dangerous` flag, so the
   deployed model answers *"does this case look serious?"*, not *"which disease
   is it?"*. The pipeline picks a `disease`/`diagnosis` column automatically if
   one is ever added (see `TARGET_CANDIDATES` in `ml/config.py`); nothing else
   has to change.
2. **Severe class imbalance.** {imbalance}x between the largest and smallest
   class. A model that always answers "{majority}" scores
   {baseline_acc:.1%} accuracy, which is why macro F1 and balanced accuracy are
   the selection and reporting metrics here. Accuracy alone is meaningless on
   this data.
3. **The minority class is tiny.** The held-out test split contains only
   {n_minority} minority-class cases, so per-class precision/recall for it move
   in large steps and carry very wide confidence intervals. The 5-fold CV macro
   F1 on train+validation is the more stable estimate.
4. **Missing clinical inputs.** No breed, age, weight or symptom-duration
   columns exist in this CSV, so genuinely predictive signals the proposal
   assumed are simply unavailable. Cleaning, parsing and feature code for all
   four is implemented and unit-tested, dormant until the columns appear.
5. **Label and text noise.** Symptoms are free text with inconsistent spelling,
   casing and encoding artefacts. A synonym table merges the obvious variants
   ({raw_terms} raw strings collapse to {clean_terms} canonical terms), but
   spelling errors and clinically ambiguous phrasings remain.
6. **Confounded species signal.** Species is one of the strongest features,
   which reflects how the data was collected rather than biology: minority-class
   cases are concentrated in a couple of species. Treat per-species behaviour as
   a sampling artefact.
7. **Unknown provenance.** The geographic coverage, breed coverage and clinical
   review process behind the source records are unknown, so nothing here should
   be assumed to generalise to a new population.
8. **Not a diagnostic tool.** {disclaimer}
"""


def render() -> str:
    if not config.EVAL_REPORT_PATH.exists():
        raise SystemExit(
            f"No {config.EVAL_REPORT_PATH}. Run `python -m ml.train` first."
        )
    report = json.loads(config.EVAL_REPORT_PATH.read_text(encoding="utf-8"))
    eda = {}
    eda_path = config.REPORT_DIR / "eda_summary.json"
    if eda_path.exists():
        eda = json.loads(eda_path.read_text(encoding="utf-8"))

    dataset = report["dataset"]
    split = report["split"]
    classes = report["confusion_matrix"]["labels"]
    counts = dataset["class_counts"]
    majority = max(counts, key=counts.get)
    baseline_acc = counts[majority] / sum(counts.values())

    lines: list[str] = []
    add = lines.append

    add("# VetDx model evaluation report")
    add("")
    add(f"*Generated from `{config.EVAL_REPORT_PATH.name}`, training run "
        f"{report['trained_at']}. Regenerate with `python -m ml.report`.*")
    add("")

    add("## Data")
    add("")
    add(f"- Source: `{dataset['path']}`")
    add(f"- Columns: {', '.join(f'`{c}`' for c in dataset['columns'])}")
    add(f"- Rows: {dataset['n_raw_rows']} raw -> {dataset['n_clean_rows']} after "
        f"dropping null targets and {dataset['duplicates_dropped']} exact duplicates")
    add(f"- Target column: `{dataset['target_column']}`")
    add(f"- Class counts: {', '.join(f'{k} = {v}' for k, v in counts.items())} "
        f"({dataset['imbalance_ratio']}x imbalance)")
    if dataset.get("fields_absent_from_dataset"):
        add(f"- **Absent from this CSV** (described in the proposal): "
            f"{', '.join(dataset['fields_absent_from_dataset'])}")
    if eda:
        add(f"- Distinct canonical symptoms: {eda.get('n_distinct_symptoms')}; "
            f"mean {eda.get('mean_symptoms_per_case')} per case")
    add(f"- Split (stratified): train {split['train']} / validation {split['val']} "
        f"/ test {split['test']}")
    add(f"- Test class counts: "
        f"{', '.join(f'{k} = {v}' for k, v in split['test_class_counts'].items())}")
    add(f"- Engineered features: {report['n_features']}")
    add("")

    add("## Model comparison")
    add("")
    add("Scores on the validation split; the CV column is 5-fold stratified "
        "macro F1 over train+validation. Selection metric: "
        f"**{report['selection_metric']}**.")
    add("")
    add("| Model | Accuracy | Macro F1 | Balanced acc. | ROC AUC | CV macro F1 | Fit (s) | Tuned |")
    add("|---|---|---|---|---|---|---|---|")
    for row in report["comparison"]:
        val = row["val"]
        tuned = row.get("tuned_params") or {}
        tuned_str = ", ".join(f"{k.split('__')[-1]}={v}" for k, v in tuned.items()) or "—"
        marker = " **(deployed)**" if row["model"] == report["best_model"] else ""
        add(
            f"| `{row['model']}`{marker} | {val['accuracy']:.3f} | {val['macro_f1']:.3f} "
            f"| {val['balanced_accuracy']:.3f} | {_fmt(val.get('roc_auc'))} "
            f"| {_fmt(row.get('cv_macro_f1_mean'))} | {row['fit_seconds']} | {tuned_str} |"
        )
    add("")

    add(f"## Chosen model: `{report['best_model']}`")
    add("")
    add("Selected on validation macro F1, then refit on train+validation and "
        "scored once on the untouched test split.")
    add("")
    test = report["test_metrics"]
    add("| Metric | Test score |")
    add("|---|---|")
    for key in ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "roc_auc"):
        add(f"| {key.replace('_', ' ')} | {_fmt(test.get(key))} |")
    add("")

    add("### Per-class performance (test split)")
    add("")
    add("| Class | Precision | Recall | F1 | Support |")
    add("|---|---|---|---|---|")
    for cls in classes:
        row = report["test_per_class"].get(cls, {})
        add(f"| {cls} | {_fmt(row.get('precision'))} | {_fmt(row.get('recall'))} "
            f"| {_fmt(row.get('f1-score'))} | {int(row.get('support', 0))} |")
    add("")

    add("### Confusion matrix (test split)")
    add("")
    add("| actual \\ predicted | " + " | ".join(classes) + " |")
    add("|---" * (len(classes) + 1) + "|")
    for label, row in zip(classes, report["confusion_matrix"]["matrix"]):
        add(f"| **{label}** | " + " | ".join(str(n) for n in row) + " |")
    add("")

    if config.SHAP_GLOBAL_PATH.exists():
        shap_data = json.loads(config.SHAP_GLOBAL_PATH.read_text(encoding="utf-8"))
        add("## What the model leans on")
        add("")
        add(f"Global importance, method: `{shap_data.get('method')}`.")
        add("")
        add("| Feature | Mean absolute contribution |")
        add("|---|---|")
        for feature in shap_data.get("features", [])[:15]:
            add(f"| {feature['label']} | {feature['importance']:.4f} |")
        add("")

    raw_terms = eda.get("n_raw_symptom_strings", "?")
    clean_terms = eda.get("n_distinct_symptoms", "?")

    add(
        LIMITATIONS.format(
            imbalance=dataset["imbalance_ratio"],
            majority=majority,
            baseline_acc=baseline_acc,
            n_minority=min(split["test_class_counts"].values()),
            raw_terms=raw_terms,
            clean_terms=clean_terms,
            disclaimer=config.DISCLAIMER,
        )
    )

    add("## Reproducing this report")
    add("")
    add("```bash")
    add("python -m ml.inspect_data   # confirm the schema of your CSV")
    add("python -m ml.eda            # figures + reports/eda_summary.json")
    add("python -m ml.train          # comparison, winner, artifacts, SHAP")
    add("python -m ml.report         # regenerate this document")
    add("```")
    add("")
    add(f"Environment: Python {report['environment']['python']} on "
        f"{report['environment']['platform']}.")
    add("")
    return "\n".join(lines)


def _fmt(value) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def main() -> None:
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.DOCS_DIR / "evaluation_report.md"
    out.write_text(render(), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
