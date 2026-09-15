"""Exploratory data analysis: class balance, symptom frequency, co-occurrence.

    python -m ml.eda

Writes PNG figures and a JSON summary to ``reports/``. Kept as a script rather
than a notebook so it runs in CI and in a terminal-only environment; the JSON
summary is what ``docs/evaluation_report.md`` quotes.
"""
from __future__ import annotations

import json
from collections import Counter
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml import config
from ml.cleaning import clean_dataframe, discover_schema, symptom_sets


def run_eda() -> dict:
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(config.RAW_CSV)
    schema = discover_schema(raw)
    df = clean_dataframe(raw, schema)
    sets = symptom_sets(df, schema)
    target = schema.target

    summary: dict = {
        "n_raw_rows": int(len(raw)),
        "n_clean_rows": int(len(df)),
        "target_column": target,
        "columns": list(raw.columns),
        "missingness_pct": {
            c: round(float(raw[c].isna().mean() * 100), 2) for c in raw.columns
        },
    }

    # --- class balance ----------------------------------------------------
    counts = df[target].value_counts()
    summary["class_balance"] = {str(k): int(v) for k, v in counts.items()}
    summary["n_classes"] = int(len(counts))
    summary["imbalance_ratio"] = round(float(counts.max() / max(counts.min(), 1)), 2)
    summary["majority_class_share"] = round(float(counts.max() / counts.sum()), 4)

    _barh(
        counts.head(20).sort_values(),
        f"Class balance of '{target}'",
        "cases",
        config.REPORT_DIR / "class_balance.png",
    )

    # --- symptom frequency ------------------------------------------------
    term_counts = Counter(t for s in sets for t in s)
    # From the RAW frame: how many distinct strings before canonicalisation.
    raw_strings = pd.concat([raw[c] for c in schema.symptom_columns]).dropna()
    summary["n_raw_symptom_strings"] = int(raw_strings.nunique())
    summary["n_distinct_symptoms"] = len(term_counts)
    summary["top_symptoms"] = dict(Counter(term_counts).most_common(25))
    summary["mean_symptoms_per_case"] = round(float(sets.map(len).mean()), 2)

    top = pd.Series(dict(Counter(term_counts).most_common(20))).sort_values()
    _barh(top, "Most frequent symptoms", "mentions", config.REPORT_DIR / "symptom_frequency.png")

    # --- symptom frequency by class --------------------------------------
    by_class: dict[str, dict[str, float]] = {}
    for cls in counts.index:
        mask = (df[target] == cls).to_numpy()
        cls_sets = [s for s, keep in zip(sets, mask) if keep]
        if not cls_sets:
            continue
        cls_counts = Counter(t for s in cls_sets for t in s)
        by_class[str(cls)] = {
            t: round(n / len(cls_sets), 3) for t, n in cls_counts.most_common(15)
        }
    summary["symptom_rate_by_class"] = by_class

    # --- lift: which symptoms separate the classes ------------------------
    if len(counts) == 2:
        pos, neg = counts.index[0], counts.index[1]
        rows = []
        for term, n in term_counts.items():
            if n < config.MIN_SYMPTOM_FREQ:
                continue
            present = np.array([term in s for s in sets])
            p_pos = float(((df[target] == pos).to_numpy() & present).sum()) / max(
                present.sum(), 1
            )
            rows.append((term, int(n), round(p_pos, 3)))
        rows.sort(key=lambda r: r[2])
        summary["symptoms_least_associated_with_majority"] = [
            {"symptom": t, "n": n, "p_majority": p} for t, n, p in rows[:15]
        ]

    # --- co-occurrence ----------------------------------------------------
    pair_counts: Counter = Counter()
    for s in sets:
        pair_counts.update(combinations(sorted(s), 2))
    summary["top_symptom_pairs"] = [
        {"pair": list(p), "n": int(n)} for p, n in pair_counts.most_common(20)
    ]

    common = [t for t, _ in term_counts.most_common(15)]
    matrix = pd.DataFrame(0, index=common, columns=common, dtype=int)
    for s in sets:
        present = [t for t in common if t in s]
        for a in present:
            for b in present:
                matrix.loc[a, b] += 1
    _heatmap(matrix, "Symptom co-occurrence (top 15)", config.REPORT_DIR / "cooccurrence.png")

    # --- species vs target ------------------------------------------------
    if "species" in df:
        cross = pd.crosstab(df["species"], df[target])
        summary["species_by_class"] = cross.to_dict(orient="index")
        summary["n_species"] = int(df["species"].nunique())

    out = config.REPORT_DIR / "eda_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"EDA written to {config.REPORT_DIR}")
    print(f"  rows {summary['n_raw_rows']} -> {summary['n_clean_rows']} after cleaning")
    print(
        f"  target '{target}': {summary['n_classes']} classes, "
        f"imbalance {summary['imbalance_ratio']}x, "
        f"majority share {summary['majority_class_share']:.1%}"
    )
    print(f"  {summary['n_distinct_symptoms']} distinct canonical symptoms")
    return summary


def _barh(series: pd.Series, title: str, xlabel: str, path) -> None:
    fig, ax = plt.subplots(figsize=(7, max(3, 0.32 * len(series))))
    ax.barh(series.index.astype(str), series.to_numpy(), color="#2f6f6b")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _heatmap(matrix: pd.DataFrame, title: str, path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix.to_numpy(), cmap="YlGnBu")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    run_eda()
