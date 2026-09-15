"""Step 0 of the pipeline: look at the real CSV before assuming anything.

    python -m ml.inspect_data [--csv data/data.csv]

Prints shape, dtypes, columns, null counts, target value counts and the roles
the pipeline inferred for each column. Run this first whenever you swap in a
new dataset -- everything downstream is driven by what this reports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ml import config
from ml.cleaning import canonical_symptom, clean_dataframe, discover_schema, symptom_sets


def inspect(csv_path: Path) -> dict:
    if not csv_path.exists():
        raise SystemExit(
            f"No CSV at {csv_path}.\n"
            "Drop your dataset there (see README > Data) and re-run."
        )

    df = pd.read_csv(csv_path)
    print("=" * 72)
    print(f"FILE      {csv_path}")
    print(f"SHAPE     {df.shape[0]} rows x {df.shape[1]} columns")
    print("=" * 72)

    print("\n--- COLUMNS / DTYPES / NULLS ---")
    summary = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "nulls": df.isna().sum(),
            "null_pct": (df.isna().mean() * 100).round(2),
            "unique": df.nunique(dropna=True),
        }
    )
    print(summary.to_string())

    schema = discover_schema(df)
    print("\n--- INFERRED ROLES ---")
    print(json.dumps(schema.to_dict(), indent=2))

    print(f"\n--- TARGET '{schema.target}' VALUE COUNTS ---")
    counts = df[schema.target].value_counts(dropna=False)
    print(counts.to_string())
    if len(counts) > 1:
        ratio = counts.max() / max(counts.min(), 1)
        print(f"\nclasses={len(counts.dropna())}  imbalance ratio (max/min)={ratio:.1f}x")

    print("\n--- SAMPLE ROWS ---")
    print(df.head(5).to_string())

    if schema.symptom_columns:
        raw = pd.concat([df[c] for c in schema.symptom_columns]).dropna()
        canon = raw.map(canonical_symptom)
        canon = canon[canon != ""]
        print(
            f"\n--- SYMPTOM VOCABULARY ---\n"
            f"raw distinct strings : {raw.nunique()}\n"
            f"after canonicalising : {canon.nunique()}"
        )
        vc = canon.value_counts()
        for threshold in (1, 2, 3, 5, 10):
            kept = (vc >= threshold).sum()
            coverage = vc[vc >= threshold].sum() / vc.sum()
            print(
                f"  terms with freq >= {threshold:>2}: {kept:>4}  "
                f"(covers {coverage:.1%} of mentions)"
            )
        print("\ntop 25 canonical symptoms:")
        print(vc.head(25).to_string())

    clean = clean_dataframe(df, schema)
    print(
        f"\n--- AFTER CLEANING ---\n"
        f"rows: {len(df)} -> {len(clean)} "
        f"(dropped {len(df) - len(clean)}: null targets + "
        f"{clean.attrs.get('duplicates_dropped', 0)} exact duplicates)"
    )
    if "species" in clean:
        print(f"canonical species: {clean['species'].nunique()}")
        print(clean["species"].value_counts().head(15).to_string())

    sets = symptom_sets(clean, schema)
    if len(sets):
        print(f"\nmean symptoms per case: {sets.map(len).mean():.2f}")

    missing = [
        name
        for name, col in (
            ("breed", schema.breed),
            ("age", schema.age),
            ("weight", schema.weight),
            ("symptom duration", schema.duration),
        )
        if col is None
    ]
    if missing:
        print(
            "\n[!] Fields described in the project proposal but ABSENT from this CSV: "
            + ", ".join(missing)
            + "\n    The pipeline simply omits them; add the columns and re-run"
            "\n    training to have them picked up automatically."
        )

    return {"schema": schema.to_dict(), "n_rows": len(df), "n_clean_rows": len(clean)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=config.RAW_CSV)
    args = parser.parse_args()
    inspect(args.csv)


if __name__ == "__main__":
    main()
