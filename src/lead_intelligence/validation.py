"""Question-driven exploratory checks, restricted to the training partition."""

import pandas as pd

from lead_intelligence.data import CATEGORICAL, FEATURES, NUMERIC, TARGET


def validate_schema(df: pd.DataFrame) -> None:
    if set(df.columns) != {"lead_id", TARGET, *FEATURES}:
        raise ValueError("Unexpected schema: possible leakage or missing feature")
    if df.empty or not df.lead_id.is_unique or df.lead_id.isna().any():
        raise ValueError("Each row must represent one unique lead")
    if df[TARGET].isna().any() or not df[TARGET].isin([0, 1]).all():
        raise ValueError("Target must be observed and binary")
    for col in NUMERIC:
        if not pd.api.types.is_numeric_dtype(df[col]) or (df[col].dropna() < 0).any():
            raise ValueError(f"Invalid numeric feature: {col}")


def exploratory_report(df: pd.DataFrame) -> dict:
    validate_schema(df)
    groups = {col: df.groupby(col, dropna=False)[TARGET].agg(["count", "mean"])
              for col in CATEGORICAL}
    for col in NUMERIC:
        groups[col] = df.groupby(pd.qcut(df[col], 4, duplicates="drop"),
                                 observed=True)[TARGET].agg(["count", "mean"])
    relationships = {col: {str(key): {"count": int(row["count"]),
                                     "conversion_rate": float(row["mean"])}
                           for key, row in table.iterrows()} for col, table in groups.items()}
    return {
        "partition": "training only",
        "question_is_conversion_imbalanced": {"rows": len(df),
            "conversion_rate": float(df[TARGET].mean()),
            "class_counts": {str(k): int(v) for k, v in df[TARGET].value_counts().items()}},
        "question_which_inputs_need_imputation": df.isna().sum().to_dict(),
        "question_are_numeric_ranges_plausible": df[NUMERIC].describe().to_dict(),
        "question_are_categories_represented": {
            col: {str(k): int(v) for k, v in df[col].value_counts(dropna=False).items()}
            for col in CATEGORICAL},
        "question_which_features_separate_outcomes": relationships,
        "leakage_checks": {
            "unique_lead_ids": True, "schema_allowlist_passed": True,
            "target_and_id_excluded_from_features": TARGET not in FEATURES and "lead_id" not in FEATURES,
            "exact_numeric_target_copies": [c for c in NUMERIC if df[c].equals(df[TARGET])],
            "temporal_review": "All features defined at day 7; outcome follows snapshot. "
                               "No real timestamps exist, so temporal validity is a design assumption, not an empirical proof.",
        },
    }
