"""Run the fixed Week 1 protocol; never tune against the held-out test set."""

import argparse
import hashlib
import json
import math
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from lead_intelligence.data import FEATURES, TARGET, generate_leads
from lead_intelligence.evaluation import evaluate
from lead_intelligence.model import build_pipeline
from lead_intelligence.validation import exploratory_report, validate_schema


def run_experiment(n_rows: int, seed: int, capacity_fraction: float,
                   output: Path, artifacts: Path) -> dict:
    if n_rows < 100 or not 0 < capacity_fraction <= 1:
        raise ValueError("Use at least 100 rows and a capacity fraction in (0, 1]")
    data = generate_leads(n_rows, seed)
    validate_schema(data)
    train, test = train_test_split(data, test_size=.2, random_state=seed, stratify=data[TARGET])
    assert set(train.lead_id).isdisjoint(test.lead_id)
    exploration = exploratory_report(train)
    pipeline = build_pipeline()
    pipeline.fit(train[FEATURES], train[TARGET])
    scores = pipeline.predict_proba(test[FEATURES])[:, 1]
    k = math.ceil(capacity_fraction * len(test))
    prevalence = float(test[TARGET].mean())
    report = {
        "protocol": {"n_rows": n_rows, "seed": seed, "test_fraction": .2,
                     "capacity_fraction": capacity_fraction, "train_rows": len(train),
                     "test_rows": len(test), "train_conversion_rate": float(train[TARGET].mean()),
                     "test_conversion_rate": prevalence, "split": "stratified random, unique leads",
                     "dataset_sha256": hashlib.sha256(data.to_csv(index=False).encode()).hexdigest()},
        "versions": {"python": platform.python_version(), **{p: version(p) for p in
                     ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "threadpoolctl"]}},
        "metrics": evaluate(test[TARGET].to_numpy(), scores, k),
        "random_ranking_expected": {"precision_at_k": prevalence,
                                    "recall_at_k": k / len(test), "lift_at_k": 1.0},
    }
    output.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    for name, payload in [("baseline_metrics.json", report), ("data_validation.json", exploration)]:
        (output / name).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    joblib.dump(pipeline, artifacts / "baseline.joblib")
    (artifacts / "metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--capacity-fraction", type=float, default=.2)
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    try:
        report = run_experiment(args.rows, args.seed, args.capacity_fraction, args.output, args.artifacts)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
