"""Model selection hanya di training; evaluasi holdout dilakukan setelah pilihan beku."""

import argparse
import hashlib
import json
import math
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits

from lead_intelligence.data import FEATURES, TARGET, generate_leads
from lead_intelligence.evaluation import evaluate
from lead_intelligence.model import build_challenger, build_pipeline
from lead_intelligence.validation import validate_schema

# Fixed before examining v0.2 validation or test performance; absolute differences.
POLICY = {
    "challenger_min_ap_gain": .02, "challenger_min_lift_gain": .15,
    "challenger_max_brier_increase": .005,
    "calibration_min_brier_reduction": .002,
    "calibration_min_log_loss_reduction": .005,
    "calibration_max_ap_loss": .01, "calibration_max_lift_loss": .10,
}
FAMILIES = ("logistic_regression", "hist_gradient_boosting")


def split_data(data, seed: int):
    """Reproduce v0.1 outer split exactly; split only training rows for selection."""
    train, test = train_test_split(data, test_size=.2, random_state=seed, stratify=data[TARGET])
    fit, validation = train_test_split(train, test_size=.25, random_state=seed,
                                     stratify=train[TARGET])
    return train, fit, validation, test


def candidate(family: str, calibrated: bool, seed: int):
    if family not in FAMILIES:
        raise ValueError(f"Unknown family: {family}")
    model = build_pipeline() if family == FAMILIES[0] else build_challenger(seed)
    if calibrated:
        # All preprocessing is inside the estimator and refitted within inner folds.
        model = CalibratedClassifierCV(
            model, method="sigmoid", ensemble=False,
            cv=StratifiedKFold(3, shuffle=True, random_state=seed), n_jobs=1)
    return model


def reliability_table(y, probabilities, n_bins: int = 10) -> list[dict]:
    """Fixed-width bins include counts and empty bins; ECE depends on this binning."""
    y, p = np.asarray(y), np.asarray(probabilities)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = []
    for index in range(n_bins):
        mask = bins == index
        count = int(mask.sum())
        rows.append({"lower": index / n_bins, "upper": (index + 1) / n_bins,
                     "count": count,
                     "mean_probability": float(p[mask].mean()) if count else None,
                     "observed_rate": float(y[mask].mean()) if count else None})
    return rows


def score_probabilities(y, probabilities) -> dict:
    p = np.asarray(probabilities)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid probabilities")
    metrics = evaluate(np.asarray(y), p, math.ceil(.1 * len(y)))
    bins = reliability_table(y, p)
    metrics.update(brier_score=float(brier_score_loss(y, p)),
                   log_loss=float(log_loss(y, p, labels=[0, 1])),
                   overall_conversion_rate=float(np.mean(y)),
                   top_10_percent_conversion_rate=metrics["precision_at_k"],
                   reliability_bins=bins,
                   ece=float(sum(b["count"] * abs(b["mean_probability"] - b["observed_rate"])
                                 for b in bins if b["count"]) / len(y)))
    return metrics


def choose_models(validation_metrics: dict) -> dict:
    """Pure selection function: accepts validation metrics only, never test data."""
    variants, calibration_decisions = {}, {}
    for family in FAMILIES:
        raw, calibrated = validation_metrics[family], validation_metrics[family + "_sigmoid"]
        gains = {"brier_reduction": raw["brier_score"] - calibrated["brier_score"],
                 "log_loss_reduction": raw["log_loss"] - calibrated["log_loss"],
                 "ap_loss": raw["pr_auc_average_precision"] - calibrated["pr_auc_average_precision"],
                 "lift_loss": raw["lift_at_k"] - calibrated["lift_at_k"]}
        adopt = (gains["brier_reduction"] >= POLICY["calibration_min_brier_reduction"]
                 and gains["log_loss_reduction"] >= POLICY["calibration_min_log_loss_reduction"]
                 and gains["ap_loss"] <= POLICY["calibration_max_ap_loss"]
                 and gains["lift_loss"] <= POLICY["calibration_max_lift_loss"])
        variants[family] = family + "_sigmoid" if adopt else family
        calibration_decisions[family] = {"adopt_sigmoid": adopt, **gains}
    baseline, challenger = [validation_metrics[variants[f]] for f in FAMILIES]
    gains = {"ap_gain": challenger["pr_auc_average_precision"] - baseline["pr_auc_average_precision"],
             "lift_gain": challenger["lift_at_k"] - baseline["lift_at_k"],
             "brier_increase": challenger["brier_score"] - baseline["brier_score"]}
    adopt = (gains["ap_gain"] >= POLICY["challenger_min_ap_gain"]
             and gains["lift_gain"] >= POLICY["challenger_min_lift_gain"]
             and gains["brier_increase"] <= POLICY["challenger_max_brier_increase"])
    return {"decision": "ADOPT CHALLENGER" if adopt else "KEEP BASELINE",
            "selected_model": variants[FAMILIES[int(adopt)]], "family_variants": variants,
            "calibration_decisions": calibration_decisions, "challenger_gains": gains,
            "policy": POLICY.copy(),
            "rationale": "Require AP gain >= 0.02 AND lift gain >= 0.15 AND Brier increase <= 0.005; "
                         "otherwise retain Logistic Regression. Calibration requires Brier reduction >= 0.002 "
                         "AND log-loss reduction >= 0.005 with AP loss <= 0.01 and lift loss <= 0.10. "
                         "All comparisons use the same training-only validation cohort."}


def compare_training(fit, validation, seed: int) -> tuple[dict, dict, dict]:
    metrics, signals = {}, {}
    for family in FAMILIES:
        for calibrated in (False, True):
            name = family + "_sigmoid" if calibrated else family
            model = candidate(family, calibrated, seed).fit(fit[FEATURES], fit[TARGET])
            metrics[name] = score_probabilities(validation[TARGET], model.predict_proba(validation[FEATURES])[:, 1])
            if not calibrated:
                importance = permutation_importance(
                    model, validation[FEATURES], validation[TARGET],
                    scoring="average_precision", n_repeats=3, random_state=seed, n_jobs=1)
                signals[family] = {"validation_permutation_ap_drop": sorted([
                    {"feature": feature, "mean": float(mean), "std": float(std)}
                    for feature, mean, std in zip(FEATURES, importance.importances_mean,
                                                 importance.importances_std)],
                    key=lambda row: row["mean"], reverse=True)}
                if family == FAMILIES[0]:
                    names = model.named_steps["preprocess"].get_feature_names_out()
                    coefficients = model.named_steps["classifier"].coef_[0]
                    signals[family]["transformed_coefficients"] = sorted([
                        {"feature": str(name), "coefficient": float(coef)}
                        for name, coef in zip(names, coefficients)],
                        key=lambda row: abs(row["coefficient"]), reverse=True)
    return metrics, choose_models(metrics), signals


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def run_comparison(history: Path, output: Path, artifacts: Path) -> dict:
    if output.resolve() == history.resolve():
        raise ValueError("Report historis tidak boleh ditimpa")
    historical_bytes = history.read_bytes()
    protocol = json.loads(historical_bytes)["protocol"]
    seed = protocol["seed"]
    data = generate_leads(protocol["n_rows"], seed)
    validate_schema(data)
    fingerprint = hashlib.sha256(data.to_csv(index=False).encode()).hexdigest()
    if fingerprint != protocol["dataset_sha256"] or protocol["test_fraction"] != .2:
        raise ValueError("Data or split differs from v0.1; do not silently replace the holdout")
    train, fit, validation, test = split_data(data, seed)
    with threadpool_limits(limits=1):
        validation_metrics, selection, signals = compare_training(fit, validation, seed)
        # Persist the decision BEFORE any test predictions. Final metrics cannot feed selection.
        write_json(artifacts / "selection_frozen.json", selection)
        final_models = {}
        names = list(dict.fromkeys([*FAMILIES, *selection["family_variants"].values()]))
        for name in names:
            family = name.removesuffix("_sigmoid")
            model = candidate(family, name.endswith("_sigmoid"), seed).fit(train[FEATURES], train[TARGET])
            final_models[name] = score_probabilities(test[TARGET], model.predict_proba(test[FEATURES])[:, 1])
            if name == selection["selected_model"]:
                joblib.dump(model, artifacts / "selected_model.joblib")
    report = {
        "version": "0.2", "seed": seed, "dataset_sha256": fingerprint,
        "historical_report_sha256": hashlib.sha256(historical_bytes).hexdigest(),
        "split": {"train_ids": train.lead_id.tolist(), "fit_ids": fit.lead_id.tolist(),
                  "validation_ids": validation.lead_id.tolist(), "test_ids": test.lead_id.tolist(),
                  "test_reused_from_v01": True},
        "candidate_models": {family: {
            "parameters": candidate(family, False, seed).named_steps["classifier"].get_params(),
            "complexity": "LOW" if family == FAMILIES[0] else "MEDIUM",
        } for family in FAMILIES},
        "calibration_protocol": "Raw vs sigmoid; inner stratified 3-fold OOF calibration on fit rows only. "
                                "ensemble=False refits estimator on all fit rows. Isotonic excluded a priori "
                                "because modest positive counts favor a lower-variance sigmoid.",
        "validation_metrics": validation_metrics, "selection": selection,
        "final_held_out_metrics": final_models, "signals": signals,
        "versions": {"python": platform.python_version(), **{p: version(p) for p in
                     ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "threadpoolctl"]}},
    }
    write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=Path("reports/baseline_metrics.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/model_comparison.json"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/v02"))
    args = parser.parse_args()
    if args.output.resolve() == args.history.resolve():
        parser.error("Report historis tidak boleh ditimpa")
    report = run_comparison(args.history, args.output, args.artifacts)
    print("Hasil model selection beku dan evaluasi holdout (field report tetap berbahasa Inggris):")
    print(json.dumps({"selection": report["selection"],
                      "final_held_out_metrics": report["final_held_out_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
