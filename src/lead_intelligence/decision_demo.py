"""Scoring dengan model pilihan v0.2; keputusan tidak melihat label conversion."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from lead_intelligence.data import FEATURES, TARGET, generate_leads
from lead_intelligence.decision import POLICY, DecisionResult, decide_batch, format_decision, summarize_decisions
from lead_intelligence.model import build_pipeline


def load_selected_model(path: Path, report: dict):
    """Hanya load artifact lokal tepercaya. Struktur/parameter dicek; bukan bukti provenance kriptografis."""
    if report["selection"]["selected_model"] != "logistic_regression":
        raise ValueError("v0.3 ini membutuhkan model Logistic Regression pilihan v0.2")
    model = joblib.load(path)
    if (not isinstance(model, Pipeline)
            or not isinstance(model.named_steps.get("classifier"), LogisticRegression)
            or list(model.feature_names_in_) != FEATURES
            or model.named_steps["classifier"].get_params() != report["candidate_models"]["logistic_regression"]["parameters"]
            or list(model.classes_) != [0, 1]):
        raise ValueError("Artifact tidak cocok dengan kontrak model pilihan v0.2")
    return model


def run_decisions(report_path: Path, history_path: Path, model_path: Path,
                  output_dir: Path, rebuild_model: bool = False) -> dict:
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    if report["selection"]["selected_model"] != "logistic_regression":
        raise ValueError("Jangan membuka ulang model selection lewat script decision")
    history_bytes = history_path.read_bytes()
    if hashlib.sha256(history_bytes).hexdigest() != report["historical_report_sha256"]:
        raise ValueError("Report historis berubah; cek provenance sebelum scoring")
    protocol = json.loads(history_bytes)["protocol"]
    data = generate_leads(protocol["n_rows"], protocol["seed"])
    if hashlib.sha256(data.to_csv(index=False).encode()).hexdigest() != report["dataset_sha256"]:
        raise ValueError("Dataset tidak cocok dengan report v0.2")
    indexed = data.set_index("lead_id", drop=False)
    train_ids, test_ids = report["split"]["train_ids"], report["split"]["test_ids"]
    if (len(set(train_ids)) != len(train_ids) or len(set(test_ids)) != len(test_ids)
            or set(train_ids) & set(test_ids) or set(train_ids) | set(test_ids) != set(data.lead_id)):
        raise ValueError("Split v0.2 tidak valid")
    if rebuild_model:
        # Jalur fresh clone: refit pilihan yang SUDAH dibekukan, tidak compare/tune/evaluate.
        model = build_pipeline()
        if model.named_steps["classifier"].get_params() != report["candidate_models"]["logistic_regression"]["parameters"]:
            raise ValueError("Parameter baseline berubah dari keputusan v0.2")
        model.fit(indexed.loc[train_ids, FEATURES], indexed.loc[train_ids, TARGET])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
    if not model_path.exists():
        raise FileNotFoundError("Artifact belum ada. Jalankan sekali dengan --rebuild-model untuk refit pilihan beku.")
    model = load_selected_model(model_path, report)
    # Label test tidak diteruskan ke prediction, policy, summary, atau report decision.
    cohort = indexed.loc[test_ids, ["lead_id", *FEATURES]].reset_index(drop=True)
    scores = pd.Series(model.predict_proba(cohort[FEATURES])[:, 1], index=cohort.lead_id)
    decisions = decide_batch(cohort, scores)
    summary = {"version": "0.3", "selected_model": "logistic_regression",
               "comparison_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
               "model_artifact_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
               "policy": POLICY, "cohort": "v0.2 test membership; outcome tidak dipakai",
               **summarize_decisions(decisions)}
    ordered = decisions.sort_values("model_score", ascending=False, kind="stable")
    summary["examples"] = ordered.head(10).to_dict("records")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv = decisions.copy()
    csv["reason_codes"] = csv.reason_codes.map(json.dumps)
    csv.to_csv(output_dir / "decisions.csv", index=False)
    (output_dir / "decision_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Contoh hasil decision (score relatif, bukan peluang conversion tervalidasi):")
    print(ordered.head(10).to_string(index=False))
    print("\nDistribusi priority dan action:")
    print(json.dumps({k: summary[k] for k in ["priority_distribution", "action_distribution", "sanity_checks", "warnings"]}, indent=2))
    example = ordered.iloc[0]
    print("\n" + format_decision(DecisionResult(example.priority, example.recommended_action, example.reason_codes)))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("reports/model_comparison.json"), help="Report pilihan model yang dibekukan")
    parser.add_argument("--history", type=Path, default=Path("reports/baseline_metrics.json"), help="Report v0.1 untuk cek provenance")
    parser.add_argument("--model", type=Path, default=Path("artifacts/v02/selected_model.joblib"), help="Artifact lokal tepercaya")
    parser.add_argument("--output", type=Path, default=Path("artifacts/v03"), help="Folder output batch dan ringkasan")
    parser.add_argument("--rebuild-model", action="store_true", help="Refit model pilihan beku pada train rows; tanpa model selection")
    args = parser.parse_args()
    run_decisions(args.report, args.history, args.model, args.output, args.rebuild_model)


if __name__ == "__main__":
    main()
