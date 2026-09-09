import joblib
import numpy as np
import pandas as pd
import pytest

from lead_intelligence.data import CATEGORICAL, FEATURES, NUMERIC, TARGET, generate_leads
from lead_intelligence.evaluation import ranking_metrics
from lead_intelligence.model import build_pipeline
from lead_intelligence.validation import validate_schema


def test_generator_reproducibility_and_seed_sensitivity():
    pd.testing.assert_frame_equal(generate_leads(200, 42), generate_leads(200, 42))
    assert not generate_leads(200, 42).equals(generate_leads(200, 43))


def test_schema_target_and_domain_constraints():
    data = generate_leads()
    validate_schema(data)
    assert list(data.columns) == ["lead_id", *FEATURES, TARGET]
    assert set(data[TARGET]) == {0, 1}
    assert .1 < data[TARGET].mean() < .4  # Demonstration design, not a market estimate.
    assert data.purchase_timeline_days.dropna().between(1, 365).all()
    assert data.response_latency_hours.dropna().between(.1, 168).all()
    assert data.loc[data.contactability == "unreachable", "response_latency_hours"].isna().all()
    assert data.previous_interactions.between(0, 20).all()
    assert data.follow_up_count.between(0, 12).all()
    assert all(data[c].notna().all() for c in [TARGET, "lead_id"])
    with pytest.raises(ValueError, match="schema"):
        validate_schema(data.assign(final_invoice_amount=100))


def test_pipeline_train_only_imputation_unseen_categories_and_roundtrip(tmp_path):
    train = generate_leads(500)
    model = build_pipeline().fit(train[FEATURES], train[TARGET])
    statistics = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"].statistics_
    np.testing.assert_allclose(statistics, train[NUMERIC].median().to_numpy())
    holdout = generate_leads(20, 123)
    holdout.loc[:, CATEGORICAL] = "unseen_category"
    holdout[NUMERIC] = np.nan
    probabilities = model.predict_proba(holdout[FEATURES])
    assert probabilities.shape == (20, 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    # Dropped identifiers/outcomes must not alter scores, even if supplied accidentally.
    np.testing.assert_allclose(probabilities, model.predict_proba(holdout.assign(lead_id=-1, converted_30d=1)))
    np.testing.assert_allclose(statistics, train[NUMERIC].median().to_numpy())
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    np.testing.assert_allclose(joblib.load(path).predict_proba(holdout[FEATURES]), probabilities)


def test_ranking_metrics_hand_calculated_and_edge_cases():
    result = ranking_metrics(np.array([1, 0, 1, 0, 0]), np.array([.9, .8, .7, .6, .5]), 2)
    assert result["precision_at_k"] == .5
    assert result["recall_at_k"] == .5
    assert result["lift_at_k"] == 1.25
    assert ranking_metrics(np.array([1, 0]), np.array([.5, .5]), 1)["precision_at_k"] == 1
    assert ranking_metrics(np.array([1, 0]), np.array([.9, .1]), 2)["lift_at_k"] == 1
    empty_positive = ranking_metrics(np.zeros(3), np.ones(3), 1)
    assert empty_positive["recall_at_k"] is None
    assert empty_positive["lift_at_k"] is None
    for invalid_k in [0, 4, 1.5]:
        with pytest.raises(ValueError):
            ranking_metrics(np.zeros(3), np.ones(3), invalid_k)
    with pytest.raises(ValueError):
        ranking_metrics(np.array([0, 1]), np.array([np.nan, .5]), 1)
