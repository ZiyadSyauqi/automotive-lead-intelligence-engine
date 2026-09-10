import hashlib
import json

import joblib
import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from lead_intelligence.comparison import (FAMILIES, candidate, choose_models,
    reliability_table, run_comparison, split_data)
from lead_intelligence.data import CATEGORICAL, FEATURES, TARGET, generate_leads


def test_nested_partition_preserves_v01_holdout():
    data = generate_leads(500)
    train, fit, validation, test = split_data(data, 42)
    _, old_test = train_test_split(data, test_size=.2, random_state=42, stratify=data[TARGET])
    assert test.lead_id.tolist() == old_test.lead_id.tolist()
    assert set(fit.lead_id) | set(validation.lead_id) == set(train.lead_id)
    assert set(fit.lead_id).isdisjoint(validation.lead_id)
    assert set(train.lead_id).isdisjoint(test.lead_id)


def test_selection_rejects_tiny_gains_and_accepts_material_gains():
    base = {"pr_auc_average_precision": .4, "lift_at_k": 2.0,
            "brier_score": .15, "log_loss": .45}
    metrics = {name + suffix: base.copy() for name in FAMILIES for suffix in ['', '_sigmoid']}
    metrics[FAMILIES[1]]["pr_auc_average_precision"] = .401
    assert choose_models(metrics)["decision"] == 'KEEP BASELINE'
    metrics[FAMILIES[1]].update(pr_auc_average_precision=.45, lift_at_k=2.3)
    assert choose_models(metrics)["decision"] == 'ADOPT CHALLENGER'
    metrics[FAMILIES[0] + '_sigmoid'].update(brier_score=.14, log_loss=.43)
    assert choose_models(metrics)["family_variants"][FAMILIES[0]].endswith('_sigmoid')


@pytest.mark.parametrize('family', FAMILIES)
def test_calibrated_unseen_categories_and_serialization(family, tmp_path):
    fit = generate_leads(400)
    unseen = generate_leads(15, 91)
    unseen[CATEGORICAL] = 'unseen'
    with threadpool_limits(limits=1):
        model = candidate(family, True, 42).fit(fit[FEATURES], fit[TARGET])
        probabilities = model.predict_proba(unseen[FEATURES])
        assert np.isfinite(probabilities).all()
        assert ((probabilities >= 0) & (probabilities <= 1)).all()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        path = tmp_path / 'model.joblib'
        joblib.dump(model, path)
        np.testing.assert_allclose(joblib.load(path).predict_proba(unseen[FEATURES]), probabilities)


def test_reliability_bins_keep_endpoints_and_counts():
    rows = reliability_table([0, 1, 1], [0., .5, 1.])
    assert sum(row['count'] for row in rows) == 3
    assert rows[0]['mean_probability'] == 0
    assert rows[-1]['observed_rate'] == 1
    assert rows[1]['mean_probability'] is None


def test_report_and_frozen_selection_contract(tmp_path):
    data = generate_leads(500)
    history = tmp_path / 'history.json'
    history.write_text(json.dumps({'protocol': {'seed': 42, 'n_rows': 500, 'test_fraction': .2,
        'dataset_sha256': hashlib.sha256(data.to_csv(index=False).encode()).hexdigest()}}))
    before = history.read_bytes()
    artifact_dir = tmp_path / 'artifacts'
    report = run_comparison(history, tmp_path / 'comparison.json', artifact_dir)
    assert history.read_bytes() == before
    assert set(report['candidate_models']) == set(FAMILIES)
    assert len(report['validation_metrics']) == 4
    assert report['selection'] == json.loads((artifact_dir / 'selection_frozen.json').read_text())
    assert report['selection']['selected_model'] in report['final_held_out_metrics']
    assert report['selection'] == choose_models(report['validation_metrics'])
    for metrics in report['final_held_out_metrics'].values():
        assert {'roc_auc', 'pr_auc_average_precision', 'precision_at_k', 'recall_at_k',
                'lift_at_k', 'brier_score', 'log_loss', 'reliability_bins'} <= metrics.keys()
        assert sum(row['count'] for row in metrics['reliability_bins']) == 100
