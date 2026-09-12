from dataclasses import asdict

import joblib
import numpy as np
import pandas as pd
import pytest

from lead_intelligence.data import FEATURES, generate_leads
from lead_intelligence.decision import (ACTIONS, PRIORITIES, DecisionResult, decide,
    decide_batch, format_decision, summarize_decisions)
from lead_intelligence.decision_demo import load_selected_model
from lead_intelligence.model import build_pipeline


def lead(**overrides):
    return dict(lead_id=1, contactability='reachable', appointment_activity='none',
                test_drive_activity='none', purchase_timeline_days=10,
                response_latency_hours=12, follow_up_count=1, previous_interactions=1) | overrides


@pytest.mark.parametrize('changes,pct,priority,action,reason', [
    ({}, 95, 'URGENT', 'CONTACT_NOW', 'SHORT_PURCHASE_TIMELINE'),
    ({}, 20, 'LOW', 'NURTURE', 'LOW_RECENT_ENGAGEMENT'),
    ({'contactability': 'unreachable'}, 95, 'MEDIUM', 'RETRY_CONTACT', 'NOT_CONTACTABLE'),
    ({'contactability': 'intermittent'}, 95, 'MEDIUM', 'RETRY_CONTACT', 'INTERMITTENT_CONTACT'),
    ({'appointment_activity': 'scheduled'}, 65, 'HIGH', 'CONFIRM_APPOINTMENT', 'APPOINTMENT_SCHEDULED'),
    ({'test_drive_activity': 'completed'}, 20, 'MEDIUM', 'FOLLOW_UP_TEST_DRIVE', 'TEST_DRIVE_COMPLETED'),
    ({'purchase_timeline_days': 90}, 95, 'HIGH', 'STANDARD_FOLLOW_UP', 'HIGH_MODEL_SCORE'),
    ({'previous_interactions': 3}, 20, 'MEDIUM', 'STANDARD_FOLLOW_UP', 'MEANINGFUL_ENGAGEMENT'),
    ({'follow_up_count': 5, 'response_latency_hours': 48}, 95, 'MEDIUM', 'NURTURE', 'FOLLOW_UP_LIMIT_REACHED'),
    ({'contactability': pd.NA}, 95, 'MEDIUM', 'REVIEW_LEAD_DATA', 'UNKNOWN_OPERATIONAL_STATUS'),
    ({'appointment_activity': 'new_status'}, 95, 'MEDIUM', 'REVIEW_LEAD_DATA', 'UNKNOWN_OPERATIONAL_STATUS'),
    ({'test_drive_activity': None}, 95, 'MEDIUM', 'REVIEW_LEAD_DATA', 'UNKNOWN_OPERATIONAL_STATUS'),
])
def test_policy_routes(changes, pct, priority, action, reason):
    result = decide(lead(**changes), .7, pct)
    assert result == decide(lead(**changes), .7, pct)
    assert result.priority == priority and result.priority in PRIORITIES
    assert result.recommended_action == action and action in ACTIONS
    assert reason in result.reason_codes
    assert 'Alasan:' in format_decision(result)


@pytest.mark.parametrize('value', [None, np.nan, -1, float('inf'), 'bad', True])
def test_missing_or_invalid_numeric_never_creates_urgent(value):
    result = decide(lead(purchase_timeline_days=value), .9, 99)
    assert result.priority != 'URGENT'
    assert 'OPERATIONAL_DATA_INCOMPLETE' in result.reason_codes


def test_threshold_boundaries_and_precedence():
    assert decide(lead(purchase_timeline_days=14), .7, 90).priority == 'URGENT'
    assert decide(lead(purchase_timeline_days=15), .7, 90).priority == 'HIGH'
    assert decide(lead(), .7, 89.9).priority == 'MEDIUM'
    assert decide(lead(appointment_activity='scheduled', follow_up_count=8,
                       response_latency_hours=100), .5, 70).recommended_action == 'CONFIRM_APPOINTMENT'
    with pytest.raises(ValueError):
        decide(lead(), np.nan, 99)


def test_batch_alignment_ties_and_label_independence():
    leads = pd.DataFrame([lead(lead_id=i) for i in range(20)])
    scores = pd.Series(np.arange(20) / 20, index=np.arange(20)).sample(frac=1, random_state=2)
    result = decide_batch(leads, scores)
    assert result.lead_id.tolist() == leads.lead_id.tolist()
    np.testing.assert_allclose(result.model_score, np.arange(20) / 20)
    assert result.iloc[-1].score_percentile == 95
    assert result.iloc[-1].priority == 'URGENT'
    assert summarize_decisions(result)['priority_distribution']['URGENT']['count'] == 2
    reordered = decide_batch(leads.iloc[::-1], scores).sort_values('lead_id').reset_index(drop=True)
    pd.testing.assert_frame_equal(result, reordered)
    pd.testing.assert_frame_equal(result, decide_batch(leads.assign(converted=1), scores))
    tied = decide_batch(leads, pd.Series(.9, index=leads.lead_id))
    assert tied.score_percentile.eq(0).all() and not tied.priority.eq('URGENT').any()
    with pytest.raises(ValueError):
        decide_batch(leads, scores.iloc[:-1])
    with pytest.raises(ValueError):
        decide_batch(pd.concat([leads, leads.iloc[:1]]), scores)


def test_sanity_rejects_contradictory_output():
    row = {'lead_id': 1, 'model_score': .9, 'score_percentile': 99,
           **asdict(DecisionResult('LOW', 'CONTACT_NOW', ('LOW_MODEL_SCORE',)))}
    with pytest.raises(ValueError, match='CONTACT_NOW'):
        summarize_decisions(pd.DataFrame([row]))


def test_selected_artifact_contract(tmp_path):
    data = generate_leads(300)
    model = build_pipeline().fit(data[FEATURES], data.converted)
    report = {'selection': {'selected_model': 'logistic_regression'}, 'candidate_models': {
        'logistic_regression': {'parameters': model.named_steps['classifier'].get_params()}}}
    path = tmp_path / 'selected.joblib'
    joblib.dump(model, path)
    loaded = load_selected_model(path, report)
    np.testing.assert_allclose(loaded.predict_proba(data[FEATURES]), model.predict_proba(data[FEATURES]))
    report['selection']['selected_model'] = 'hist_gradient_boosting'
    with pytest.raises(ValueError, match='v0.2'):
        load_selected_model(path, report)
