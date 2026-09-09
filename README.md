# Automotive Lead Intelligence Engine

## Problem

Dealership sales teams may receive more leads than they can immediately follow
up. This creates a prioritization problem: which leads should receive their
limited attention first?

## Product concept

Lead Data → Conversion Model → Ranked Leads → later Priority / Recommended Action / Explanation

The current user is a sales representative or team lead reviewing a ranked queue.
The model estimates purchase probability within 30 days after a scoring snapshot.
For this first cohort, the snapshot is seven days after lead creation, for leads
still open then. It does not yet score brand-new leads at arrival.

## Current version — v0.1

- Reproducible 5,000-lead synthetic generator with numerical and categorical inputs.
- Training-only imputation, scaling, and OneHotEncoder in a ColumnTransformer/Pipeline.
- Logistic Regression with an 80/20 stratified split and seed 42.
- Conversion probabilities, ranked leads, threshold metrics, and top-10% evaluation.
- A printed table of the top 10 held-out leads: `lead_id`, `conversion_probability`,
  and `actual_conversion`. Actual outcomes are for offline evaluation only.

Priority rules, recommended actions, explanations, stronger models, calibration,
FastAPI, Docker, LLMs, SHAP, frontend, cloud deployment, MLflow, Kubernetes,
databases, and CI/CD are not implemented.

## Synthetic data disclaimer

All leads are fictional because this is a public portfolio without a suitable
real customer dataset. The simulated relationships are demonstration assumptions,
**not empirical claims about automotive customers**. No company data or credentials
are used. No generated CSV or model binary is committed.

Inputs cover purchase timeline, contactability, financing, test drives,
appointments, trade-in interest, follow-ups, interactions, response latency,
source, and vehicle segment. Existing categorical statuses preserve more detail
than booleans (for example, requested versus completed test drive). The binary
target is `converted`.

Hidden intent, random noise, nonlinear effects, missing observations, and a
Bernoulli outcome draw create overlapping classes. Financing, source, and vehicle
segment are weak predictors. Hidden generating scores/probabilities, lead ID, and
target are excluded from training. Preprocessing fits only on training data.
See [product and data contract](docs/product-and-data.md) for the exact schema,
label timing, and leakage assumptions, and [exploratory findings](docs/week1-findings.md)
for training-only validation.

## Evaluation

Actual execution with 5,000 leads, seed 42, Python 3.12.14, and 1,000 test leads:

| Metric | Result |
|---|---:|
| Overall conversion rate (test cohort; lift denominator) | 21.50% (215/1,000) |
| Full generated dataset conversion rate | 21.46% (1,073/5,000) |
| Top-10% conversion rate | 59.00% (59/100) |
| Precision@10% | 0.5900 |
| Recall@10% | 0.2744 |
| Lift@10% | 2.7442× |
| ROC-AUC | 0.7502 |
| PR-AUC (average precision) | 0.4822 |
| Precision at threshold 0.5 | 0.6800 |
| Recall at threshold 0.5 | 0.1581 |
| F1 at threshold 0.5 | 0.2566 |

Confusion matrix at 0.5 (rows actual 0/1, columns predicted 0/1):
`[[769, 16], [181, 34]]`. Full results, top 10 leads, dependency versions, and data
fingerprint are in [baseline_metrics.json](reports/baseline_metrics.json).

For a sales team prioritizing 10% of leads, K = ceil(0.10 × test cohort size).
Precision@10% = selected conversions / K; Recall@10% = selected conversions / all
test conversions; Lift@10% = selected conversion rate / overall test conversion
rate. Stable input order breaks score ties. The capacity fraction is configurable.

The queue captures 59 of 215 converters; random selection would capture 21.5 in
expectation. It still includes 41 nonconverters and misses 156 converters. AUC
0.7502 and these errors show this is not a near-perfect synthetic exercise.
**Lift measures concentration, not additional sales caused by contacting leads.**
Real-world probability calibration and business value remain unvalidated.

## How to run

Use Python 3.11+ from the repository root, preferably in a virtual environment:

```bash
python -m pip install -r requirements.txt
python scripts/train_baseline.py
python -m pytest -q
```

`requirements.txt` installs the local package and pytest; dependency pins live in
`pyproject.toml` to avoid maintaining two version lists. The command generates
small JSON reports in `reports/`, prints metrics and the ranked table, and saves
an ignored local pipeline in `artifacts/baseline.joblib`. Re-running overwrites
these outputs. Only load trusted joblib files.

To evaluate a different staffing budget without overwriting the recorded run:

```bash
python scripts/train_baseline.py --capacity-fraction 0.2 --output artifacts/capacity-check --artifacts artifacts/capacity-check
```

Choose capacity from staffing, not test-set results. The baseline model and
threshold are fixed; future model selection must use training-only validation.

Code is in one `src/lead_intelligence/` package: `data.py`, `model.py`,
`evaluation.py`, `validation.py`, and `experiment.py`. The small `scripts/` entry
point delegates to that package. Five tests cover generation, schema/target,
missing/unseen inference, probabilities, serialization, ranking calculations,
and held-out table/label alignment. There are no empty architectural layers.

## Roadmap

| Version | Scope |
|---|---|
| v0.1 | Baseline ranking system — implemented |
| v0.2 | Compare one stronger tree-based model, calibration, explainability |
| v0.3 | Decision layer: probability → priority → recommended action |
| v0.4 | FastAPI |
| v0.5 | Docker and final portfolio polish |

MIT licensed; see [LICENSE](LICENSE).
