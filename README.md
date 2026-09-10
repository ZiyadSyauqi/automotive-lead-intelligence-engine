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

## v0.1 — baseline lead ranking

- Reproducible 5,000-lead synthetic generator with numerical and categorical inputs.
- Training-only imputation, scaling, and OneHotEncoder in a ColumnTransformer/Pipeline.
- Logistic Regression with an 80/20 stratified split and seed 42.
- Conversion probabilities, ranked leads, threshold metrics, and top-10% evaluation.
- A printed table of the top 10 held-out leads: `lead_id`, `conversion_probability`,
  and `actual_conversion`. Actual outcomes are for offline evaluation only.

Priority rules, recommended actions, FastAPI, Docker, LLMs, SHAP, frontend,
cloud deployment, MLflow, Kubernetes, databases, and CI/CD are not implemented.

## v0.2 — comparison, calibration, and model reliance

Added exactly one challenger (HistGradientBoosting), training-only sigmoid
calibration evaluation, and lightweight feature-reliance analysis. The original
v0.1 report is preserved. [Audit and protocol](docs/model-selection.md) document
what was checked against the code and how selection was frozen before testing.

The original 1,000-lead test set stays separate. The original 4,000 training leads
are split into 3,000 fit / 1,000 validation leads. Calibration uses three-fold CV
inside the fit portion, including preprocessing. There is one fixed configuration
per family, with raw and sigmoid variants; no broad tuning search.

Actual **validation** results (the source of model selection):

| Model | PR-AUC (AP) | Lift@10% | Brier | Complexity |
|---|---:|---:|---:|---|
| Logistic Regression | 0.4677 | 2.5701 | 0.1450 | LOW |
| HistGradientBoosting | 0.4648 | 2.5701 | 0.1450 | MEDIUM |

**KEEP BASELINE — uncalibrated Logistic Regression.** Challenger AP gain was
-0.002910 and lift gain was zero. Neither meets the predeclared requirements of
at least +0.02 AP and +0.15 lift, with at most +0.005 Brier deterioration.
Complexity did not earn its place. These are project criteria, not significance tests.

Calibration was also rejected on validation evidence:

| Model | Raw Brier → sigmoid | Raw log loss → sigmoid |
|---|---|---|
| Logistic Regression | 0.144997 → 0.145697 | 0.454130 → 0.456108 |
| HistGradientBoosting | 0.145036 → 0.145426 | 0.454597 → 0.456020 |

Lower is better for both scores. Sigmoid made both worse. Ten-bin reliability
tables, counts, ECE, and full metrics appear in
[model_comparison.json](reports/model_comparison.json). Isotonic was excluded in
advance to limit calibration flexibility with modest positive sample counts.

Final **held-out** evaluation after freezing the decision:

| Metric | Logistic Regression | HistGradientBoosting |
|---|---:|---:|
| ROC-AUC | 0.7502 | 0.7543 |
| PR-AUC (AP) | 0.4822 | 0.4961 |
| Precision@10% / top-10% conversion rate | 59.00% | 62.00% |
| Recall@10% | 27.44% | 28.84% |
| Lift@10% | 2.7442× | 2.8837× |
| Brier | 0.143269 | 0.141717 |
| Log loss | 0.451095 | 0.446661 |
| Overall test conversion rate | 21.50% | 21.50% |

The challenger scored better on this test sample, but changing the choice now
would contaminate model selection. The baseline remains selected. The original
v0.1 baseline ranking metrics reproduce; no generator or threshold was retuned.

**Ranking and calibration are different.** Ranking determines who reaches the top
of the queue; calibration asks whether assigned probabilities match observed
rates. A monotonic sigmoid can change probabilities while leaving rankings
unchanged. Future probability thresholds need calibration evidence, not just good
lift. All probabilities here still describe synthetic data, not real customers.

The baseline's top validation permutation signals are **test-drive activity,
appointment activity, and purchase timeline**. The report also includes transformed
Logistic Regression coefficients and challenger permutation importance. These
measure fitted-model reliance, not causal effects; correlated inputs and imputation
complicate interpretation. No SHAP or customer-level explanation service is added.

The main remaining limitation is synthetic-only validation. One validation split
also has sampling uncertainty. The preserved test set was already viewed in v0.1;
repeated test comparisons must not become a tuning loop.

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
python scripts/compare_models.py
python -m pytest -q
```

`requirements.txt` installs the local package and pytest; dependency pins live in
`pyproject.toml` to avoid maintaining two version lists. The v0.2 command writes `reports/model_comparison.json`, freezes selection in
`artifacts/v02/selection_frozen.json` before test predictions, and saves
`artifacts/v02/selected_model.joblib`. It never overwrites the historical v0.1
report. Re-running reproduces the frozen experiment and overwrites its v0.2
outputs; it must not guide further test-based choices. Only load trusted joblib files.

The v0.1 command remains `python scripts/train_baseline.py`; use a separate output
folder when rerunning it to preserve the recorded historical metrics.

To evaluate a different staffing budget without overwriting the recorded run:

```bash
python scripts/train_baseline.py --capacity-fraction 0.2 --output artifacts/capacity-check --artifacts artifacts/capacity-check
```

Choose capacity from staffing, not test-set results. The baseline model and
threshold are fixed; future model selection must use training-only validation.

Code is in one `src/lead_intelligence/` package: `data.py`, `model.py`,
`evaluation.py`, `validation.py`, and `experiment.py`. The small `scripts/` entry
point delegates to that package. Tests cover generation, schema/target,
missing/unseen inference, probabilities, serialization, ranking calculations,
and held-out table/label alignment. There are no empty architectural layers.

## Roadmap

| Version | Scope |
|---|---|
| v0.1 | Baseline ranking system — implemented |
| v0.2 | Model comparison, calibration, lightweight explainability — implemented |
| v0.3 | Decision layer: probability → priority → recommended action |
| v0.4 | FastAPI |
| v0.5 | Docker and final portfolio polish |

MIT licensed; see [LICENSE](LICENSE).
