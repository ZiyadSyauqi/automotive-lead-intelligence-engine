# Product and data contract

## Decision and user

The user is a dealership sales representative or team lead deciding which eligible
leads to contact next with limited staff time. The eventual product ranks leads,
assigns priority, suggests a next action, and explains the recommendation. v0.1
implements only the offline probability/ranking baseline and its evaluation.

**Prediction:** probability that an open lead completes a vehicle purchase within
the 30 days after a scoring snapshot, conditional on observed interactions and the
assumed sales process. A completed purchase is the conceptual target; the public
dataset contains a fictional binary draw, with no invoice or customer records.

**Snapshot:** seven days after lead creation, restricted to leads still open and
not converted at that point. All interaction counts and statuses describe only
that first seven-day window. Purchase timeline is the prospect's estimate of days
remaining as of scoring. Each row is a different fictional lead. A requested or
scheduled activity was recorded before the snapshot; a completed or attended
activity happened before it. Response latency is the mean customer response time
to outbound contact in that window, not a future salesperson response SLA.

This first cohort deliberately excludes immediate scoring of brand-new leads.
Applying its model to day-zero leads or daily repeat snapshots requires a revised
data contract, evaluation, and grouping by customer. In a real dataset, only
snapshots with a complete 30-day outcome window would be eligible for training;
unresolved outcomes must not be silently labeled negative.

## Intended action and boundaries

The future queue will recommend whom to contact and when, with a human retaining
control. A contact-within-two-hours action is a possible v0.3 policy, not a
learned causal conclusion or an implemented service. Contact consent, suppression
lists, business hours, duplicate handling, and reachable status must gate a real
queue. The current top-K evaluation ranks the full synthetic cohort; it does not
yet apply those eligibility rules or promise K successful contacts.

Conversion propensity is not the incremental benefit of contacting a lead. High
propensity leads may buy anyway. This project does not measure causal uplift,
revenue, salesperson productivity gains, financing eligibility, customer worth,
or fairness in real deployment. It does not deny service, approve credit, send
messages, ingest a CRM, or use an LLM. No personal identifiers or protected
attributes are generated. Their absence does not establish fairness.

## Synthetic schema

| Field | Values / units | Snapshot meaning |
|---|---|---|
| lead_id | Unique integer | Bookkeeping only; excluded from ML |
| purchase_timeline_days | 1–365; sometimes missing | Stated days until intended purchase |
| contactability | reachable / intermittent / unreachable | Observed contact status |
| financing_interest | yes / no / undecided; sometimes missing | Expressed interest, not creditworthiness |
| test_drive_activity | none / requested / completed | Latest recorded status |
| vehicle_price_segment | entry / mid / premium | Fictional relative segment; no price or brand |
| previous_interactions | 0–20 | Customer interactions, excluding outbound follow-up attempts |
| follow_up_count | 0–12 | Outbound follow-up attempts |
| lead_source | website / marketplace / walk_in / referral / event | Acquisition channel |
| trade_in_interest | yes / no / undecided | Expressed interest |
| response_latency_hours | 0.1–168; sometimes missing | Mean observed customer response delay |
| appointment_activity | none / scheduled / attended / missed | Latest appointment status |
| converted | 0 / 1 | Purchase after snapshot within 30 days; target only |

No proprietary records, organization-specific terminology, credentials, or
confidential policies are used. The schema and relationships are author-designed
assumptions, not claims about real automotive customers. Test drives and sales
appointments are separate activities, so a test drive does not require the
appointment field to be attended.

## Generating assumptions

A private normal latent-intent variable correlates timelines, interactions,
response delays, appointments, and test drives. A logistic score combines these
observations with hidden intent and additional normal noise (standard deviation
0.85). Short timelines, completed drives, and attended appointments have stronger
positive effects; financing, trade-in, source, and segment are deliberately weak.
Interaction effects, saturated interaction counts, and penalties for excessive
follow-ups depart from a purely linear relationship. These are associations in
the simulator, not estimates of what changing a feature would cause.

Probability is `0.025 + 0.95 * sigmoid(score)`; a Bernoulli draw produces the label.
Thus even high-intent leads can fail and low-intent leads can convert. Roughly 5%
recording gaps are added to timeline, latency, and financing. Unreachable leads
have no measured latency. Missingness is added after outcome generation from
complete latent observations, so the model sees less information than the
simulator. Hidden intent, score, and probability are never exported.

## Leakage and evaluation protocol

- Split independent leads 80/20 with stratification and seed 42 before exploration
  or fitting. Explore training data only; hold test data for the single fixed baseline.
- Fit imputation, scaling, category encoding, and Logistic Regression only on
  training features using Pipeline / ColumnTransformer. Explicitly exclude ID and
  target; do not derive fields from post-purchase activity.
- Automated checks catch schema changes, duplicate IDs, invalid targets, and exact
  numeric target copies. They cannot prove point-in-time correctness. With real
  data, audit timestamps, duplicated people, censoring, and sales-process effects;
  use a temporal holdout and customer grouping instead of assuming IID rows.
- Use unweighted Logistic Regression, C=1, max_iter=1000. Avoid class reweighting
  merely to raise recall: it changes probability interpretation. Threshold 0.5 is
  fixed for reference metrics; it is not a sales policy.
- PR-AUC here means average precision. Confusion matrix rows are actual 0/1 and
  columns predicted 0/1. Zero-division threshold metrics return zero.
- Capacity defaults to top 10% of test leads, `K = ceil(fraction * test_size)`.
  Precision@K is conversions among selected / K; Recall@K is selected conversions
  / all conversions; Lift@K is Precision@K / test conversion prevalence. Random
  ranking expectations are prevalence, K/N, and 1, respectively. Stable input order
  breaks score ties. For no positives, recall and lift are undefined (`null`).

Ranking metrics answer whether a fixed contact budget concentrates likely buyers.
They do not establish additional sales caused by the queue. Reported probabilities
are fitted synthetic propensities, not validated real-world probabilities. The
single held-out estimate has sampling uncertainty; no confidence interval or
calibration claim is made. Do not tune models, K, or thresholds to this test set in
v0.2: use training-only cross-validation for selection and retain this baseline
protocol for comparison. Repeated test comparisons eventually require a new holdout.
