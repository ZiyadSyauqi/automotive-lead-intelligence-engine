# v0.2 audit and model selection

## Audit of v0.1

Audited main at `a61d747ff5cfa9f3da9d9ea4f999e1cb1162788c`, including the generator,
feature allowlist, preprocessing, split, metrics, tests, README, and JSON reports.
Local source/report Git blob hashes matched the remote files. No material leakage
or metric error was found, so the generator and original model were preserved.

- Inputs precede the outcome by contract; hidden intent, generating probability,
  ID, and target do not enter the feature allowlist. There are no real timestamps
  to independently verify that timing assumption.
- Median imputation, scaling, and category encoding fit inside Pipeline on the
  training portion. Validation/test transforms do not refit preprocessing.
- The seeded 80/20 stratified split is by independent fictional lead. Top-K uses
  descending probability, stable ties, and the test prevalence as the lift denominator.
- PR-AUC means average precision, not trapezoidal PR integration. Precision and
  recall at 0.5 are separate from the capacity-based ranking measurements.
- Bernoulli outcomes, hidden noise, and overlap prevent deterministic labels.
  The generator deliberately favors several signals; this is a simulator bias,
  not leakage or market evidence. It was not changed to improve challenger scores.
- Final Logistic Regression metrics reproduce v0.1, including ROC-AUC 0.750236,
  AP 0.482240, and 59 conversions among the top 100. The historical JSON is unchanged.

The test set has already been viewed in v0.1. We preserve its membership and avoid
using it for v0.2 choices; it cannot honestly be called a never-seen dataset.

## Protocol frozen before execution

Generate the exact v0.1 dataset from its report's seed and row count. Check its
SHA-256 before fitting. Reproduce its 4,000/1,000 outer split exactly. Split the
4,000 training rows into 3,000 fit and 1,000 validation rows, stratified with seed 42.
Store row IDs in the comparison report so membership is auditable.

Compare two model families, with one configuration each:

- Original Logistic Regression: C=1, lbfgs, max_iter=1000.
- HistGradientBoosting: 100 iterations, learning_rate=0.05, max_leaf_nodes=7,
  min_samples_leaf=40, l2_regularization=1.0, early_stopping=False, seed 42.

The small tree limits variance and can learn threshold/interaction effects. Dense
one-hot preprocessing is practical for these few low-cardinality categories.
Scaling is retained for preprocessing consistency; trees do not require it.
No hyperparameter search or extra model family is added.

Each family has a raw and sigmoid-calibrated candidate. CalibratedClassifierCV
uses stratified three-fold out-of-fold predictions inside the 3,000 fit rows only.
The complete preprocessing pipeline is refitted inside every inner fold.
`ensemble=False` fits the calibrator to inner OOF predictions and refits its base
estimator on all fit rows. All four candidates are then scored on the same 1,000
validation leads. Sigmoid was chosen a priori over isotonic to limit flexibility
with only hundreds of positive examples; we did not compare methods on test data.

Adopt calibration only if validation Brier improves by at least 0.002 AND log loss
by at least 0.005, with AP loss at most 0.01 and lift loss at most 0.10. Evaluate
this modest calibration candidate for both models even if bin statistics do not
show obvious severe miscalibration.

After each family's calibration decision, adopt the challenger only if validation
AP improves by at least 0.02 AND lift by at least 0.15, with Brier deterioration no
more than 0.005. Otherwise keep Logistic Regression. At 21.4% validation prevalence,
a 0.15 lift gain means roughly three extra observed converters in a 100-lead queue.
These are practical demonstration criteria, not measured financial returns or
statistical significance tests. AP assesses ranking across recall levels; top-10%
lift tests the actual capacity assumption. ROC-AUC alone cannot win selection.

Persist `artifacts/v02/selection_frozen.json` before making any test predictions.
Refit the raw baseline, raw challenger, and any adopted calibration variant on the
4,000 training rows. Evaluate each once on the original test rows. Reuse the chosen
model's entry as its final evaluation; do not select again. The main experiment
ran once after the protocol was fixed. Automated tests use a separate 500-row
fixture, not the real holdout. Reproduction will necessarily reevaluate the same
frozen experiment; it is not authorization to iterate choices on test results.

## Executed decision

**KEEP BASELINE: uncalibrated Logistic Regression.** On validation, challenger AP
was 0.464771 versus baseline 0.467682 (difference -0.002910), and both had lift
2.570093. Added model complexity did not meet either improvement requirement.

Sigmoid worsened validation Brier/log loss for both models:

| Candidate | Validation AP | Lift@10% | Brier | Log loss |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.467682 | 2.570093 | 0.144997 | 0.454130 |
| Logistic Regression + sigmoid | 0.467682 | 2.570093 | 0.145697 | 0.456108 |
| HistGradientBoosting | 0.464771 | 2.570093 | 0.145036 | 0.454597 |
| HistGradientBoosting + sigmoid | 0.464771 | 2.570093 | 0.145426 | 0.456020 |

The challenger happened to score better on final test data (AP 0.496096 versus
0.482240). **That does not reverse the validation decision.** Choosing it after
seeing this comparison would turn the holdout into another selection set.

## Ranking, calibration, and signals

Ranking determines queue order. Calibration asks whether groups assigned a given
probability convert at a similar rate. A monotonic sigmoid can change probability
values without changing ordering, as happened on this validation set. Numerical
Brier/log-loss results and ten fixed-width reliability bins with counts are in
`reports/model_comparison.json` for every candidate and final evaluated model.
Empty bins are explicit; sparse bins and bin-dependent ECE are not proof of good
calibration. Both proper scores also reflect predictive discrimination.

Future priority thresholds may use literal probabilities, so ranking performance
alone cannot justify them. Neither raw nor calibrated synthetic probabilities
are real-world conversion estimates.

Raw Logistic Regression coefficients are paired with transformed feature names:
numerical coefficients correspond to standardized, imputed inputs; categorical
coefficients are regularized one-hot terms (all levels are retained, so do not
interpret one coefficient as a reference-category contrast). Missing indicators
are named. Coefficients describe log-odds within the fitted model, not causality.

Validation-only permutation importance (AP decrease, three repeats) measures
original-feature reliance for both raw families. The selected baseline's largest
signals are test-drive activity (0.0997 AP drop), appointment activity (0.0602), and
purchase timeline (0.0586). Coefficients and importance come from fit-only models,
not from fitting or explaining the final test cohort. Correlated inputs can share
importance; shuffling may create unusual combinations. Three-repeat standard
deviations are permutation variability, not population confidence intervals.

## Limits

A single validation split can be noisy; no significance or real-world generalization
claim is made. The simulator defines the relationships, and its distributions lack
real CRM drift, repeated customers, timestamp audits, and censoring. The test set
must not become a tuning loop across future versions. No priority policy, API,
SHAP, infrastructure, or deployment is implemented here.
