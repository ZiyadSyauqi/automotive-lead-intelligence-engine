# v0.1 exploratory findings

Protocol: default 5,000-row generator, seed 42. These observations use only the
4,000 training rows. Full counts, distributions and rates are in
`reports/data_validation.json`; evaluation on held-out rows is separate.

| Question | Observation | Modeling / product consequence |
|---|---|---|
| Is conversion imbalanced? | 858 / 4,000 (21.45%) convert | Accuracy alone is inadequate; report PR and ranking metrics |
| Where is data incomplete? | Timeline: 191 missing; latency: 649; financing: 206; ID and outcome: zero | Fit median/missing-indicator and categorical imputation on training data |
| Does shorter stated timeline associate with conversion? | Lowest timeline quartile: 34.94%; highest: 11.18% | Useful signal with substantial overlap; neither group determines outcome |
| Does completed activity carry signal? | Completed test drive: 37.40%; no drive: 17.01%. Attended appointment: 31.22%; none: 17.79% | Keep pre-snapshot activity, but do not interpret association as treatment effect |
| Is source a reliable standalone ranking rule? | Source rates range only from 20.66% to 22.52% | Deliberately weak predictor; even the simulator's small referral effect does not guarantee an observed advantage |
| Is obvious leakage present? | Unique IDs; exact schema; no numeric target copies; ID and target excluded | Basic checks pass, but temporal validity rests on the synthetic data contract |

Numeric distributions and category counts are recorded without decorative plots.
Numeric outcome groups use quartile bins (ties may reduce the number of bins);
missing rows are counted separately in the missingness section. These results
support a moderately difficult demonstration dataset, not market conclusions.

Test-set ranking achieves lift 2.7442 at K=100, yet 156 of 215 converters remain
outside the selected queue. This cost of limited capacity must remain visible in
future product explanations. Calibration, uncertainty estimates, operational
eligibility, and causal benefit have not been established.
