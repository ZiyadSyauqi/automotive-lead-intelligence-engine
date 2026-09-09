"""Fictional lead snapshots; relationships are illustrative, not market evidence."""

import numpy as np
import pandas as pd

NUMERIC = ["purchase_timeline_days", "previous_interactions", "follow_up_count",
           "response_latency_hours"]
CATEGORICAL = ["contactability", "financing_interest", "test_drive_activity",
               "vehicle_price_segment", "lead_source", "trade_in_interest",
               "appointment_activity"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "converted_30d"


def generate_leads(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate independent day-seven snapshots and subsequent 30-day outcomes.

    Unobserved intent affects several observations and the outcome. Bernoulli
    sampling, hidden noise, weak effects, and missing observations prevent a
    deterministic rule from recovering labels. No hidden variables are exported.
    """
    if n_rows < 1:
        raise ValueError("n_rows must be positive")
    rng = np.random.default_rng(seed)
    intent = rng.normal(size=n_rows)
    timeline = np.clip(np.rint(rng.lognormal(4.0 - .25 * intent, .7)), 1, 365)
    interactions = np.clip(rng.poisson(np.exp(.8 + .25 * intent)), 0, 20)
    followups = np.minimum(rng.poisson(2, n_rows), 12)
    contact = rng.choice(["reachable", "intermittent", "unreachable"], n_rows,
                         p=[.6, .28, .12])
    financing = rng.choice(["yes", "no", "undecided"], n_rows, p=[.45, .3, .25])
    appointment = np.where(rng.random(n_rows) < 1 / (1 + np.exp(1 - .7 * intent)),
                           "attended", "none")
    appointment = np.where((appointment == "none") & (rng.random(n_rows) < .25),
                           "scheduled", appointment)
    appointment = np.where((appointment == "scheduled") & (rng.random(n_rows) < .25),
                           "missed", appointment)
    drive = np.where(rng.random(n_rows) < 1 / (1 + np.exp(1.4 - .6 * intent)),
                     "completed", "none")
    drive = np.where((drive == "none") & (rng.random(n_rows) < .18), "requested", drive)
    segment = rng.choice(["entry", "mid", "premium"], n_rows, p=[.4, .45, .15])
    source = rng.choice(["website", "marketplace", "walk_in", "referral", "event"],
                        n_rows, p=[.3, .3, .15, .15, .1])
    trade = rng.choice(["yes", "no", "undecided"], n_rows, p=[.3, .5, .2])
    latency = np.clip(rng.lognormal(2.4 - .3 * intent, .9), .1, 168)
    # Nonlinear saturation and interaction deliberately misspecify a linear baseline.
    logit = (-2.9 + .95 * (timeline <= 30) + .4 * (timeline <= 60)
             + .55 * (contact == "reachable") - .65 * (contact == "unreachable")
             + .85 * (drive == "completed") + .6 * (appointment == "attended")
             - .45 * (appointment == "missed") + .12 * np.minimum(interactions, 5)
             - .16 * np.log1p(latency) - .08 * np.maximum(followups - 4, 0)
             + .18 * (financing == "yes") + .12 * (trade == "yes")
             + .15 * (source == "referral") - .1 * (segment == "premium")
             + .35 * ((drive == "completed") & (timeline <= 30))
             + .5 * intent + rng.normal(0, .85, n_rows))
    probability = .025 + .95 / (1 + np.exp(-logit))
    df = pd.DataFrame({
        "lead_id": np.arange(1, n_rows + 1), "purchase_timeline_days": timeline,
        "contactability": contact, "financing_interest": financing,
        "test_drive_activity": drive, "vehicle_price_segment": segment,
        "previous_interactions": interactions, "follow_up_count": followups,
        "lead_source": source, "trade_in_interest": trade,
        "response_latency_hours": latency, "appointment_activity": appointment,
        TARGET: rng.binomial(1, probability),
    })
    # Nonresponders have no measured response time; other fields have recording gaps.
    df.loc[df.contactability == "unreachable", "response_latency_hours"] = np.nan
    for column in ["purchase_timeline_days", "response_latency_hours", "financing_interest"]:
        df.loc[rng.random(n_rows) < .05, column] = np.nan
    return df[["lead_id"] + FEATURES + [TARGET]]
