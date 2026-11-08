import numpy as np
import pytest

from desk import engine as E
from desk.synthetic import TRUTH, generate


@pytest.fixture(scope="module")
def eng():
    return E.Engine(generate(n_customers=30_000, seed=7))


def test_engine_finds_the_planted_affinities(eng):
    top4 = {(r["from_brand"], r["to_brand"]) for r in eng.brand_pair_stats(top=4)}
    assert top4 == set(TRUTH.true_pairs())


def test_planted_non_opportunity_ranks_low(eng):
    ranked = [(r["from_brand"], r["to_brand"]) for r in eng.brand_pair_stats(top=30)]
    assert ranked.index(("Electronics", "Kids")) > 15


def test_policy_blocks_discount_above_ceiling(eng):
    r = eng.check_policy("Fashion", "Beauty", "email", 0.20)
    assert not r["passed"] and "ceiling" in r["violations"][0]


def test_policy_excludes_no_consent_recent_contact_and_global_holdout(eng):
    r = eng.check_policy("Fashion", "Beauty", "whatsapp", 0.0)
    assert r["eligible_audience"] < r["audience_before_rules"]
    ids = eng.eligible_ids("Fashion", "Beauty", "whatsapp")
    c = eng.customers.loc[ids]
    assert c.consent_whatsapp.all()
    assert (c.days_since_last_contact >= E.POLICY["min_days_since_contact"]).all()


def test_global_holdout_customers_are_never_treated(eng):
    res = E.simulate_send(eng, E.Action("Fashion", "Beauty", "email", 0.10, holdout_share=0.4))
    assert not res[res.global_holdout].treated.any()
    assert abs(eng.global_holdout.mean() - E.POLICY["global_holdout"]) < 0.01


def test_measurement_recovers_truth_on_average(eng):
    """Across many simulated sends, the CI should contain the true effect ~95% of the time."""
    a = E.Action("Fashion", "Beauty", "email", 0.10, holdout_share=0.4)
    truth = E.true_incremental_margin_per_customer(eng, a)
    cover = 0
    for seed in range(60):
        lo, hi = E.measure_lift(E.simulate_send(eng, a, seed))["ci95_per_customer_aed"]
        cover += lo <= truth <= hi
    assert cover / 60 >= 0.85


def test_holdout_planner_bounds():
    small = E.plan_holdout(300, 0.01, 0.02, 200, 0.5, 0.0)
    assert small["holdout_share"] == E.POLICY["holdout_max"] and not small["measurable"]
    big = E.plan_holdout(200_000, 0.02, 0.02, 200, 0.5, 0.0)
    assert big["measurable"] and big["holdout_share"] == E.POLICY["holdout_min"]


def test_estimate_never_reads_truth(eng, monkeypatch):
    before = eng.estimate_margin("Fashion", "Beauty", "email", 0.1)
    monkeypatch.setattr(TRUTH, "uplift_content_per_aff", 99.0)
    assert eng.estimate_margin("Fashion", "Beauty", "email", 0.1) == before
