"""Deterministic engine: every number the agents use comes from here, never from the model.

Tools exposed to the agents (see TOOLS at the bottom):
  brand_pair_stats      which brand pairs co-occur more than chance (affinity lift), and audience sizes
  pair_detail           the evidence for one pair: audience, natural cross-rate, basket, margin, languages
  estimate_margin       expected incremental gross margin for a proposed action (planning estimate)
  check_policy          consent, contact-frequency, discount and budget rules -> eligible audience
The simulator and lift measurement run after a human approves (not agent tools).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .synthetic import B, TRUTH

ACTIVE_DAYS = 90
POLICY = {
    "max_discount": 0.15,          # above this, the action is blocked
    "min_days_since_contact": 7,   # contacted more recently -> excluded
    "max_contacts_30d": 4,         # contacted more often -> excluded
    "campaign_budget_aed": 50_000,  # max expected discount cost per campaign
    "global_holdout": 0.10,        # owner call: 10% of ALL customers never get cross-brand campaigns (programme control)
    "holdout_min": 0.10,           # control group is sized by a power calculation, within these bounds (owner call)
    "holdout_max": 0.50,
    "target_power": 0.80,          # chance of detecting the expected lift if it is real
    "z_alpha": 1.96,               # 95% confidence
    "channels": ["email", "sms", "app_push", "whatsapp"],
}
# Planning assumption (would come from the group's past campaign tests; synthetic here):
# uplift ~ a little per point of affinity lift + a fixed amount per point of discount.
PLANNING = {"uplift_per_lift_point": 0.010, "uplift_per_discount": 0.10}  # deliberately ~30% conservative vs reality


def plan_holdout(n: int, uplift: float, base: float, basket: float, gm: float, discount: float) -> dict:
    """Smallest control-group share that detects the expected per-customer margin lift with the target power."""
    from statistics import NormalDist
    z_b = NormalDist().inv_cdf(POLICY["target_power"])
    m2 = np.exp(2 * 0.35 ** 2)                     # E[basket^2] / basket^2 for the lognormal basket spread
    p_t, p_h = base + uplift, base
    effect = basket * np.exp(0.35 ** 2 / 2) * (p_t * (gm - discount) - p_h * gm)
    var_t = p_t * (basket * (gm - discount)) ** 2 * m2 - (p_t * basket * (gm - discount)) ** 2
    var_h = p_h * (basket * gm) ** 2 * m2 - (p_h * basket * gm) ** 2
    for h in np.arange(POLICY["holdout_min"], POLICY["holdout_max"] + 1e-9, 0.05):
        se = np.sqrt(var_t / (n * (1 - h)) + var_h / (n * h)) if n > 0 else np.inf
        if effect > 0 and effect >= (POLICY["z_alpha"] + z_b) * se:
            return {"holdout_share": round(float(h), 2), "measurable": True,
                    "reason": f"smallest control group with {POLICY['target_power']:.0%} power to detect the expected lift"}
    return {"holdout_share": POLICY["holdout_max"], "measurable": False,
            "reason": "audience too small to detect the expected lift even with a 50% control group: "
                      "combine with another week or widen the audience"}


class Engine:
    def __init__(self, data: dict[str, pd.DataFrame]):
        self.brands = data["brands"].set_index("brand")
        self.customers = data["customers"].set_index("customer_id")
        self.tx = data["transactions"]
        bought = self.tx.groupby(["customer_id", "brand"]).size().unstack(fill_value=0) > 0
        self.bought = bought.reindex(columns=B, fill_value=False)
        last = self.tx.groupby("customer_id").days_ago.min()
        self.active = last <= ACTIVE_DAYS
        # deterministic, effectively random assignment by customer id (stable across runs)
        ids = self.customers.index.to_numpy()
        self.global_holdout = pd.Series((ids * 2654435761 % 1000) < POLICY["global_holdout"] * 1000, index=self.customers.index)

    # ------------------------------------------------------------------ tools
    def brand_pair_stats(self, top: int = 10) -> list[dict]:
        """Directional affinity lift for A -> B, measured on customers who already shop 2+ brands:
        share of A-home customers who also bought B, divided by the share of all other customers who bought B.
        (Home brand = where a customer shops most; it is what makes the lift directional.)"""
        multi_ids = self.bought.index[self.bought.sum(axis=1) >= 2]
        multi = self.bought.loc[multi_ids]
        home = self.customers.home_brand.reindex(multi_ids)
        out = []
        for a in B:
            in_a = multi[home == a]
            for b in B:
                if a == b or len(in_a) == 0:
                    continue
                others = multi[(home != a) & (home != b)]
                p_b = others[b].mean()
                lift = float(in_a[b].mean() / p_b) if p_b > 0 else 0.0
                aud = self._audience(a, b)
                out.append({"from_brand": a, "to_brand": b, "affinity_lift": round(lift, 2),
                            "multi_brand_evidence_n": int(len(in_a)),
                            "audience_size": int(len(aud))})
        out.sort(key=lambda r: -r["affinity_lift"])
        return out[:top]

    def pair_detail(self, from_brand: str, to_brand: str) -> dict:
        aud = self._audience(from_brand, to_brand)
        c = self.customers.loc[aud]
        tb = self.brands.loc[to_brand]
        # natural cross-rate: share of from_brand-home customers whose FIRST to_brand purchase was in the last 90 days
        home = self.customers.index[self.customers.home_brand == from_brand]
        first_b = self.tx[(self.tx.customer_id.isin(home)) & (self.tx.brand == to_brand)].groupby("customer_id").days_ago.max()
        natural = float((first_b <= ACTIVE_DAYS).sum() / max(len(home), 1))
        stats = {r["to_brand"]: r for r in self.brand_pair_stats(top=60) if r["from_brand"] == from_brand}
        return {
            "from_brand": from_brand, "to_brand": to_brand,
            "affinity_lift": stats[to_brand]["affinity_lift"],
            "audience_size": int(len(aud)),
            "audience_definition": f"{from_brand} customers active in the last {ACTIVE_DAYS} days who have never bought {to_brand}",
            "natural_90d_cross_rate": round(natural, 4),
            "to_brand_avg_basket_aed": float(tb.avg_basket),
            "to_brand_gross_margin": float(tb.gross_margin),
            "language_mix": c.language.value_counts(normalize=True).round(2).to_dict(),
            "city_mix": c.city.value_counts(normalize=True).round(2).to_dict(),
            "consent_rates": {ch: round(float(c[f"consent_{ch}"].mean()), 2) for ch in POLICY["channels"]},
        }

    def estimate_margin(self, from_brand: str, to_brand: str, channel: str, discount: float,
                        audience_size: int | None = None) -> dict:
        """Planning estimate BEFORE sending. Uses planning assumptions, never the true uplift."""
        d = self.pair_detail(from_brand, to_brand)
        n = audience_size if audience_size is not None else d["audience_size"]
        uplift = PLANNING["uplift_per_lift_point"] * max(d["affinity_lift"] - 1, 0) + PLANNING["uplift_per_discount"] * discount
        basket, gm, base = d["to_brand_avg_basket_aed"], d["to_brand_gross_margin"], d["natural_90d_cross_rate"]
        hold = plan_holdout(n, uplift, base, basket, gm, discount)
        n_treated = int(n * (1 - hold["holdout_share"]))
        incr_margin = n_treated * uplift * basket * (gm - discount)
        # customers who would have bought anyway still redeem the discount: margin given away
        cannibal = n_treated * base * basket * discount
        disc_cost = n_treated * (base + uplift) * basket * discount
        return {"from_brand": from_brand, "to_brand": to_brand, "channel": channel, "discount": discount,
                "treated": n_treated, "assumed_uplift_pp": round(uplift * 100, 2),
                "expected_incremental_conversions": round(n_treated * uplift, 1),
                "expected_incremental_gross_margin_aed": round(incr_margin - cannibal, 0),
                "expected_discount_cost_aed": round(disc_cost, 0),
                "control_group": hold,
                "margin_withheld_by_control_aed": round(n * hold["holdout_share"] * uplift * basket * (gm - discount), 0),
                "note": "planning estimate from assumptions; the real number comes from the holdout after sending"}

    def check_policy(self, from_brand: str, to_brand: str, channel: str, discount: float) -> dict:
        aud = self._audience(from_brand, to_brand)
        c = self.customers.loc[aud]
        violations = []
        if channel not in POLICY["channels"]:
            violations.append(f"unknown channel {channel}")
        if discount > POLICY["max_discount"]:
            violations.append(f"discount {discount:.0%} above the {POLICY['max_discount']:.0%} ceiling")
        excluded = {
            "no_consent": int((~c[f"consent_{channel}"]).sum()) if channel in POLICY["channels"] else len(c),
            "contacted_recently": int((c.days_since_last_contact < POLICY["min_days_since_contact"]).sum()),
            "contact_cap": int((c.contacts_last_30d > POLICY["max_contacts_30d"]).sum()),
            "global_holdout_not_contacted": int(self.global_holdout.reindex(c.index).sum()),
        }
        ok = c[f"consent_{channel}"] if channel in POLICY["channels"] else pd.Series(False, index=c.index)
        ok &= c.days_since_last_contact >= POLICY["min_days_since_contact"]
        ok &= c.contacts_last_30d <= POLICY["max_contacts_30d"]
        eligible = int(ok.sum())
        est = self.estimate_margin(from_brand, to_brand, channel, discount, eligible) if not violations else None
        if est and est["expected_discount_cost_aed"] > POLICY["campaign_budget_aed"]:
            violations.append(f"expected discount cost AED {est['expected_discount_cost_aed']:,.0f} above the "
                              f"AED {POLICY['campaign_budget_aed']:,} budget")
        return {"passed": not violations, "violations": violations, "audience_before_rules": int(len(c)),
                "eligible_audience": eligible, "excluded_by_rule": excluded, "estimate_for_eligible": est}

    def compare_options(self, from_brand: str, to_brand: str, options: list[dict]) -> dict:
        """Evaluate several (channel, discount) options side by side, including each option's gap to the best,
        so the agents never have to subtract or compare numbers themselves."""
        rows = []
        for o in options:
            ch, disc = o["channel"], float(o["discount"])
            pol = self.check_policy(from_brand, to_brand, ch, disc)
            est = pol["estimate_for_eligible"] or {}
            rows.append({"channel": ch, "discount": disc, "passes_policy": pol["passed"], "violations": pol["violations"],
                         "eligible_audience": pol["eligible_audience"],
                         "expected_incremental_gross_margin_aed": est.get("expected_incremental_gross_margin_aed"),
                         "expected_discount_cost_aed": est.get("expected_discount_cost_aed"),
                         "control_group_share": (est.get("control_group") or {}).get("holdout_share"),
                         "measurable": (est.get("control_group") or {}).get("measurable")})
        ok = [r for r in rows if r["passes_policy"]]
        best = max(ok, key=lambda r: r["expected_incremental_gross_margin_aed"]) if ok else None
        best_nd = max((r for r in ok if r["discount"] == 0), key=lambda r: r["expected_incremental_gross_margin_aed"], default=None)
        for r in rows:
            r["gap_vs_best_aed"] = (round(best["expected_incremental_gross_margin_aed"] - r["expected_incremental_gross_margin_aed"], 0)
                                    if best and r["passes_policy"] else None)
            r["share_of_best"] = (round(r["expected_incremental_gross_margin_aed"] / best["expected_incremental_gross_margin_aed"], 2)
                                  if best and r["passes_policy"] and best["expected_incremental_gross_margin_aed"] else None)
        rows.sort(key=lambda r: -(r["expected_incremental_gross_margin_aed"] or -1e9))
        return {"options": rows, "best": best and {"channel": best["channel"], "discount": best["discount"]},
                "best_no_discount": best_nd and {"channel": best_nd["channel"], "discount": 0.0},
                "discount_premium_over_best_no_discount_aed": (round(best["expected_incremental_gross_margin_aed"]
                    - best_nd["expected_incremental_gross_margin_aed"], 0) if best and best_nd else None)}

    # ------------------------------------------------------------------ after approval (not agent tools)
    def eligible_ids(self, from_brand: str, to_brand: str, channel: str) -> np.ndarray:
        c = self.customers.loc[self._audience(from_brand, to_brand)]
        ok = c[f"consent_{channel}"] & (c.days_since_last_contact >= POLICY["min_days_since_contact"]) \
            & (c.contacts_last_30d <= POLICY["max_contacts_30d"])
        return c.index[ok].to_numpy()

    def _audience(self, a: str, b: str) -> pd.Index:
        cust = self.customers
        mask = (cust.home_brand == a) & ~self.bought[b].reindex(cust.index, fill_value=False) & self.active.reindex(cust.index, fill_value=False)
        return cust.index[mask]


@dataclass
class Action:
    from_brand: str
    to_brand: str
    channel: str
    discount: float
    holdout_share: float = 0.20
    message_en: str = ""
    message_ar: str = ""


def simulate_send(engine: Engine, action: Action, seed: int = 11) -> pd.DataFrame:
    """Simulated 90-day outcome. Uses the planted TRUTH; this is the 'real world', not the engine."""
    rng = np.random.default_rng(seed)
    ids = engine.eligible_ids(action.from_brand, action.to_brand, action.channel)
    in_global = engine.global_holdout.reindex(ids).to_numpy()
    # campaign control = the global-holdout customers in this audience, topped up to the planned share if needed
    g = POLICY["global_holdout"]
    extra = max(action.holdout_share - g, 0) / (1 - g)
    treated = ~in_global & (rng.random(len(ids)) >= extra)
    base = TRUTH.base_rate(action.from_brand, action.to_brand)
    p = base + treated * TRUTH.uplift(action.from_brand, action.to_brand, action.channel, action.discount)
    converted = rng.random(len(ids)) < p
    tb = engine.brands.loc[action.to_brand]
    basket = np.where(converted, tb.avg_basket * rng.lognormal(0, 0.35, len(ids)), 0.0)
    discount = np.where(treated, action.discount, 0.0)
    margin = basket * (tb.gross_margin - discount)
    return pd.DataFrame({"customer_id": ids, "treated": treated, "global_holdout": in_global, "converted": converted,
                         "basket_aed": basket.round(2), "gross_margin_aed": margin.round(2)})


def measure_lift(results: pd.DataFrame) -> dict:
    """Treated vs holdout: incremental gross margin per customer with a 95% confidence interval."""
    t, h = results[results.treated], results[~results.treated]
    diff = t.gross_margin_aed.mean() - h.gross_margin_aed.mean()
    se = np.sqrt(t.gross_margin_aed.var() / len(t) + h.gross_margin_aed.var() / len(h))
    n_t = len(t)
    return {
        "treated": int(n_t), "holdout": int(len(h)),
        "conversion_treated": round(float(t.converted.mean()), 4),
        "conversion_holdout": round(float(h.converted.mean()), 4),
        "incremental_margin_per_customer_aed": round(float(diff), 2),
        "ci95_per_customer_aed": [round(float(diff - 1.96 * se), 2), round(float(diff + 1.96 * se), 2)],
        "incremental_gross_margin_aed": round(float(diff * n_t), 0),
        "ci95_total_aed": [round(float((diff - 1.96 * se) * n_t), 0), round(float((diff + 1.96 * se) * n_t), 0)],
        "significant": bool(diff - 1.96 * se > 0),
    }


def measure_programme(all_results: list[pd.DataFrame]) -> dict:
    """Programme level: everyone contacted across all campaigns vs everyone held back (global + campaign controls)."""
    r = pd.concat(all_results, ignore_index=True)
    out = measure_lift(r)
    out["campaigns"] = len(all_results)
    return out


def true_incremental_margin_per_customer(engine: Engine, action: Action) -> float:
    """Ground truth for evaluation only: expected treated-minus-holdout margin per customer."""
    tb = engine.brands.loc[action.to_brand]
    base = TRUTH.base_rate(action.from_brand, action.to_brand)
    up = TRUTH.uplift(action.from_brand, action.to_brand, action.channel, action.discount)
    return float(tb.avg_basket * np.exp(0.35 ** 2 / 2) * ((base + up) * (tb.gross_margin - action.discount) - base * tb.gross_margin))


def action_dict(a: Action) -> dict:
    return asdict(a)
