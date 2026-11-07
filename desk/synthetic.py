"""Synthetic data for a FICTIONAL GCC multi-brand retail group, "Al Waha Retail".

No real customers, brands or companies. Everything is generated from a fixed seed, and the
"truth" the system is supposed to discover is planted here and saved separately (TRUTH), so
evaluation can check whether the engine and the agents find it. The engine and the agents
never read TRUTH; only the simulator (for outcomes) and the evaluation do.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TODAY = pd.Timestamp("2026-11-01")
HISTORY_DAYS = 540

BRANDS = pd.DataFrame([
    # brand, category, avg first basket (AED), gross margin
    ("Mirage", "Fashion", 320, 0.55),
    ("Haven", "Home", 450, 0.45),
    ("Pace", "Sports", 280, 0.40),
    ("Glow", "Beauty", 180, 0.60),
    ("Little Oasis", "Kids", 220, 0.45),
    ("Volt", "Electronics", 900, 0.20),
], columns=["brand", "category", "avg_basket", "gross_margin"])
B = list(BRANDS.brand)
HOME_SHARE = [0.28, 0.16, 0.18, 0.17, 0.12, 0.09]


def _planted_affinity() -> np.ndarray:
    """Row = brand a customer already shops, column = brand they might cross to. 1.0 = neutral."""
    a = np.ones((6, 6))
    i = {b: k for k, b in enumerate(B)}
    a[i["Mirage"], i["Glow"]] = 6.0         # fashion -> beauty: the strongest planted pair
    a[i["Little Oasis"], i["Haven"]] = 5.0  # kids -> home
    a[i["Pace"], i["Volt"]] = 4.0           # sports -> electronics (wearables)
    a[i["Glow"], i["Mirage"]] = 3.0         # beauty -> fashion
    a[i["Volt"], i["Little Oasis"]] = 0.3   # electronics -> kids: a planted NON-opportunity
    np.fill_diagonal(a, 0.0)
    return a


@dataclass
class Truth:
    """What a real company would never know for sure, and what evaluation checks against."""
    affinity: np.ndarray = field(default_factory=_planted_affinity)
    base_rate_per_aff: float = 0.006       # 90-day natural cross-purchase rate per unit of affinity (sqrt-scaled)
    uplift_content_per_aff: float = 0.0025  # extra conversion from a relevant message, per unit of affinity
    uplift_per_discount: float = 0.12       # extra conversion per 1.0 of discount (10% off -> +1.2pp)
    channel_factor: dict = field(default_factory=lambda: {"email": 0.8, "sms": 1.0, "app_push": 1.1, "whatsapp": 1.3})

    def base_rate(self, a: str, b: str) -> float:
        return self.base_rate_per_aff * np.sqrt(self.affinity[B.index(a), B.index(b)])

    def uplift(self, a: str, b: str, channel: str, discount: float) -> float:
        aff = self.affinity[B.index(a), B.index(b)]
        return self.channel_factor[channel] * (self.uplift_content_per_aff * aff + self.uplift_per_discount * discount)

    def true_pairs(self, threshold: float = 2.0) -> list[tuple[str, str]]:
        idx = np.argwhere(self.affinity >= threshold)
        return [(B[r], B[c]) for r, c in idx]


TRUTH = Truth()


def generate(n_customers: int = 50_000, seed: int = 7) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = n_customers
    home = rng.choice(6, size=n, p=HOME_SHARE)
    single = rng.random(n) < 0.62                     # most loyalty members shop one brand only
    p_home = np.where(single, 1.0, rng.beta(5, 3, n))
    activity = rng.gamma(2.0, 0.6, n)                 # purchases per 30 days, skewed
    customers = pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "home_brand": [B[h] for h in home],
        "city": rng.choice(["Dubai", "Abu Dhabi", "Sharjah", "Al Ain"], n, p=[0.45, 0.30, 0.15, 0.10]),
        "language": rng.choice(["en", "ar"], n, p=[0.65, 0.35]),
        "consent_email": rng.random(n) < 0.70,
        "consent_sms": rng.random(n) < 0.55,
        "consent_app_push": rng.random(n) < 0.40,
        "consent_whatsapp": rng.random(n) < 0.35,
        "days_since_last_contact": rng.integers(0, 60, n),
        "contacts_last_30d": rng.poisson(1.6, n),
    })

    # transactions: each purchase is at the home brand, or crosses according to the planted affinity
    aff = TRUTH.affinity
    rows = []
    n_tx = rng.poisson(activity * HISTORY_DAYS / 30 / 3) + 1   # at least one purchase each
    for cid, h, ph, k in zip(customers.customer_id, home, p_home, n_tx):
        cross_p = aff[h] / aff[h].sum()
        brands = np.where(rng.random(k) < ph, h, rng.choice(6, size=k, p=cross_p))
        days_ago = np.sort(rng.integers(0, HISTORY_DAYS, k))[::-1]
        for b, d in zip(brands, days_ago):
            rows.append((cid, b, d))
    tx = pd.DataFrame(rows, columns=["customer_id", "brand_idx", "days_ago"])
    tx["brand"] = [B[i] for i in tx.brand_idx]
    mean_basket = BRANDS.set_index("brand").avg_basket
    tx["amount_aed"] = np.round(mean_basket.loc[tx.brand].values * rng.lognormal(0, 0.35, len(tx)), 2)
    tx["date"] = TODAY - pd.to_timedelta(tx.days_ago, unit="D")
    tx = tx.drop(columns="brand_idx")
    return {"brands": BRANDS.copy(), "customers": customers, "transactions": tx}
