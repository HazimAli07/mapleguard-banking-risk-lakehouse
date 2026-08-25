"""Deterministic, privacy-safe transaction simulator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GenerationConfig:
    rows: int = 60_000
    seed: int = 2027
    start_date: str = "2026-01-01"
    days: int = 238


PROVINCES = np.array(["ON", "QC", "BC", "AB", "MB", "SK", "NS", "NB"])
PROVINCE_P = np.array([0.39, 0.22, 0.14, 0.12, 0.04, 0.03, 0.04, 0.02])
CHANNELS = np.array(["Point of sale", "E-commerce", "Mobile wallet", "ATM", "Recurring"])
CHANNEL_P = np.array([0.41, 0.25, 0.14, 0.11, 0.09])
MERCHANTS = np.array(
    ["Grocery", "Fuel", "Dining", "Travel", "Electronics", "Digital services", "Cash", "Other"]
)
MERCHANT_P = np.array([0.22, 0.11, 0.17, 0.08, 0.09, 0.13, 0.08, 0.12])
SEGMENTS = np.array(["Everyday", "Student", "New-to-bank", "Affluent", "Small business"])
SEGMENT_P = np.array([0.43, 0.18, 0.12, 0.15, 0.12])


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def generate_transactions(config: GenerationConfig | None = None) -> pd.DataFrame:
    """Generate synthetic transactions with no real identifiers or customer data."""
    cfg = config or GenerationConfig()
    rng = np.random.default_rng(cfg.seed)
    rows = cfg.rows

    day_offset = rng.integers(0, cfg.days, rows)
    hour = rng.integers(0, 24, rows)
    minute = rng.integers(0, 60, rows)
    event_ts = (
        pd.Timestamp(cfg.start_date)
        + pd.to_timedelta(day_offset, unit="D")
        + pd.to_timedelta(hour, unit="h")
        + pd.to_timedelta(minute, unit="m")
    )

    province = rng.choice(PROVINCES, size=rows, p=PROVINCE_P)
    channel = rng.choice(CHANNELS, size=rows, p=CHANNEL_P)
    merchant = rng.choice(MERCHANTS, size=rows, p=MERCHANT_P)
    segment = rng.choice(SEGMENTS, size=rows, p=SEGMENT_P)

    base_amount = rng.lognormal(mean=4.15, sigma=1.02, size=rows)
    multiplier = np.select(
        [merchant == "Travel", merchant == "Electronics", merchant == "Cash", merchant == "Grocery"],
        [2.1, 1.7, 1.35, 0.82],
        default=1.0,
    )
    amount = np.clip(base_amount * multiplier, 1.25, 4_900.0).round(2)

    is_international = rng.binomial(1, np.where(merchant == "Travel", 0.24, 0.055), rows)
    card_present = np.where(channel == "Point of sale", 1, 0)
    device_trust = np.clip(rng.normal(72, 20, rows), 0, 100).round(1)
    account_age = np.clip(rng.gamma(shape=2.2, scale=470, size=rows), 2, 5_000).astype(int)
    velocity = np.clip(rng.poisson(lam=2.6, size=rows) + 1, 1, 22)
    distance = np.clip(rng.exponential(scale=31, size=rows), 0.1, 1_500).round(1)

    linear_risk = (
        -9.20
        + 0.62 * np.log1p(amount)
        + 1.22 * (channel == "E-commerce")
        + 0.82 * (channel == "Mobile wallet")
        + 2.05 * is_international
        + 0.070 * np.maximum(50 - device_trust, 0)
        + 0.48 * np.maximum(velocity - 5, 0)
        + 0.0110 * np.maximum(distance - 45, 0)
        + 1.10 * ((hour <= 4) | (hour >= 23))
        + 1.25 * (account_age < 90)
        + 0.72 * (merchant == "Electronics")
        + 0.48 * (merchant == "Digital services")
    )
    probability = np.clip(_sigmoid(linear_risk), 0.001, 0.94)
    is_fraud = rng.binomial(1, probability, rows).astype(int)

    frame = pd.DataFrame(
        {
            "transaction_id": [f"TX{i:07d}" for i in range(1, rows + 1)],
            "customer_id": [f"C{i:06d}" for i in rng.integers(1, max(5_000, rows // 4), rows)],
            "event_ts": event_ts,
            "province": province,
            "channel": channel,
            "merchant_category": merchant,
            "customer_segment": segment,
            "amount_cad": amount,
            "is_international": is_international,
            "is_card_present": card_present,
            "device_trust_score": device_trust,
            "account_age_days": account_age,
            "transactions_24h": velocity,
            "distance_from_home_km": distance,
            "hour_of_day": hour,
            "is_fraud": is_fraud,
        }
    )
    frame["event_date"] = frame["event_ts"].dt.date
    frame["week_start"] = (frame["event_ts"] - pd.to_timedelta(frame["event_ts"].dt.weekday, unit="D")).dt.date
    return frame.sort_values(["event_ts", "transaction_id"]).reset_index(drop=True)
