"""Time-aware model training, threshold selection, and scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORICAL_FEATURES = ["province", "channel", "merchant_category", "customer_segment"]
NUMERIC_FEATURES = [
    "amount_cad",
    "is_international",
    "is_card_present",
    "device_trust_score",
    "account_age_days",
    "transactions_24h",
    "distance_from_home_km",
    "hour_of_day",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


@dataclass
class ModelResult:
    scored: pd.DataFrame
    metrics: dict[str, float]
    threshold: float
    train_end: pd.Timestamp
    validation_end: pd.Timestamp
    pipeline: Pipeline


def _make_pipeline() -> Pipeline:
    transformer = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("features", transformer),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=600,
                    solver="liblinear",
                    random_state=2027,
                ),
            ),
        ]
    )


def _select_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    candidates = np.arange(0.10, 0.81, 0.02)
    scored = []
    for value in candidates:
        prediction = probabilities >= value
        recall = recall_score(y_true, prediction, zero_division=0)
        f1 = f1_score(y_true, prediction, zero_division=0)
        scored.append((value, recall, f1))
    eligible = [row for row in scored if row[1] >= 0.55]
    pool = eligible if eligible else scored
    return float(max(pool, key=lambda row: row[2])[0])


def _reason_code(row: pd.Series) -> str:
    reasons: list[str] = []
    if row["is_international"] == 1:
        reasons.append("international")
    if row["device_trust_score"] < 35:
        reasons.append("low device trust")
    if row["transactions_24h"] >= 8:
        reasons.append("high velocity")
    if row["amount_cad"] >= 500:
        reasons.append("high amount")
    if row["hour_of_day"] <= 4 or row["hour_of_day"] >= 23:
        reasons.append("unusual hour")
    if row["distance_from_home_km"] >= 120:
        reasons.append("distance anomaly")
    if row["account_age_days"] < 90:
        reasons.append("new account")
    return ", ".join(reasons[:3]) if reasons else "combined behavioural signal"


def train_evaluate_score(transactions: pd.DataFrame) -> ModelResult:
    """Train on early events, tune on later validation events, evaluate on a final holdout."""
    ordered = transactions.sort_values("event_ts").reset_index(drop=True).copy()
    train_end = ordered["event_ts"].quantile(0.56)
    validation_end = ordered["event_ts"].quantile(0.70)

    train = ordered[ordered["event_ts"] <= train_end]
    validation = ordered[(ordered["event_ts"] > train_end) & (ordered["event_ts"] <= validation_end)]
    test = ordered[ordered["event_ts"] > validation_end]

    tuning_pipeline = _make_pipeline()
    tuning_pipeline.fit(train[FEATURES], train["is_fraud"])
    validation_probability = tuning_pipeline.predict_proba(validation[FEATURES])[:, 1]
    threshold = _select_threshold(validation["is_fraud"], validation_probability)

    development = ordered[ordered["event_ts"] <= validation_end]
    final_pipeline = _make_pipeline()
    final_pipeline.fit(development[FEATURES], development["is_fraud"])

    test_probability = final_pipeline.predict_proba(test[FEATURES])[:, 1]
    test_prediction = (test_probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(test["is_fraud"], test_prediction, labels=[0, 1]).ravel()

    metrics = {
        "roc_auc": float(roc_auc_score(test["is_fraud"], test_probability)),
        "average_precision": float(average_precision_score(test["is_fraud"], test_probability)),
        "precision": float(precision_score(test["is_fraud"], test_prediction, zero_division=0)),
        "recall": float(recall_score(test["is_fraud"], test_prediction, zero_division=0)),
        "f1": float(f1_score(test["is_fraud"], test_prediction, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)),
        "test_rows": float(len(test)),
        "test_positive_rate": float(test["is_fraud"].mean()),
        "true_positives": float(tp),
        "false_positives": float(fp),
        "false_negatives": float(fn),
        "true_negatives": float(tn),
    }

    scored = ordered.copy()
    scored["risk_score"] = final_pipeline.predict_proba(scored[FEATURES])[:, 1]
    scored["is_alert"] = (scored["risk_score"] >= threshold).astype(int)
    scored["risk_tier"] = pd.cut(
        scored["risk_score"], bins=[-0.001, 0.25, 0.50, 0.75, 1.001], labels=["Low", "Guarded", "High", "Critical"]
    ).astype(str)
    scored["reason_code"] = scored.apply(_reason_code, axis=1)

    return ModelResult(
        scored=scored,
        metrics=metrics,
        threshold=threshold,
        train_end=pd.Timestamp(train_end),
        validation_end=pd.Timestamp(validation_end),
        pipeline=final_pipeline,
    )
