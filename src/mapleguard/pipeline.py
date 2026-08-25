"""Gold marts and local visual QA for MapleGuard."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .model import ModelResult


def _risk_mart(scored: pd.DataFrame, dimension: str) -> pd.DataFrame:
    return (
        scored.groupby(dimension, observed=True)
        .agg(
            transactions=("transaction_id", "count"),
            value_cad=("amount_cad", "sum"),
            alerts=("is_alert", "sum"),
            observed_fraud=("is_fraud", "sum"),
            average_risk_score=("risk_score", "mean"),
        )
        .assign(
            alert_rate=lambda x: x["alerts"] / x["transactions"],
            observed_fraud_rate=lambda x: x["observed_fraud"] / x["transactions"],
        )
        .reset_index()
        .sort_values("alert_rate", ascending=False)
    )


def build_gold_tables(result: ModelResult) -> dict[str, pd.DataFrame]:
    scored = result.scored.copy()
    scored["event_date"] = pd.to_datetime(scored["event_date"])
    scored["week_start"] = pd.to_datetime(scored["week_start"])

    daily = (
        scored.groupby("event_date")
        .agg(
            transactions=("transaction_id", "count"),
            value_cad=("amount_cad", "sum"),
            alerts=("is_alert", "sum"),
            observed_fraud=("is_fraud", "sum"),
            average_risk_score=("risk_score", "mean"),
        )
        .assign(
            alert_rate=lambda x: x["alerts"] / x["transactions"],
            observed_fraud_rate=lambda x: x["observed_fraud"] / x["transactions"],
        )
        .reset_index()
    )

    weekly = (
        scored.groupby("week_start")
        .agg(
            transactions=("transaction_id", "count"),
            value_cad=("amount_cad", "sum"),
            alerts=("is_alert", "sum"),
            observed_fraud=("is_fraud", "sum"),
            average_risk_score=("risk_score", "mean"),
        )
        .assign(
            alert_rate=lambda x: x["alerts"] / x["transactions"],
            observed_fraud_rate=lambda x: x["observed_fraud"] / x["transactions"],
        )
        .reset_index()
    )

    alerts = scored[scored["is_alert"] == 1].copy()
    alerts = alerts.sort_values(["risk_score", "amount_cad"], ascending=False).head(500)
    alerts = alerts[
        [
            "transaction_id",
            "event_ts",
            "province",
            "channel",
            "merchant_category",
            "customer_segment",
            "amount_cad",
            "risk_score",
            "risk_tier",
            "reason_code",
            "is_fraud",
        ]
    ]

    metrics = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in result.metrics.items()]
        + [{"metric": "decision_threshold", "value": result.threshold}]
    )
    overview = pd.DataFrame(
        [
            {
                "transactions": len(scored),
                "value_cad": scored["amount_cad"].sum(),
                "alerts": int(scored["is_alert"].sum()),
                "alert_rate": scored["is_alert"].mean(),
                "observed_fraud_rate": scored["is_fraud"].mean(),
                "average_risk_score": scored["risk_score"].mean(),
                "model_roc_auc": result.metrics["roc_auc"],
                "model_recall": result.metrics["recall"],
            }
        ]
    )
    return {
        "gold_overview": overview,
        "gold_daily_kpis": daily,
        "gold_weekly_kpis": weekly,
        "gold_risk_by_channel": _risk_mart(scored, "channel"),
        "gold_risk_by_province": _risk_mart(scored, "province"),
        "gold_risk_by_merchant": _risk_mart(scored, "merchant_category"),
        "gold_model_metrics": metrics,
        "gold_alert_queue": alerts,
    }


def write_outputs(
    transactions: pd.DataFrame,
    result: ModelResult,
    gold: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    transactions.head(2_000).to_csv(output_dir / "synthetic_transactions_sample.csv", index=False)
    result.scored.head(2_000).to_csv(output_dir / "scored_transactions_sample.csv", index=False)
    for name, frame in gold.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    metadata = {
        "rows": len(transactions),
        "seed": 2027,
        "synthetic": True,
        "threshold": result.threshold,
        "train_end": result.train_end.isoformat(),
        "validation_end": result.validation_end.isoformat(),
        "metrics": result.metrics,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def create_dashboard_preview(gold: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Create a static visual QA preview; the live deliverable is the Databricks dashboard."""
    navy, blue, gold_colour, orange, slate = "#0B1F3A", "#1976D2", "#D6A84B", "#F28E2B", "#5F6B7A"
    overview = gold["gold_overview"].iloc[0]
    weekly = gold["gold_weekly_kpis"].copy()
    channel = gold["gold_risk_by_channel"].sort_values("alert_rate")
    province = gold["gold_risk_by_province"].sort_values("alert_rate", ascending=False)
    merchant = gold["gold_risk_by_merchant"].sort_values("alert_rate", ascending=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold"})
    fig = plt.figure(figsize=(16, 10), facecolor="#F4F7FB")
    grid = fig.add_gridspec(3, 12, height_ratios=[1.0, 2.2, 2.2], hspace=0.65, wspace=0.8, top=0.86)
    fig.suptitle("MAPLEGUARD  |  Transaction Risk Command Centre", x=0.055, y=0.975, ha="left", fontsize=22, weight="bold", color=navy)
    fig.text(0.055, 0.94, "Synthetic Canadian banking data • decision support for human review", fontsize=10.5, color=slate)

    card_specs = [
        ("Transactions", f"{int(overview['transactions']):,}"),
        ("Value monitored", f"${overview['value_cad']/1_000_000:.1f}M"),
        ("Alerts", f"{int(overview['alerts']):,}"),
        ("Holdout ROC AUC", f"{overview['model_roc_auc']:.3f}"),
    ]
    for index, (label, value) in enumerate(card_specs):
        ax = fig.add_subplot(grid[0, index * 3 : (index + 1) * 3])
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.06, 0.67, label.upper(), transform=ax.transAxes, color=slate, fontsize=9, weight="bold")
        ax.text(0.06, 0.22, value, transform=ax.transAxes, color=navy, fontsize=24, weight="bold")

    ax1 = fig.add_subplot(grid[1, :7])
    ax1.set_facecolor("white")
    ax1.plot(pd.to_datetime(weekly["week_start"]), weekly["alert_rate"] * 100, color=blue, linewidth=2.6, marker="o", markersize=4, label="Alert rate")
    ax1.plot(pd.to_datetime(weekly["week_start"]), weekly["observed_fraud_rate"] * 100, color=gold_colour, linewidth=2.2, marker="o", markersize=3, label="Observed synthetic fraud")
    ax1.set_title("Weekly risk movement", loc="left", color=navy, pad=12)
    ax1.set_ylabel("Rate (%)", color=slate)
    ax1.grid(axis="y", color="#E4EAF2", linewidth=0.8)
    ax1.legend(frameon=False, ncol=2, loc="upper left")
    ax1.spines[["top", "right"]].set_visible(False)

    ax2 = fig.add_subplot(grid[1, 7:])
    ax2.set_facecolor("white")
    ax2.barh(channel["channel"], channel["alert_rate"] * 100, color=[blue, "#4C9BE8", gold_colour, orange, navy])
    ax2.set_title("Where alerts concentrate", loc="left", color=navy, pad=12)
    ax2.set_xlabel("Alert rate (%)", color=slate)
    ax2.grid(axis="x", color="#E4EAF2", linewidth=0.8)
    ax2.spines[["top", "right", "left"]].set_visible(False)

    ax3 = fig.add_subplot(grid[2, :6])
    ax3.set_facecolor("white")
    top_province = province.head(8).sort_values("alert_rate")
    ax3.barh(top_province["province"], top_province["alert_rate"] * 100, color=navy)
    ax3.set_title("Province risk profile", loc="left", color=navy, pad=12)
    ax3.set_xlabel("Alert rate (%)", color=slate)
    ax3.grid(axis="x", color="#E4EAF2", linewidth=0.8)
    ax3.spines[["top", "right", "left"]].set_visible(False)

    ax4 = fig.add_subplot(grid[2, 6:])
    ax4.set_facecolor("white")
    top_merchant = merchant.head(8).sort_values("alert_rate")
    colours = np.where(top_merchant["alert_rate"] >= top_merchant["alert_rate"].median(), orange, blue)
    ax4.barh(top_merchant["merchant_category"], top_merchant["alert_rate"] * 100, color=colours)
    ax4.set_title("Merchant-category risk", loc="left", color=navy, pad=12)
    ax4.set_xlabel("Alert rate (%)", color=slate)
    ax4.grid(axis="x", color="#E4EAF2", linewidth=0.8)
    ax4.spines[["top", "right", "left"]].set_visible(False)

    fig.text(0.055, 0.015, "Portfolio demonstration by Hazim Ali • Synthetic data • Not for production decisions", fontsize=9, color=slate)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
