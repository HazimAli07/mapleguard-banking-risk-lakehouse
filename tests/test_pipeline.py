from mapleguard.generate import GenerationConfig, generate_transactions
from mapleguard.model import train_evaluate_score
from mapleguard.pipeline import build_gold_tables


def test_generator_contract_and_determinism() -> None:
    config = GenerationConfig(rows=5_000)
    first = generate_transactions(config)
    second = generate_transactions(config)
    assert len(first) == config.rows
    assert first.head(25).equals(second.head(25))
    assert first["transaction_id"].is_unique
    assert first.isna().sum().sum() == 0
    assert first["amount_cad"].gt(0).all()
    assert first["device_trust_score"].between(0, 100).all()
    assert set(first["is_fraud"].unique()).issubset({0, 1})


def test_model_and_gold_quality() -> None:
    transactions = generate_transactions(GenerationConfig(rows=18_000))
    result = train_evaluate_score(transactions)
    gold = build_gold_tables(result)
    assert 0.10 <= result.threshold <= 0.80
    assert result.metrics["roc_auc"] >= 0.74
    assert result.metrics["recall"] >= 0.45
    assert result.scored["risk_score"].between(0, 1).all()
    assert result.scored["is_alert"].sum() > 0
    assert len(gold["gold_weekly_kpis"]) >= 20
    assert len(gold["gold_risk_by_channel"]) >= 5
    assert not gold["gold_alert_queue"].empty
