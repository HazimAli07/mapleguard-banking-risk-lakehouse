"""Run the complete local MapleGuard pipeline."""

from pathlib import Path

from mapleguard.generate import GenerationConfig, generate_transactions
from mapleguard.model import train_evaluate_score
from mapleguard.pipeline import build_gold_tables, create_dashboard_preview, write_outputs


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    transactions = generate_transactions(GenerationConfig())
    result = train_evaluate_score(transactions)
    gold = build_gold_tables(result)
    write_outputs(transactions, result, gold, ROOT / "data" / "generated")
    create_dashboard_preview(gold, ROOT / "docs" / "mapleguard_dashboard_preview.png")
    print(f"Rows: {len(transactions):,}")
    print(f"Synthetic positive rate: {transactions['is_fraud'].mean():.3%}")
    print(f"Decision threshold: {result.threshold:.2f}")
    print(f"Holdout ROC AUC: {result.metrics['roc_auc']:.3f}")
    print(f"Holdout precision: {result.metrics['precision']:.3f}")
    print(f"Holdout recall: {result.metrics['recall']:.3f}")


if __name__ == "__main__":
    main()

