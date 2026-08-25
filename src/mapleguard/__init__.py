"""MapleGuard synthetic banking-risk portfolio package."""

from .generate import GenerationConfig, generate_transactions
from .model import ModelResult, train_evaluate_score
from .pipeline import build_gold_tables

__all__ = [
    "GenerationConfig",
    "ModelResult",
    "generate_transactions",
    "train_evaluate_score",
    "build_gold_tables",
]

