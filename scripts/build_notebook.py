"""Build the reader-facing Jupyter notebook with nbformat."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "MapleGuard_Portfolio_Walkthrough.ipynb"


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# MapleGuard — Banking Transaction Risk Lakehouse\n\n"
            "**Author:** Hazim Ali  \n"
            "**Goal:** demonstrate an evidence-led, privacy-safe banking data/AI workflow for Winter 2027 co-op recruiting.\n\n"
            "> All records are deterministic and synthetic. Results are not suitable for real financial decisions."
        ),
        nbf.v4.new_markdown_cell(
            "## Decision context\n\n"
            "Risk teams need a reliable path from raw transactions to monitored indicators and a prioritized human-review queue. "
            "This notebook reproduces the local pipeline; the Databricks source notebook builds the scalable Delta version."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import sys\n"
            "ROOT = Path.cwd().resolve().parent if Path.cwd().name == 'notebooks' else Path.cwd().resolve()\n"
            "sys.path.insert(0, str(ROOT / 'src'))\n"
            "from mapleguard import GenerationConfig, generate_transactions, train_evaluate_score, build_gold_tables"
        ),
        nbf.v4.new_code_cell(
            "transactions = generate_transactions(GenerationConfig())\n"
            "transactions.head()"
        ),
        nbf.v4.new_markdown_cell(
            "## Data contract\n\n"
            "The generator creates anonymous IDs and behavioural features only. It includes explicit event time so the model can be evaluated on future-like data."
        ),
        nbf.v4.new_code_cell(
            "transactions[['amount_cad','device_trust_score','account_age_days','transactions_24h','is_fraud']].describe().round(2)"
        ),
        nbf.v4.new_markdown_cell(
            "## Time-aware model evaluation\n\n"
            "Training, validation, and testing follow event time. The threshold is selected on validation data and reported metrics come from the untouched holdout period."
        ),
        nbf.v4.new_code_cell(
            "result = train_evaluate_score(transactions)\n"
            "result.metrics"
        ),
        nbf.v4.new_code_cell(
            "gold = build_gold_tables(result)\n"
            "gold['gold_overview'].T"
        ),
        nbf.v4.new_markdown_cell(
            "## Human-review queue\n\n"
            "Alerts are ordered by model score and include non-causal behavioural reason codes. A production workflow would require case-management controls and investigator outcomes."
        ),
        nbf.v4.new_code_cell(
            "gold['gold_alert_queue'].head(15)"
        ),
        nbf.v4.new_markdown_cell(
            "## What this proves—and what it does not\n\n"
            "The project proves reproducibility, time-aware validation, governed metric definitions, and decision-oriented communication. "
            "Synthetic performance does not prove real-world accuracy, fairness, privacy compliance, or production readiness."
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()

