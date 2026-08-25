# Databricks dashboard blueprint

## Audience and decision

**Audience:** risk analytics manager, fraud operations lead, data/AI recruiter.  
**Decision:** assess changing risk, verify model utility, and prioritize human review.

## Page 1 — Risk command centre

Use a 12-column canvas and a restrained navy / blue / gold / orange palette.

| Row | Widget | Query | Recommended visual |
|---|---|---|---|
| 1 | Transactions | `overview` | KPI counter |
| 1 | Value monitored | `overview` | KPI counter with CAD formatting |
| 1 | Alerts | `overview` | KPI counter |
| 1 | Holdout ROC AUC | `model_metrics` | KPI counter, 3 decimals |
| 2 | Weekly alert vs observed rate | `weekly_trend` | Two-series line chart |
| 2 | Alert rate by channel | `channel_risk` | Horizontal bar chart |
| 3 | Alert rate by province | `province_risk` | Sorted bar chart |
| 3 | Risk by merchant category | `merchant_risk` | Sorted bar chart |
| 4 | Priority review queue | `alert_queue` | Table with conditional formatting on risk score |

## Filters

- Event date range
- Province
- Channel
- Merchant category
- Risk tier

## Interpretation notes

- “Observed fraud” refers only to the synthetic simulator label.
- The alert rate is a workload measure, not a confirmed fraud rate.
- Risk reason codes are behavioural flags, not causal model explanations.
- Model metrics must be read together; a high recall can still create costly false positives.

## Accessibility

- Do not rely on red/green alone.
- Use direct, descriptive titles and percentage formatting.
- Keep legends near the chart and sort categorical bars.
- Add this visible subtitle: **Synthetic data • Portfolio demonstration • Human review required**.

