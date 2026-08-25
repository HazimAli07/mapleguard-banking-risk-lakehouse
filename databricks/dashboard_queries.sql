-- MapleGuard Lakeview dashboard datasets
-- All tables contain deterministic synthetic data only.

-- Dataset: overview
SELECT * FROM mapleguard.gold_overview;

-- Dataset: weekly_trend
SELECT
  week_start,
  transactions,
  value_cad,
  alerts,
  ROUND(100 * alert_rate, 2) AS alert_rate_pct,
  ROUND(100 * observed_fraud_rate, 2) AS observed_fraud_rate_pct,
  ROUND(average_risk_score, 4) AS average_risk_score
FROM mapleguard.gold_weekly_kpis
ORDER BY week_start;

-- Dataset: channel_risk
SELECT channel, transactions, alerts, ROUND(100 * alert_rate, 2) AS alert_rate_pct
FROM mapleguard.gold_risk_by_channel
ORDER BY alert_rate_pct DESC;

-- Dataset: province_risk
SELECT province, transactions, alerts, ROUND(100 * alert_rate, 2) AS alert_rate_pct
FROM mapleguard.gold_risk_by_province
ORDER BY alert_rate_pct DESC;

-- Dataset: merchant_risk
SELECT merchant_category, transactions, alerts, ROUND(100 * alert_rate, 2) AS alert_rate_pct
FROM mapleguard.gold_risk_by_merchant
ORDER BY alert_rate_pct DESC;

-- Dataset: model_metrics
SELECT metric, ROUND(value, 4) AS value
FROM mapleguard.gold_model_metrics
ORDER BY metric;

-- Dataset: alert_queue
SELECT
  transaction_id,
  event_ts,
  province,
  channel,
  merchant_category,
  amount_cad,
  ROUND(risk_score, 4) AS risk_score,
  risk_tier,
  reason_code
FROM mapleguard.gold_alert_queue
ORDER BY risk_score DESC, amount_cad DESC
LIMIT 250;

