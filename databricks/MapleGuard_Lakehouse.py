# Databricks notebook source
# MAGIC %md
# MAGIC # MapleGuard — Banking Transaction Risk Lakehouse
# MAGIC **Author:** Hazim Ali  
# MAGIC **Portfolio goal:** demonstrate the data engineering, AI, risk, controls, and communication skills repeatedly requested in Winter 2027 Canadian bank co-op postings.
# MAGIC
# MAGIC > Every record is deterministic and synthetic. This notebook is not a production fraud, credit, or customer-decision system.

# COMMAND ----------
from datetime import datetime, timedelta
import math
import numpy as np
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
from pyspark.sql import types as T

SEED = 2027
ROWS = 60_000
SCHEMA = "mapleguard"
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Bronze — deterministic synthetic events
# MAGIC Raw records are generated in-memory with anonymous identifiers and behavioural attributes only.

# COMMAND ----------
rng = np.random.default_rng(SEED)
provinces = np.array(["ON", "QC", "BC", "AB", "MB", "SK", "NS", "NB"])
channels = np.array(["Point of sale", "E-commerce", "Mobile wallet", "ATM", "Recurring"])
merchants = np.array(["Grocery", "Fuel", "Dining", "Travel", "Electronics", "Digital services", "Cash", "Other"])
segments = np.array(["Everyday", "Student", "New-to-bank", "Affluent", "Small business"])

day_offset = rng.integers(0, 238, ROWS)
hour = rng.integers(0, 24, ROWS)
minute = rng.integers(0, 60, ROWS)
event_ts = pd.Timestamp("2026-01-01") + pd.to_timedelta(day_offset, unit="D") + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(minute, unit="m")
province = rng.choice(provinces, ROWS, p=[.39, .22, .14, .12, .04, .03, .04, .02])
channel = rng.choice(channels, ROWS, p=[.41, .25, .14, .11, .09])
merchant = rng.choice(merchants, ROWS, p=[.22, .11, .17, .08, .09, .13, .08, .12])
segment = rng.choice(segments, ROWS, p=[.43, .18, .12, .15, .12])
base_amount = rng.lognormal(4.15, 1.02, ROWS)
multiplier = np.select([merchant == "Travel", merchant == "Electronics", merchant == "Cash", merchant == "Grocery"], [2.1, 1.7, 1.35, .82], default=1.0)
amount = np.clip(base_amount * multiplier, 1.25, 4900).round(2)
is_international = rng.binomial(1, np.where(merchant == "Travel", .24, .055), ROWS)
device_trust = np.clip(rng.normal(72, 20, ROWS), 0, 100).round(1)
account_age = np.clip(rng.gamma(2.2, 470, ROWS), 2, 5000).astype(int)
velocity = np.clip(rng.poisson(2.6, ROWS) + 1, 1, 22)
distance = np.clip(rng.exponential(31, ROWS), .1, 1500).round(1)
linear_risk = (-9.20 + .62*np.log1p(amount) + 1.22*(channel == "E-commerce") + .82*(channel == "Mobile wallet") + 2.05*is_international + .070*np.maximum(50-device_trust, 0) + .48*np.maximum(velocity-5, 0) + .0110*np.maximum(distance-45, 0) + 1.10*((hour <= 4) | (hour >= 23)) + 1.25*(account_age < 90) + .72*(merchant == "Electronics") + .48*(merchant == "Digital services"))
probability = np.clip(1 / (1 + np.exp(-linear_risk)), .001, .94)
is_fraud = rng.binomial(1, probability, ROWS).astype(int)

pdf = pd.DataFrame({
    "transaction_id": [f"TX{i:07d}" for i in range(1, ROWS + 1)],
    "customer_id": [f"C{i:06d}" for i in rng.integers(1, 15000, ROWS)],
    "event_ts": event_ts,
    "province": province,
    "channel": channel,
    "merchant_category": merchant,
    "customer_segment": segment,
    "amount_cad": amount,
    "is_international": is_international,
    "is_card_present": (channel == "Point of sale").astype(int),
    "device_trust_score": device_trust,
    "account_age_days": account_age,
    "transactions_24h": velocity,
    "distance_from_home_km": distance,
    "hour_of_day": hour,
    "is_fraud": is_fraud,
})
bronze = spark.createDataFrame(pdf)
bronze.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{SCHEMA}.bronze_transactions")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Silver — quality checks and trusted features
# MAGIC The pipeline validates the business key and numeric ranges before producing a clean, time-aware feature table.

# COMMAND ----------
bronze = spark.table(f"{SCHEMA}.bronze_transactions")
quality = bronze.agg(
    F.count("*").alias("rows"),
    F.countDistinct("transaction_id").alias("unique_ids"),
    F.sum(F.col("transaction_id").isNull().cast("int")).alias("null_ids"),
    F.sum((F.col("amount_cad") <= 0).cast("int")).alias("invalid_amounts"),
    F.sum((~F.col("device_trust_score").between(0, 100)).cast("int")).alias("invalid_device_scores"),
).first()
assert quality.rows == ROWS and quality.unique_ids == ROWS
assert quality.null_ids == 0 and quality.invalid_amounts == 0 and quality.invalid_device_scores == 0

silver = (
    bronze
    .withColumn("event_date", F.to_date("event_ts"))
    .withColumn("week_start", F.to_date(F.date_trunc("week", "event_ts")))
    .withColumn("log_amount", F.log1p("amount_cad"))
    .withColumn("is_unusual_hour", ((F.hour("event_ts") <= 4) | (F.hour("event_ts") >= 23)).cast("int"))
    .withColumn("is_new_account", (F.col("account_age_days") < 90).cast("int"))
    .withColumn("is_low_device_trust", (F.col("device_trust_score") < 35).cast("int"))
)
silver.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{SCHEMA}.silver_transactions")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Model — event-time validation and holdout evaluation
# MAGIC The final 30% of events form the holdout period. A class-weighted logistic model keeps the method interpretable and reproducible.

# COMMAND ----------
silver = spark.table(f"{SCHEMA}.silver_transactions")
events = silver.withColumn("_event_epoch", F.col("event_ts").cast("double"))
train_cutoff, validation_cutoff = events.approxQuantile("_event_epoch", [0.56, 0.70], 0.001)
train = events.filter(F.col("_event_epoch") <= F.lit(train_cutoff))
validation = events.filter((F.col("_event_epoch") > F.lit(train_cutoff)) & (F.col("_event_epoch") <= F.lit(validation_cutoff)))
development = events.filter(F.col("_event_epoch") <= F.lit(validation_cutoff))
holdout = events.filter(F.col("_event_epoch") > F.lit(validation_cutoff))

categorical = ["province", "channel", "merchant_category", "customer_segment"]
numeric = ["log_amount", "is_international", "is_card_present", "device_trust_score", "account_age_days", "transactions_24h", "distance_from_home_km", "hour_of_day"]
indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in categorical]
encoder = OneHotEncoder(inputCols=[f"{c}_idx" for c in categorical], outputCols=[f"{c}_ohe" for c in categorical])
assembler = VectorAssembler(inputCols=[f"{c}_ohe" for c in categorical] + numeric, outputCol="features")

def add_class_weights(frame):
    counts = frame.groupBy("is_fraud").count().collect()
    count_map = {row["is_fraud"]: row["count"] for row in counts}
    positive_weight = count_map.get(0, 1) / max(count_map.get(1, 1), 1)
    return frame.withColumn("class_weight", F.when(F.col("is_fraud") == 1, F.lit(float(positive_weight))).otherwise(F.lit(1.0)))

lr = LogisticRegression(featuresCol="features", labelCol="is_fraud", weightCol="class_weight", maxIter=80, regParam=.02, elasticNetParam=0.0)
pipeline = Pipeline(stages=indexers + [encoder, assembler, lr])
threshold_model = pipeline.fit(add_class_weights(train))
validation_pairs = (
    threshold_model.transform(validation)
    .select("is_fraud", vector_to_array("probability")[1].alias("risk_score"))
    .toPandas()
)

threshold_results = []
for threshold in np.arange(.10, .81, .02):
    predicted = validation_pairs["risk_score"].to_numpy() >= threshold
    actual = validation_pairs["is_fraud"].to_numpy() == 1
    tp_v = int(np.sum(predicted & actual))
    fp_v = int(np.sum(predicted & ~actual))
    fn_v = int(np.sum(~predicted & actual))
    precision_v = tp_v / max(tp_v + fp_v, 1)
    recall_v = tp_v / max(tp_v + fn_v, 1)
    f1_v = 2 * precision_v * recall_v / max(precision_v + recall_v, 1e-12)
    threshold_results.append((float(threshold), recall_v, f1_v))
eligible_thresholds = [row for row in threshold_results if row[1] >= .55]
THRESHOLD = max(eligible_thresholds or threshold_results, key=lambda row: row[2])[0]

model = threshold_model

holdout_scored = model.transform(holdout).withColumn("risk_score", vector_to_array("probability")[1])
holdout_scored = holdout_scored.withColumn("is_alert", (F.col("risk_score") >= THRESHOLD).cast("int"))

roc_auc = BinaryClassificationEvaluator(labelCol="is_fraud", rawPredictionCol="rawPrediction", metricName="areaUnderROC").evaluate(holdout_scored)
avg_precision = BinaryClassificationEvaluator(labelCol="is_fraud", rawPredictionCol="rawPrediction", metricName="areaUnderPR").evaluate(holdout_scored)
confusion = holdout_scored.agg(
    F.sum(((F.col("is_fraud") == 1) & (F.col("is_alert") == 1)).cast("int")).alias("tp"),
    F.sum(((F.col("is_fraud") == 0) & (F.col("is_alert") == 1)).cast("int")).alias("fp"),
    F.sum(((F.col("is_fraud") == 1) & (F.col("is_alert") == 0)).cast("int")).alias("fn"),
    F.sum(((F.col("is_fraud") == 0) & (F.col("is_alert") == 0)).cast("int")).alias("tn"),
).first()
precision = confusion.tp / max(confusion.tp + confusion.fp, 1)
recall = confusion.tp / max(confusion.tp + confusion.fn, 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-12)
false_positive_rate = confusion.fp / max(confusion.fp + confusion.tn, 1)

# Score the full history for monitored operational views.
scored = model.transform(silver).withColumn("risk_score", vector_to_array("probability")[1])
scored = (
    scored
    .withColumn("is_alert", (F.col("risk_score") >= THRESHOLD).cast("int"))
    .withColumn("risk_tier", F.when(F.col("risk_score") >= .75, "Critical").when(F.col("risk_score") >= .50, "High").when(F.col("risk_score") >= .25, "Guarded").otherwise("Low"))
    .withColumn("reason_code", F.concat_ws(", ",
        F.when(F.col("is_international") == 1, F.lit("international")),
        F.when(F.col("device_trust_score") < 35, F.lit("low device trust")),
        F.when(F.col("transactions_24h") >= 8, F.lit("high velocity")),
        F.when(F.col("amount_cad") >= 500, F.lit("high amount")),
        F.when(F.col("is_unusual_hour") == 1, F.lit("unusual hour")),
        F.when(F.col("distance_from_home_km") >= 120, F.lit("distance anomaly")),
        F.when(F.col("is_new_account") == 1, F.lit("new account")),
    ))
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Gold — decision-ready risk products

# COMMAND ----------
overview = scored.agg(
    F.count("*").alias("transactions"),
    F.round(F.sum("amount_cad"), 2).alias("value_cad"),
    F.sum("is_alert").alias("alerts"),
    F.avg("is_alert").alias("alert_rate"),
    F.avg("is_fraud").alias("observed_fraud_rate"),
    F.avg("risk_score").alias("average_risk_score"),
).withColumn("model_roc_auc", F.lit(float(roc_auc))).withColumn("model_recall", F.lit(float(recall)))
overview.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{SCHEMA}.gold_overview")

def write_time_mart(group_col, table_name):
    mart = scored.groupBy(group_col).agg(
        F.count("*").alias("transactions"),
        F.round(F.sum("amount_cad"), 2).alias("value_cad"),
        F.sum("is_alert").alias("alerts"),
        F.sum("is_fraud").alias("observed_fraud"),
        F.avg("risk_score").alias("average_risk_score"),
    ).withColumn("alert_rate", F.col("alerts") / F.col("transactions")).withColumn("observed_fraud_rate", F.col("observed_fraud") / F.col("transactions"))
    mart.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{SCHEMA}.{table_name}")

def write_dimension_mart(group_col, table_name):
    mart = scored.groupBy(group_col).agg(
        F.count("*").alias("transactions"),
        F.round(F.sum("amount_cad"), 2).alias("value_cad"),
        F.sum("is_alert").alias("alerts"),
        F.sum("is_fraud").alias("observed_fraud"),
        F.avg("risk_score").alias("average_risk_score"),
    ).withColumn("alert_rate", F.col("alerts") / F.col("transactions")).withColumn("observed_fraud_rate", F.col("observed_fraud") / F.col("transactions"))
    mart.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{SCHEMA}.{table_name}")

write_time_mart("event_date", "gold_daily_kpis")
write_time_mart("week_start", "gold_weekly_kpis")
write_dimension_mart("channel", "gold_risk_by_channel")
write_dimension_mart("province", "gold_risk_by_province")
write_dimension_mart("merchant_category", "gold_risk_by_merchant")

metrics = spark.createDataFrame([
    ("roc_auc", float(roc_auc)),
    ("average_precision", float(avg_precision)),
    ("precision", float(precision)),
    ("recall", float(recall)),
    ("f1", float(f1)),
    ("false_positive_rate", float(false_positive_rate)),
    ("decision_threshold", float(THRESHOLD)),
], ["metric", "value"])
metrics.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{SCHEMA}.gold_model_metrics")

alert_queue = scored.filter("is_alert = 1").select(
    "transaction_id", "event_ts", "province", "channel", "merchant_category", "customer_segment", "amount_cad", "risk_score", "risk_tier", "reason_code", "is_fraud"
).orderBy(F.desc("risk_score"), F.desc("amount_cad")).limit(500)
alert_queue.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{SCHEMA}.gold_alert_queue")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Validation summary
# MAGIC The following tables power the Databricks Lakeview dashboard. Treat the observed label and all performance metrics as synthetic demonstrations only.

# COMMAND ----------
display(spark.table(f"{SCHEMA}.gold_overview"))
display(spark.table(f"{SCHEMA}.gold_model_metrics"))
display(spark.table(f"{SCHEMA}.gold_alert_queue").limit(20))
