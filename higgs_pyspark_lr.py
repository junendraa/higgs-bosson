"""
Higgs Boson Classification - PySpark MLlib
Algoritma: Logistic Regression (LR)
Dataset: training.csv (Kaggle Higgs Boson)
"""

import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

spark = SparkSession.builder \
    .appName("HiggsBoson_LR") \
    .master("yarn") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "2g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("=" * 50)
print("=== MEMULAI PENGUJIAN PYSPARK: LOGISTIC REGRESSION ===")
print("=" * 50)
print("Spark Version:", spark.version)

total_start = time.time()

print("\nMemuat data training.csv...")
DATA_PATH = "hdfs://master-template.lxd:9000/user/root/data/training.csv"

t0 = time.time()
df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)
df.cache()
jumlah_baris = df.count()
waktu_load = time.time() - t0

print(f"[-] Waktu Load Data  : {waktu_load:.2f} detik")
print(f"[-] Jumlah Baris Data: {jumlah_baris}")

print("\nMemproses dataset (VectorAssembler & StringIndexer)...")

df = df.withColumn(
    "label",
    when(col("Label") == "s", 1.0).otherwise(0.0)
)

feature_cols = [c for c in df.columns if c.startswith("DER_") or c.startswith("PRI_")]

for c in feature_cols:
    df = df.withColumn(c, col(c).cast("double"))

df = df.dropna(subset=feature_cols + ["label"])

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
scaler = StandardScaler(inputCol="features_raw", outputCol="features",
                        withMean=True, withStd=True)

df_prep = assembler.transform(df)

train_df, test_df = df_prep.randomSplit([0.7, 0.3], seed=42)

scaler_model = scaler.fit(train_df)
train_scaled = scaler_model.transform(train_df)
test_scaled = scaler_model.transform(test_df)

train_scaled.cache()
test_scaled.cache()

print("\nMemulai proses training Logistic Regression Terdistribusi...")

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=0.01
)

t_train = time.time()
lr_model = lr.fit(train_scaled)
waktu_training = time.time() - t_train

print(f"[-] Waktu Training   : {waktu_training:.2f} detik")

predictions = lr_model.transform(test_scaled)

auc_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
acc_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

auc = auc_evaluator.evaluate(predictions)
acc = acc_evaluator.evaluate(predictions)

total_waktu = time.time() - total_start

print("\n" + "=" * 50)
print("========== <== HASIL AKHIR ====================")
print(f"Kualitas Model (AUC)     : {auc:.4f}")
print(f"Kualitas Model (Akurasi) : {acc:.4f}")
print(f"Waktu Load Data          : {waktu_load:.2f} detik")
print(f"[-] Waktu Training       : {waktu_training:.2f} detik")
print(f"Total Waktu Eksekusi     : {total_waktu:.2f} detik")
print("=" * 50)

with open("/root/higsbosson/hasil_lr_pyspark.txt", "w") as f:
    f.write("HASIL PYSPARK - LOGISTIC REGRESSION\n")
    f.write("=" * 50 + "\n")
    f.write(f"Spark Version    : {spark.version}\n")
    f.write(f"Jumlah Data      : {jumlah_baris}\n")
    f.write(f"Jumlah Fitur     : {len(feature_cols)}\n")
    f.write(f"AUC              : {auc:.4f}\n")
    f.write(f"Accuracy         : {acc:.4f}\n")
    f.write(f"Waktu Load       : {waktu_load:.2f} detik\n")
    f.write(f"Waktu Training   : {waktu_training:.2f} detik\n")
    f.write(f"Total Waktu      : {total_waktu:.2f} detik\n")
    f.write("=" * 50 + "\n")

print("\nHasil disimpan ke: /root/higsbosson/hasil_lr_pyspark.txt")
spark.stop()
