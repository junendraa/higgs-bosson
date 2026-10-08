"""
Higgs Boson Classification - PySpark MLlib
Algoritma: Logistic Regression, Decision Tree, Random Forest, Gradient Boosted Tree
Dataset: training.csv (Kaggle Higgs Boson)
"""

import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier,
    GBTClassifier
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# ============================================================
# 1. INISIALISASI SPARK SESSION
# ============================================================
spark = SparkSession.builder \
    .appName("HiggsBosonClassification") \
    .master("spark://master-template:7077") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print("=" * 60)
print("Spark Version:", spark.version)
print("=" * 60)

# ============================================================
# 2. LOAD DATASET
# ============================================================
print("\n[1] Loading dataset...")
DATA_PATH = "file:///root/higsbosson/training.csv"

df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)

print(f"Total baris: {df.count()}")
print(f"Total kolom: {len(df.columns)}")
print("Kolom:", df.columns)

# ============================================================
# 3. PREPROCESSING
# ============================================================
print("\n[2] Preprocessing...")

# Cek nama kolom label (bisa 'Label' atau 'label')
label_col = None
for c in df.columns:
    if c.lower() == 'label':
        label_col = c
        break

if label_col is None:
    raise ValueError("Kolom label tidak ditemukan!")

print(f"Kolom label ditemukan: '{label_col}'")

# Konversi label: 's' -> 1.0 (signal), 'b' -> 0.0 (background)
df = df.withColumn(
    "label_num",
    when(col(label_col) == "s", 1.0)
    .when(col(label_col) == "b", 0.0)
    .when(col(label_col) == 1, 1.0)
    .when(col(label_col) == 0, 0.0)
    .otherwise(col(label_col).cast("double"))
)

# Distribusi label
print("\nDistribusi label:")
df.groupBy("label_num").count().show()

# Pilih 29 fitur sesuai jurnal (DER_* dan PRI_*)
feature_cols = [c for c in df.columns if c.startswith("DER_") or c.startswith("PRI_")]

# Jika tidak ada prefix DER/PRI, gunakan semua kolom numerik kecuali label
if len(feature_cols) < 5:
    exclude_cols = {label_col, "label_num", "EventId", "Weight", "KaggleSet", "KaggleWeight"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]

print(f"\nJumlah fitur yang digunakan: {len(feature_cols)}")
print("Fitur:", feature_cols[:10], "...")

# Drop baris dengan null
df = df.dropna(subset=feature_cols + ["label_num"])

# Cast semua fitur ke double
for c in feature_cols:
    df = df.withColumn(c, col(c).cast("double"))

# ============================================================
# 4. VECTOR ASSEMBLER & SCALER
# ============================================================
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)

# Prepare dataframe
df_prep = assembler.transform(df)

# ============================================================
# 5. SPLIT DATA: 70% TRAIN, 30% TEST
# ============================================================
print("\n[3] Split data 70% train, 30% test...")
train_df, test_df = df_prep.randomSplit([0.7, 0.3], seed=42)
print(f"Training: {train_df.count()} baris")
print(f"Testing : {test_df.count()} baris")

# Fit scaler pada training data
print("\n[4] Fitting scaler...")
scaler_model = scaler.fit(train_df)
train_scaled = scaler_model.transform(train_df)
test_scaled  = scaler_model.transform(test_df)

# Rename label column
train_scaled = train_scaled.drop("label").withColumnRenamed("label_num", "label")
test_scaled  = test_scaled.drop("label").withColumnRenamed("label_num", "label")

# Cache untuk performa
train_scaled.cache()
test_scaled.cache()

# ============================================================
# 6. EVALUATOR
# ============================================================
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

# ============================================================
# 7. TRAINING & EVALUASI 4 ALGORITMA
# ============================================================
results = {}

def train_evaluate(name, model):
    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"{'='*60}")
    start = time.time()
    fitted = model.fit(train_scaled)
    train_time = time.time() - start
    print(f"  Waktu training : {train_time:.2f} detik")

    start = time.time()
    predictions = fitted.transform(test_scaled)
    predict_time = time.time() - start

    auc = auc_evaluator.evaluate(predictions)
    acc = acc_evaluator.evaluate(predictions)

    print(f"  AUC            : {auc:.4f}")
    print(f"  Accuracy       : {acc:.4f}")
    print(f"  Waktu prediksi : {predict_time:.2f} detik")

    results[name] = {
        "AUC": round(auc, 4),
        "Accuracy": round(acc, 4),
        "Train Time (s)": round(train_time, 2),
        "Predict Time (s)": round(predict_time, 2)
    }
    return fitted

# --- Logistic Regression ---
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=0.01
)
lr_model = train_evaluate("Logistic Regression", lr)

# --- Decision Tree ---
dt = DecisionTreeClassifier(
    featuresCol="features",
    labelCol="label",
    maxDepth=10,
    seed=42
)
dt_model = train_evaluate("Decision Tree", dt)

# --- Random Forest ---
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=50,
    maxDepth=10,
    seed=42
)
rf_model = train_evaluate("Random Forest", rf)

# --- Gradient Boosted Tree ---
gbt = GBTClassifier(
    featuresCol="features",
    labelCol="label",
    maxIter=50,
    maxDepth=5,
    seed=42
)
gbt_model = train_evaluate("Gradient Boosted Tree", gbt)

# ============================================================
# 8. RINGKASAN HASIL
# ============================================================
print("\n")
print("=" * 70)
print("  RINGKASAN HASIL - HIGGS BOSON CLASSIFICATION (PySpark)")
print("=" * 70)
print(f"{'Classifier':<25} {'AUC':>8} {'Accuracy':>10} {'Train(s)':>10} {'Predict(s)':>12}")
print("-" * 70)
for name, metrics in results.items():
    print(f"{name:<25} {metrics['AUC']:>8.4f} {metrics['Accuracy']:>10.4f} "
          f"{metrics['Train Time (s)']:>10.2f} {metrics['Predict Time (s)']:>12.2f}")
print("=" * 70)

# Simpan hasil ke file
with open("/root/higsbosson/hasil_pyspark.txt", "w") as f:
    f.write("HASIL EKSPERIMEN PYSPARK - HIGGS BOSON CLASSIFICATION\n")
    f.write("=" * 70 + "\n")
    f.write(f"{'Classifier':<25} {'AUC':>8} {'Accuracy':>10} {'Train(s)':>10} {'Predict(s)':>12}\n")
    f.write("-" * 70 + "\n")
    for name, metrics in results.items():
        f.write(f"{name:<25} {metrics['AUC']:>8.4f} {metrics['Accuracy']:>10.4f} "
                f"{metrics['Train Time (s)']:>10.2f} {metrics['Predict Time (s)']:>12.2f}\n")
    f.write("=" * 70 + "\n")

print("\nHasil disimpan ke: /root/higsbosson/hasil_pyspark.txt")
print("\nDone!")
spark.stop()
