Higgs Boson Classification - PySpark MLlib

Proyek klasifikasi partikel Higgs Boson menggunakan algoritma Logistic Regression yang dijalankan secara terdistribusi dengan Apache Spark di atas cluster Hadoop.

DATASET
Source: Kaggle - Higgs Boson Machine Learning Challenge (https://www.kaggle.com/c/higgs-boson)
Ukuran: 250.000 baris, 30 fitur (DER_* dan PRI_*)
Label: s (signal) -> 1.0, b (background) -> 0.0
CATATAN: Dataset tidak disertakan di repo ini karena ukuran file melebihi batas maksimum GitHub (lebih dari 100MB). Silakan download langsung dari Kaggle.

TECH STACK
Apache Spark 3.5.5 (YARN mode)
Hadoop HDFS 3.4.1
Hadoop YARN 3.4.1
PySpark MLlib
LXD Container (simulasi multi-node cluster)

ARSITEKTUR CLUSTER
Master Node (master-template): Spark Master, HDFS NameNode, YARN ResourceManager
Worker Node 1 (10.10.10.10): 6 Core, 12.4 GiB
Worker Node 2 (10.10.10.11): 6 Core, 12.4 GiB
Worker Node 3 (10.10.10.12): 6 Core, 12.4 GiB
Total: 18 Core, 37.3 GiB Memory

PIPELINE
1. Load dataset dari HDFS
2. Preprocessing: konversi label, cast fitur ke double, drop null
3. VectorAssembler untuk menggabungkan fitur
4. StandardScaler untuk normalisasi fitur
5. Split 70% train dan 30% test
6. Training Logistic Regression dengan maxIter=100 dan regParam=0.01
7. Evaluasi AUC dan Accuracy

HASIL
AUC              : 0.8111
Accuracy         : 0.7468
Jumlah Data      : 250.000 baris
Jumlah Fitur     : 30 fitur
Waktu Load Data  : 40.31 detik
Waktu Training   : 24.66 detik
Total Eksekusi   : 85.47 detik

CARA MENJALANKAN
1. Pastikan HDFS dan YARN sudah jalan
   start-dfs.sh && start-yarn.sh

2. Upload dataset ke HDFS
   hdfs dfs -mkdir -p /user/root/data
   hdfs dfs -put training.csv /user/root/data/

3. Jalankan script
   spark-submit --master yarn --deploy-mode client higgs_pyspark_lr.py
