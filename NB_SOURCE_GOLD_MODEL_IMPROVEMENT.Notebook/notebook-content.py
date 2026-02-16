# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "3ee4579c-22ab-494d-8b27-fbf8a3b342fb",
# META       "default_lakehouse_name": "LK_SOURCE_CREDIT_GOLD",
# META       "default_lakehouse_workspace_id": "76eb933a-950e-4895-8f00-76ccb4a5f37d",
# META       "known_lakehouses": [
# META         {
# META           "id": "3ee4579c-22ab-494d-8b27-fbf8a3b342fb"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

# Loading dataset ML from Lakehouse Gold
df_ml = spark.read.format("delta").load(
    "abfss://76eb933a-950e-4895-8f00-76ccb4a5f37d@onelake.dfs.fabric.microsoft.com/3ee4579c-22ab-494d-8b27-fbf8a3b342fb/Tables/dbo/credit_scoring_ml"
)

display(df_ml.limit(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Identification des variables catégorielles et numériques
categorical_cols = [c for c, t in df_ml.dtypes if t == "string" and c != "credit_risk"]
numeric_cols = [c for c, t in df_ml.dtypes if t != "string" and c != "credit_risk"]

print("Variables catégorielles :", categorical_cols)
print("Variables numériques :", numeric_cols)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import col
# forcing the target to be int if in case it is not(but here we saw that it is) ==> the best practice
df_ml = df_ml.withColumn("label", col("credit_risk").cast("double"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Pour chaque varaible categoriel on va calculer le taux de defaux le plus bas pour degager les cat de reference
# Afin que  les coefficients du modèle logistique soient interprétables comme :“sur‑risque par rapport à la catégorie saine”.
# calculer les catégories triées par taux de défaut
from pyspark.sql import functions as F

category_orders = {}

for col in categorical_cols:
    df_rates = (
        df_ml.groupBy(col)
        .agg(F.mean("credit_risk").alias("default_rate"))
        .orderBy(F.col("default_rate").desc())  # tri décroissant = la moins risquée en dernier
    )
    
    ordered_labels = [row[col] for row in df_rates.collect()]
    category_orders[col] = ordered_labels



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ici on impose l'indexation avec un ordre basé sur 
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import StringIndexerModel

indexers = []

for col in categorical_cols:
    indexer_model = StringIndexerModel.from_labels(
        labels=category_orders[col],     # ordre imposé
        inputCol=col,
        outputCol=f"{col}_idx"
    )
    indexers.append(indexer_model)



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.ml.feature import OneHotEncoder
encoder = OneHotEncoder(
    inputCols=[f"{col}_idx" for col in categorical_cols],
    outputCols=[f"{col}_ohe" for col in categorical_cols],
    dropLast=True
)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.ml.feature import VectorAssembler

# Colonnes OHE
ohe_cols = [f"{col}_ohe" for col in categorical_cols]

# VectorAssembler complet
assembler = VectorAssembler(
    inputCols=ohe_cols + numeric_cols,
    outputCol="features"
)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=50,          # Bonne pratique : éviter les modèles trop complexes
    regParam=0.0,        # Pas de régularisation pour un modèle interprétable
    elasticNetParam=0.0  # 0 = L2, standard bancaire
)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.ml import Pipeline
pipeline = Pipeline(stages=indexers + [encoder, assembler, lr])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Split train/test
train_df, test_df = df_ml.randomSplit([0.7, 0.3], seed=42)

print("Taille train :", train_df.count())
print("Taille test  :", test_df.count())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# On va equilibrer les classes par la methode recommandée en scoring bancaire = weightCol  
# on calcul un ratio qui sera appliqué comme poids à la classe minotitaire(label=0)
from pyspark.sql import functions as F

nb_bad = train_df.filter(F.col("label") == 1).count()
nb_good = train_df.filter(F.col("label") == 0).count()

ratio = nb_good / nb_bad
ratio

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# On ajoute une colonne weight(poid) dans le train dataframe
train_df = train_df.withColumn(
    "weight",
    F.when(F.col("label") == 1, ratio).otherwise(1.0)
)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#Ainsi le model de la regression logistique(lr) devient 
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    weightCol="weight",   # ← ajout ici
    maxIter=50,
    regParam=0.0,
    elasticNetParam=0.0
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Réconstruction de Pipeline
pipeline = Pipeline(stages=indexers + [encoder, assembler, lr])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#Entrainement du model
model = pipeline.fit(train_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Evaluation du model sur le dataframe test
predictions = model.transform(test_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# calcul de l'AUC (ROC)
from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator_auc = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

auc = evaluator_auc.evaluate(predictions)
print("AUC :", auc)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#Calculer le KS (Kolmogorov–Smirnov)
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.functions import vector_to_array

# 1) Convertir le vecteur probability en array, puis extraire la proba de la classe 1
df = predictions.withColumn(
    "prob_array",
    vector_to_array("probability")
).withColumn(
    "proba_1",
    F.col("prob_array")[1]
)

# 2) Trier par score décroissant
df = df.orderBy(F.col("proba_1").desc())

# 3) Calcul des distributions cumulées bons / mauvais
w = Window.orderBy(F.col("proba_1").desc()).rowsBetween(Window.unboundedPreceding, Window.currentRow)

total_bad = df.agg(F.sum("label")).first()[0]
total_good = df.agg(F.sum(1 - F.col("label"))).first()[0]

df = df.withColumn("cum_bad", F.sum("label").over(w) / F.lit(total_bad))
df = df.withColumn("cum_good", F.sum(1 - F.col("label")).over(w) / F.lit(total_good))

# 4) KS = max(|cum_bad - cum_good|)
ks_value = df.select(F.max(F.abs(F.col("cum_bad") - F.col("cum_good")))).first()[0]

print("KS bancaire =", ks_value)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType

# UDF pour extraire la 2e valeur du vecteur rawPrediction
def extract_score(v):
    try:
        return float(v[1])
    except:
        return None

extract_score_udf = F.udf(extract_score, DoubleType())

df_ks = predictions.withColumn("score", extract_score_udf(F.col("rawPrediction")))


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

cum = df_ks.orderBy("score") \
    .withColumn("cum_good", F.sum(F.when(F.col("label") == 0, 1).otherwise(0)).over(Window.orderBy("score"))) \
    .withColumn("cum_bad",  F.sum(F.when(F.col("label") == 1, 1).otherwise(0)).over(Window.orderBy("score")))

total_good = df_ks.filter("label = 0").count()
total_bad  = df_ks.filter("label = 1").count()

ks = cum.withColumn(
    "ks",
    F.abs(F.col("cum_bad")/total_bad - F.col("cum_good")/total_good)
).agg(F.max("ks")).collect()[0][0]

print("KS :", ks)



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator_recall = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="recallByLabel"
)

recall_bad = evaluator_recall.evaluate(predictions, {evaluator_recall.metricLabel: 1})
print("Recall classe 1 (mauvais) :", recall_bad)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

predictions.groupBy("label", "prediction").count().show()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

# Récupérer les labels et prédictions depuis Spark
pdf = predictions.select("label", "prediction", "probability").toPandas()

# Matrice de confusion
conf_mat = confusion_matrix(pdf["label"], pdf["prediction"])


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Extraire la probabilité de la classe 1
pdf["p1"] = pdf["probability"].apply(lambda v: float(v[1]))

# Courbe ROC
fpr, tpr, _ = roc_curve(pdf["label"], pdf["p1"])
roc_auc = auc(fpr, tpr)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# --- Matrice de confusion ---
sns.heatmap(
    conf_mat,
    annot=True,
    fmt='d',
    cmap='viridis',
    cbar=False,
    xticklabels=['Prédit : Bon', 'Prédit : Mauvais'],
    yticklabels=['Réel : Bon', 'Réel : Mauvais'],
    ax=axes[0]
)
axes[0].set_title("Matrice de confusion - Test", pad=20)
axes[0].set_xlabel('Valeurs prédites')
axes[0].set_ylabel('Valeurs réelles')

# --- Courbe ROC ---
axes[1].plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.3f}')
axes[1].plot([0, 1], [0, 1], color='gray', linestyle='--')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('Taux de Faux Positifs (FPR)')
axes[1].set_ylabel('Taux de Vrais Positifs (TPR)')
axes[1].set_title('Courbe ROC')
axes[1].legend(loc="lower right")

plt.tight_layout()
plt.show()

please upload to lakehouse

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

indexers = model.stages[:-3]      # StringIndexers
encoder = model.stages[-3]        # OneHotEncoderModel
assembler = model.stages[-2]      # VectorAssembler
lr_model = model.stages[-1]       # LogisticRegressionModel


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

ohe_feature_names = []

for input_col, size in zip(encoder.getInputCols(), encoder.categorySizes):
    for i in range(size - 1):   # Spark dropLast=True
        ohe_feature_names.append(f"{input_col}_{i}")



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

final_feature_names = ohe_feature_names + numeric_cols

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

print("Nombre de features :", len(final_feature_names))
print("Nombre de coefficients :", len(lr_model.coefficients))


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import pandas as pd

coef_df = pd.DataFrame({
    "feature": final_feature_names,
    "coefficient": lr_model.coefficients.toArray()
}).sort_values(by="coefficient", key=abs, ascending=False)

coef_df


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

print("Nombre de features :", len(final_feature_names))
print("Nombre de coefficients :", len(lr_model.coefficients))


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 1. Sélection des variables (suppression des variables faibles)
# ============================================================

# Variables catégorielles à retirer (faibles ou inutiles)
vars_to_remove = [
    "other_installment_plans_idx_1",
    "job_idx_0",
    "job_idx_1",
    "other_debtors_idx_1",
    "employment_duration_idx_3",
    "savings_idx_1",
    "housing_idx_0",
    "purpose_idx_4"
]

# Filtrer les colonnes catégorielles
categorical_cols_opt = [c for c in categorical_cols if c not in vars_to_remove]

# Filtrer les colonnes numériques (si certaines sont à retirer)
numeric_cols_opt = numeric_cols.copy()  # ici aucune numérique à retirer


# ============================================================
# 2. Pipeline optimisé : Indexer + OHE + Assembler + LR
# ============================================================

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

# Indexation
indexers_opt = [
    StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
    for c in categorical_cols_opt
]

# Colonnes indexées
indexed_cols_opt = [f"{c}_idx" for c in categorical_cols_opt]

# Encodage OneHot
encoder_opt = OneHotEncoder(
    inputCols=indexed_cols_opt,
    outputCols=[f"{c}_ohe" for c in indexed_cols_opt],
    dropLast=True
)

# Colonnes OHE finales
ohe_cols_opt = [f"{c}_ohe" for c in indexed_cols_opt]

# Assemblage des features
assembler_opt = VectorAssembler(
    inputCols=ohe_cols_opt + numeric_cols_opt,
    outputCol="features"
)

# Modèle logistique
lr_opt = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    weightCol="weight",
    maxIter=50,
    regParam=0.01
)

# Pipeline complet
pipeline_opt = Pipeline(stages=indexers_opt + [encoder_opt, assembler_opt, lr_opt])


# ============================================================
# 3. Entraînement du modèle optimisé
# ============================================================

model_opt = pipeline_opt.fit(train_df)
pred_opt = model_opt.transform(test_df)


# ============================================================
# 4. Évaluation : AUC, KS, Recall
# ============================================================

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType

# AUC
evaluator_auc = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
auc_opt = evaluator_auc.evaluate(pred_opt)

# KS via UDF (compatible Fabric)
def extract_score(v):
    try:
        return float(v[1])
    except:
        return None

extract_score_udf = F.udf(extract_score, DoubleType())

df_ks = pred_opt.withColumn("score", extract_score_udf(F.col("rawPrediction")))

cum = df_ks.orderBy("score") \
    .withColumn("cum_good", F.sum(F.when(F.col("label") == 0, 1).otherwise(0)).over(Window.orderBy("score"))) \
    .withColumn("cum_bad",  F.sum(F.when(F.col("label") == 1, 1).otherwise(0)).over(Window.orderBy("score")))

total_good = df_ks.filter("label = 0").count()
total_bad  = df_ks.filter("label = 1").count()

ks_opt = cum.withColumn(
    "ks",
    F.abs(F.col("cum_bad")/total_bad - F.col("cum_good")/total_good)
).agg(F.max("ks")).collect()[0][0]

# Recall classe 1
evaluator_recall = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="recallByLabel"
)
recall_opt = evaluator_recall.evaluate(pred_opt, {evaluator_recall.metricLabel: 1})

print("=== PERFORMANCE MODÈLE OPTIMISÉ ===")
print("AUC :", auc_opt)
print("KS :", ks_opt)
print("Recall classe 1 :", recall_opt)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import math

ScoreRef = 600
OddsRef = 50
PDO = 20

B = PDO / math.log(2)
A = ScoreRef + B * math.log(OddsRef)

A, B


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import pandas as pd

coef_df["points"] = -B * coef_df["coefficient"]
coef_df.sort_values(by="points", ascending=False, inplace=True)
coef_df.head(20)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

beta0 = lr_model.intercept
BaseScore = A - B * beta0
BaseScore

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import math
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

# === 1. UDF pour extraire la probabilité de défaut (classe 1) ===
def extract_prob_bad(v):
    try:
        return float(v[1])   # lecture Python du vecteur ML
    except:
        return None

extract_prob_bad_udf = F.udf(extract_prob_bad, DoubleType())

# === 2. UDF pour calculer le score bancaire ===
def compute_score(p):
    try:
        eps = 1e-9
        p = min(max(p, eps), 1 - eps)
        odds = p / (1 - p)
        return A - B * math.log(odds)
    except:
        return None

score_udf = F.udf(compute_score, DoubleType())

# === 3. Application au DataFrame Spark ===
scored = pred_opt.withColumn(
    "prob_bad",
    extract_prob_bad_udf(F.col("probability"))
).withColumn(
    "score",
    score_udf(F.col("prob_bad"))
)

scored.select("label", "prob_bad", "score").show(10)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import matplotlib.pyplot as plt

pdf_scores = scored.select("score").toPandas()

plt.figure(figsize=(10,5))
plt.hist(pdf_scores["score"], bins=30, color="steelblue", edgecolor="black")
plt.title("Distribution des scores")
plt.xlabel("Score")
plt.ylabel("Nombre de clients")
plt.show()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

bins = [0, 650, 700, 750, 800, 1000]
labels = ["<650", "650-700", "700-750", "750-800", ">800"]

pdf = scored.select("label", "score").toPandas()
pdf["score_band"] = pd.cut(pdf["score"], bins=bins, labels=labels, right=False)

pdf.groupby("score_band")["label"].mean()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import numpy as np

# On teste tous les cut-offs possibles
thresholds = np.linspace(pdf["score"].min(), pdf["score"].max(), 200)

best_ks = -1
best_cutoff = None

for t in thresholds:
    pred = (pdf["score"] < t).astype(int)  # 1 = mauvais
    tp = ((pred == 1) & (pdf["label"] == 1)).sum()
    fp = ((pred == 1) & (pdf["label"] == 0)).sum()
    fn = ((pred == 0) & (pdf["label"] == 1)).sum()
    tn = ((pred == 0) & (pdf["label"] == 0)).sum()

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    ks = abs(tpr - fpr)

    if ks > best_ks:
        best_ks = ks
        best_cutoff = t

best_cutoff, best_ks


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import pandas as pd

cutoff = 719.2721999297512

pdf = scored.select("label", "score").toPandas()

def decision_from_score(s):
    if s < 650:
        return "Refus automatique"
    elif 650 <= s < cutoff:
        return "Analyse manuelle / conditions"
    elif cutoff <= s < 780:
        return "Acceptation automatique"
    else:
        return "Premium / très bon client"

pdf["decision"] = pdf["score"].apply(decision_from_score)

pdf.head()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

summary = (
    pdf
    .groupby("decision")
    .agg(
        volume=("label", "count"),
        default_rate=("label", "mean")
    )
    .sort_values("default_rate", ascending=False)
)

summary["default_rate"] = (summary["default_rate"] * 100).round(2)
summary


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

accepted = pdf[pdf["score"] >= cutoff]
rejected = pdf[pdf["score"] < cutoff]

accept_rate = len(accepted) / len(pdf)
default_rate_accepted = accepted["label"].mean()
default_rate_rejected = rejected["label"].mean()

print("Taux d'acceptation :", round(accept_rate * 100, 2), "%")
print("Taux de défaut parmi les acceptés :", round(default_rate_accepted * 100, 2), "%")
print("Taux de défaut parmi les refusés :", round(default_rate_rejected * 100, 2), "%")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

bins = [0, 650, 700, 719.27, 780, 1000]
labels = ["<650", "650-700", "700-719", "719-780", ">780"]

pdf["score_band"] = pd.cut(pdf["score"], bins=bins, labels=labels, right=False)

pd_band = (
    pdf.groupby("score_band")["label"]
       .agg(["count", "mean"])
       .rename(columns={"count": "volume", "mean": "pd_empirique"})
)

pd_band["pd_empirique"] = (pd_band["pd_empirique"] * 100).round(2)
pd_band


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
