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

# Identification categoriel columns (toutes les colonnes de type string sauf la cible)
categorical_cols = [
    col for col, dtype in df_ml.dtypes
    if dtype == "string" and col != "credit_risk"
]

categorical_cols


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import col
# forcing the target to be int if in case it is not
df_ml = df_ml.withColumn("label", col("credit_risk").cast("double"))


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.ml.feature import StringIndexer

# Création d'un StringIndexer pour chaque colonne catégorielle
indexers = [
    StringIndexer(
        inputCol=col,
        outputCol=f"{col}_idx",
        handleInvalid="keep"   # Bonne pratique : évite les erreurs si une nouvelle catégorie apparaît
    )
    for col in categorical_cols
]

indexers


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.ml.feature import OneHotEncoder

# Création d'un OneHotEncoder pour chaque colonne indexée
encoder = OneHotEncoder(
    inputCols=[f"{col}_idx" for col in categorical_cols],
    outputCols=[f"{col}_ohe" for col in categorical_cols],
    dropLast=True  # Bonne pratique : évite la multicolinéarité (dummy variable trap)
)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.ml.feature import VectorAssembler

# Colonnes OHE générées par OneHotEncoder
ohe_cols = [f"{col}_ohe" for col in categorical_cols]

# VectorAssembler pour créer la colonne 'features'
assembler = VectorAssembler(
    inputCols=ohe_cols,
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

# Entraînement du pipeline sur train_df
model = pipeline.fit(train_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Prédictions sur test_df
predictions = model.transform(test_df)
display(predictions.limit(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator_auc = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

auc = evaluator_auc.evaluate(predictions)
print("AUC =", auc)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# vector_to_array convertit un vecteur MLlib en un array Spark, ce qui permet ensuite d’extraire p1.
from pyspark.ml.functions import vector_to_array

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import col
from pyspark.ml.functions import vector_to_array

# Convertir le vecteur probability en array
preds = predictions.withColumn(
    "prob_array",
    vector_to_array(col("probability"))
)

# Extraire la probabilité de classe 1
preds = preds.select(
    col("label"),
    col("prob_array")[1].alias("p1")
)

display(preds.limit(10))


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

pdf = preds.toPandas()
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(pdf["label"], pdf["p1"])
ks = max(tpr - fpr)

print("KS =", ks)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import col

preds = predictions.select("label", "prediction")

tp = preds.filter((col("label") == 1) & (col("prediction") == 1)).count()
tn = preds.filter((col("label") == 0) & (col("prediction") == 0)).count()
fp = preds.filter((col("label") == 0) & (col("prediction") == 1)).count()
fn = preds.filter((col("label") == 1) & (col("prediction") == 0)).count()

print("TP =", tp)
print("TN =", tn)
print("FP =", fp)
print("FN =", fn)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#Spark stocke probability sous forme de SparseVector, donc on doit d’abord le convertir en array.
from pyspark.sql.functions import col
from pyspark.ml.functions import vector_to_array

# Convertir la probabilité en array et extraire p1
preds = predictions.withColumn(
    "prob_array",
    vector_to_array(col("probability"))
).select(
    col("label").alias("y_true"),
    col("prob_array")[1].alias("p1")
)

# Conversion en pandas pour sklearn
pdf = preds.toPandas()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#Générer les prédictions binaires (seuil 0.5)
import numpy as np

# Prédiction binaire
pdf["y_pred"] = (pdf["p1"] > 0.5).astype(int)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#Matrice de confusion + ROC + AUC (sklearn)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Matrice de confusion
conf_mat = confusion_matrix(pdf["y_true"], pdf["y_pred"])

# ROC
fpr, tpr, thresholds = roc_curve(pdf["y_true"], pdf["p1"])
roc_auc = auc(fpr, tpr)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#Visualisation : Matrice de confusion + Courbe ROC
# Création des subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Matrice de confusion
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='viridis', cbar=False,
            xticklabels=['Négatif', 'Positif'], 
            yticklabels=['Négatif', 'Positif'],
            ax=axes[0])
axes[0].set_title("Matrice de confusion - Test", pad=20)
axes[0].set_xlabel('Valeurs Prédites')
axes[0].set_ylabel('Valeurs Réelles')

# Courbe ROC
axes[1].plot(fpr, tpr, color='blue', lw=2, label='AUC = %0.2f' % roc_auc)
axes[1].plot([0, 1], [0, 1], color='gray', linestyle='--')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('Taux de Faux Positifs')
axes[1].set_ylabel('Taux de Vrais Positifs')
axes[1].set_title('Courbe ROC')
axes[1].legend(loc="lower right")

plt.tight_layout()
plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Rapport de classification
report = classification_report(pdf["y_true"], pdf["y_pred"])
print(report)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df_ml.count(), df_ml.dropDuplicates().count()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
