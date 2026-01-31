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

# CELL ********************

# 1) Récupérer le modèle de régression logistique à l'intérieur du pipeline entraîné
lr_model = model.stages[-1]

# 2) Récupérer le vecteur de coefficients et l'intercept
coeffs = lr_model.coefficients
intercept = lr_model.intercept

print("Nombre de coefficients :", len(coeffs))
print("Intercept :", intercept)
print("Extrait des premiers coefficients :", coeffs[:10])


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#récupérer les noms des features
# Récupérer l'assembler (juste avant la régression)
assembler = model.stages[-2]

# Les noms des colonnes assemblées dans "features"
feature_names = assembler.getInputCols()

print("Nombre de features :", len(feature_names))
feature_names[:20]  # afficher les 20 premières


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# rveconstruction des noms des colonnes OHE
from pyspark.ml.feature import OneHotEncoderModel

# Récupérer le modèle OHE dans le pipeline
ohe_model = model.stages[-3]

# Récupérer les colonnes d'entrée et de sortie du OHE
input_cols = ohe_model.getInputCols()
output_cols = ohe_model.getOutputCols()

print("Colonnes OHE d'entrée :", input_cols)
print("Colonnes OHE de sortie :", output_cols)



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

ohe_model = model.stages[-3]
input_cols = ohe_model.getInputCols()
output_cols = ohe_model.getOutputCols()

print("Colonnes OHE d'entrée :", input_cols)
print("Colonnes OHE de sortie :", output_cols)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Récupérer le modèle OHE
ohe_model = model.stages[-3]

# Récupérer le nombre de catégories pour chaque variable
category_sizes = ohe_model.categorySizes

# Afficher proprement
for col, size in zip(ohe_model.getInputCols(), category_sizes):
    print(f"{col} : {size} catégories → {size - 1} colonnes OHE")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Récupérer le modèle OHE
ohe_model = model.stages[-3]

# Colonnes d'entrée et de sortie du OHE
input_cols = ohe_model.getInputCols()
output_cols = ohe_model.getOutputCols()
category_sizes = ohe_model.categorySizes

# Liste finale des features
final_feature_names = []

# Pour chaque variable OHE
for out_col, size in zip(output_cols, category_sizes):
    # Spark crée (size - 1) colonnes OHE
    for i in range(size - 1):
        final_feature_names.append(f"{out_col}_{i}")

# Vérification
print("Nombre total de features reconstruits :", len(final_feature_names))
final_feature_names[:20]  # afficher les 20 premiers


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import pandas as pd

# Convertir les coefficients Spark en array numpy
coef_values = lr_model.coefficients.toArray()

# Construire un tableau propre
df_coef = pd.DataFrame({
    "feature": final_feature_names,
    "coefficient": coef_values
})

# Importance = valeur absolue du coefficient
df_coef["importance"] = df_coef["coefficient"].abs()

# Trier par importance décroissante
df_coef = df_coef.sort_values(by="importance", ascending=False)

df_coef.head(20)  # afficher les 20 plus importants


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Récupérer le modèle StringIndexer (il est avant le OHE dans le pipeline)
indexer_model = model.stages[-4]

# Récupérer les labels (catégories réelles)
indexer_labels = indexer_model.labels

print("Catégories réelles pour status_idx :", indexer_labels)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

indexer_model = model.stages[-4]
indexer_model.labels


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

for i, stage in enumerate(model.stages):
    print(i, type(stage))


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

for i, stage in enumerate(model.stages):
    if "StringIndexerModel" in str(type(stage)):
        print(f"Stage {i} — inputCol = {stage.getInputCol()}, outputCol = {stage.getOutputCol()}")
        print("  labels =", stage.labels)
        print()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

for i, stage in enumerate(model.stages):
    if "StringIndexerModel" in str(type(stage)):
        print(f"Stage {i} — inputCol = {stage.getInputCol()}, outputCol = {stage.getOutputCol()}")
        print("  labels =", stage.labels)
        print()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

decoded_features = []

# On parcourt tous les StringIndexerModel
for stage in model.stages:
    if "StringIndexerModel" in str(type(stage)):
        input_col = stage.getInputCol()
        output_col = stage.getOutputCol()
        labels = stage.labels
        
        # Nombre de colonnes OHE = nb catégories - 1
        for i in range(len(labels) - 1):
            feature_name = f"{output_col.replace('_idx','_ohe')}_{i}"
            category_label = labels[i + 1]  # labels[0] = catégorie de référence
            decoded_features.append((feature_name, input_col, category_label))

import pandas as pd
df_decoded = pd.DataFrame(decoded_features, columns=["feature", "variable", "category"])
df_decoded.head(20)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Convertir les coefficients en DataFrame
df_coef = pd.DataFrame({
    "feature": final_feature_names,
    "coefficient": lr_model.coefficients.toArray()
})

# Fusionner avec les noms métier
df_full = df_coef.merge(df_decoded, on="feature", how="left")

# Importance = valeur absolue du coefficient
df_full["importance"] = df_full["coefficient"].abs()

# Trier par importance décroissante
df_full = df_full.sort_values(by="importance", ascending=False)

df_full.head(20)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Les features réels sont ceux de df_decoded
real_feature_names = df_decoded["feature"].tolist()

# Vérification : doit être égal à la longueur des coefficients
print(len(real_feature_names), len(lr_model.coefficients))


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df_coef = pd.DataFrame({
    "feature": real_feature_names,
    "coefficient": lr_model.coefficients.toArray()
})

df_full = df_coef.merge(df_decoded, on="feature", how="left")

df_full["importance"] = df_full["coefficient"].abs()

# Réordonner les colonnes
df_full = df_full[["variable", "category", "feature", "coefficient", "importance"]]

df_full.sort_values(by="importance", ascending=False).head(20)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

print("Nombre de coefficients :", len(lr_model.coefficients))
print("Nombre de features dans df_decoded :", len(df_decoded))
print("Nombre de features dans final_feature_names :", len(final_feature_names))


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

assembler = model.stages[20]  # VectorAssembler
assembler.getInputCols()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

assembler = model.stages[20]  # VectorAssembler
assembler.getInputCols()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df_coef = pd.DataFrame({
    "feature": real_feature_names,
    "coefficient": lr_model.coefficients.toArray()
})

df_full = df_coef.merge(df_decoded, on="feature", how="left")

df_full["importance"] = df_full["coefficient"].abs()

df_full = df_full[["variable", "category", "feature", "coefficient", "importance"]]

df_full.sort_values(by="importance", ascending=False).head(20)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
