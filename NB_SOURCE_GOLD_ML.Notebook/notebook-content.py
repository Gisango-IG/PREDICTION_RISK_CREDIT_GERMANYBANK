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

# Prédictions sur test_df(it adds columns like rawPrediction,probability,prediction etc..)
predictions = model.transform(test_df)
display(predictions.limit(3))

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

#Calculer la matrice de confusion
from pyspark.sql import functions as F

confusion = predictions.groupBy("label", "prediction").count()
confusion.show()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

predictions.select("probability").show(5, truncate=False)




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

# Récupérer le modèle de régression logistique
lr_model = model.stages[-1]

# Coefficients
coeffs = lr_model.coefficients.toArray()

# Intercept
intercept = lr_model.intercept

print("Intercept :", intercept)
print("Nombre de coefficients :", len(coeffs))


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

train_df.groupBy("label").count().show()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

assembler = None
for stage in model.stages:
    if stage.__class__.__name__ == "VectorAssembler":
        assembler = stage
        break

feature_names = assembler.getInputCols()

print("Nombre de features :", len(feature_names))
print(feature_names)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Récupérer tous les StringIndexerModel du pipeline
indexer_models = [
    stage for stage in model.stages
    if stage.__class__.__name__ == "StringIndexerModel"
]

# Extraire les catégories pour chaque variable
indexer_info = {}
for idx_model in indexer_models:
    input_col = idx_model.getInputCol()
    labels = idx_model.labels  # catégories dans l'ordre Spark
    indexer_info[input_col] = labels

indexer_info


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.ml.feature import OneHotEncoderModel

# 1) Récupérer le OneHotEncoderModel
encoder_model = None
for stage in model.stages:
    if stage.__class__.__name__ == "OneHotEncoderModel":
        encoder_model = stage
        break

# 2) Récupérer les colonnes OHE dans l'ordre
ohe_input_cols = encoder_model.getInputCols()
ohe_output_cols = encoder_model.getOutputCols()

# 3) Reconstruire les noms des colonnes OHE
expanded_feature_names = []

for input_col, output_col in zip(ohe_input_cols, ohe_output_cols):
    # Récupérer les catégories du StringIndexer correspondant
    original_col = input_col.replace("_idx", "")
    categories = indexer_info[original_col]
    
    # dropLast=True → on enlève la dernière catégorie
    kept_categories = categories[:-1]
    
    # Construire les noms des colonnes OHE
    for cat in kept_categories:
        expanded_feature_names.append(f"{original_col}={cat}")

# 4) Vérifier le nombre total
print("Nombre total de colonnes OHE reconstruites :", len(expanded_feature_names))
expanded_feature_names


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

feature_names


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 1) Récupérer le VectorAssembler
assembler = None
for stage in model.stages:
    if stage.__class__.__name__ == "VectorAssembler":
        assembler = stage
        break

assembler_inputs = assembler.getInputCols()

# 2) Colonnes OHE (déjà connues)
ohe_cols = feature_names  # ta liste de 19 blocs OHE

# 3) Colonnes numériques = colonnes du VectorAssembler qui ne sont pas des OHE
numeric_features = [col for col in assembler_inputs if col not in ohe_cols]

print("Colonnes numériques :", numeric_features)
print("Nombre :", len(numeric_features))


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

for stage in model.stages:
    print(stage.__class__.__name__, " → ", stage)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

indexer_info['people_liable_cat']


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

encoder_model = None
for stage in model.stages:
    if stage.__class__.__name__ == "OneHotEncoderModel":
        encoder_model = stage
        break

encoder_model.categorySizes


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 1) Récupérer le OneHotEncoderModel
encoder_model = None
for stage in model.stages:
    if stage.__class__.__name__ == "OneHotEncoderModel":
        encoder_model = stage
        break

# 2) Récupérer les tailles OHE finales (dropLast=True)
sizes = [s - 1 for s in encoder_model.categorySizes]

# 3) Récupérer les colonnes d'entrée du OneHotEncoder
ohe_input_cols = encoder_model.getInputCols()

# 4) Reconstruire les 77 noms de features
full_feature_list = []

for input_col, size in zip(ohe_input_cols, sizes):
    original_col = input_col.replace("_idx", "")
    categories = indexer_info[original_col]
    kept_categories = categories[:-1]  # dropLast=True

    # Vérification de cohérence
    if len(kept_categories) != size:
        print(f"⚠️ Incohérence détectée pour {original_col}: "
              f"{len(kept_categories)} catégories vs {size} attendu")

    for cat in kept_categories:
        full_feature_list.append(f"{original_col}={cat}")

# 5) Vérification finale
print("Nombre total de features :", len(full_feature_list))
full_feature_list


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# 1) Récupérer les coefficients du modèle
coeffs = model.stages[-1].coefficients.toArray()

print("Taille du vecteur de coefficients :", len(coeffs))

# 2) Extraire les 58 premiers coefficients (les seuls réellement utilisés)
real_coeffs = coeffs[:len(full_feature_list)]

print("Taille des coefficients réels :", len(real_coeffs))

# 3) Construire la table feature ↔ coefficient
feature_coef_pairs = list(zip(full_feature_list, real_coeffs))

# 4) Afficher les 10 premières lignes pour vérification
feature_coef_pairs[:10]


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Trier les coefficients par importance absolue
sorted_importances = sorted(
    feature_coef_pairs,
    key=lambda x: abs(x[1]),
    reverse=True
)

# Afficher les 15 plus importants
sorted_importances[:15]


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import pandas as pd

# Construire un DataFrame propre
df_top15 = pd.DataFrame(sorted_importances[:15], columns=["feature", "coefficient"])

# Ajouter une colonne importance absolue
df_top15["abs_importance"] = df_top15["coefficient"].abs()

# Ajouter un rang
df_top15.insert(0, "rank", range(1, len(df_top15) + 1))

# Afficher le tableau
df_top15


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import pandas as pd

# Construire le DataFrame complet des 58 features + coefficients
df_all = pd.DataFrame(feature_coef_pairs, columns=["feature", "coefficient"])

# Ajouter l’importance absolue
df_all["abs_importance"] = df_all["coefficient"].abs()

# Ajouter un rang basé sur l’importance
df_all = df_all.sort_values("abs_importance", ascending=False).reset_index(drop=True)
df_all.insert(0, "rank", range(1, len(df_all) + 1))

# Afficher le tableau complet
df_all


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

import pandas as pd
import numpy as np

# Construire le DataFrame complet
df_audit = pd.DataFrame(feature_coef_pairs, columns=["feature", "coefficient"])

# Importance absolue
df_audit["abs_importance"] = df_audit["coefficient"].abs()

# Signe
df_audit["sign"] = np.where(df_audit["coefficient"] > 0, "+", "-")

# Interprétation automatique
df_audit["risk_effect"] = np.where(
    df_audit["coefficient"] > 0,
    "Augmente le risque",
    "Diminue le risque"
)

# Tri par importance
df_audit = df_audit.sort_values("abs_importance", ascending=False).reset_index(drop=True)
df_audit.insert(0, "rank", range(1, len(df_audit) + 1))

df_audit


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import matplotlib.pyplot as plt

# On prend les 20 plus importantes pour la lisibilité
df_plot = df_audit.head(20).sort_values("abs_importance")

plt.figure(figsize=(10, 12))

colors = df_plot["coefficient"].apply(lambda x: "red" if x > 0 else "blue")

plt.barh(df_plot["feature"], df_plot["coefficient"], color=colors)
plt.xlabel("Coefficient (impact sur le log-odds)")
plt.title("Top 20 des variables les plus influentes (modèle logistique)")
plt.grid(axis="x", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

example = spark.createDataFrame([
    {
        "status": "no checking account",
        "credit_history": "existing credits paid back duly till now",
        "purpose": "car (new)",
        "savings": "... < 100 DM",
        "employment_duration": "1 <= ... < 4 years",
        "personal_status_sex": "male : single",
        "other_debtors": "none",
        "property": "car or other",
        "other_installment_plans": "none",
        "housing": "own",
        "job": "skilled employee/official",
        "telephone": "no",
        "foreign_worker": "yes",
        "duration_cat": "13-24",
        "amount_cat": "1001-2000",
        "present_residence_cat": "2-3",
        "age_cat": "25-35",
        "number_credits_cat": "0-1",
        "people_liable_cat": "1"
    }
])


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

result = model.transform(example)
result.select("probability", "prediction").show(truncate=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

p_default = result.collect()[0]["probability"][1]
p_default


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#sauvegarde du model dans le lakehouse Gold et mise en commentaire des commandes pour empecher la maj du model
#model.write() \
#    .overwrite() \
#   .save("abfss://76eb933a-950e-4895-8f00-76ccb4a5f37d@onelake.dfs.fabric.microsoft.com/3ee4579c-22ab-494d-8b27-fbf8a3b342fb/Files/CREDIT_SCORING_MODEL")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

payload = {
    "status": "no checking account",
    "credit_history": "existing credits paid back duly till now",
    "purpose": "car (new)",
    "savings": "... < 100 DM",
    "employment_duration": "1 <= ... < 4 years",
    "personal_status_sex": "male : single",
    "other_debtors": "none",
    "property": "car or other",
    "other_installment_plans": "none",
    "housing": "own",
    "job": "skilled employee/official",
    "telephone": "no",
    "foreign_worker": "yes",
    "duration_cat": "13-24",
    "amount_cat": "1001-2000",
    "present_residence_cat": "2-3",
    "age_cat": "25-35",
    "number_credits_cat": "0-1",
    "people_liable_cat": "1"
}



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df_request = spark.createDataFrame([payload])


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.ml import PipelineModel

loaded_model = PipelineModel.load(
    "abfss://76eb933a-950e-4895-8f00-76ccb4a5f37d@onelake.dfs.fabric.microsoft.com/3ee4579c-22ab-494d-8b27-fbf8a3b342fb/Files/CREDIT_SCORING_MODEL"
)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

result = loaded_model.transform(df_request)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

row = result.select("probability", "prediction").collect()[0]

p_default = float(row["probability"][1])
prediction = int(row["prediction"])
decision = "ACCEPTE" if prediction == 0 else "REFUSE"

{
    "probability_default": p_default,
    "prediction": prediction,
    "decision": decision
}


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.ml import PipelineModel

loaded_model = PipelineModel.load(
    "abfss://76eb933a-950e-4895-8f00-76ccb4a5f37d@onelake.dfs.fabric.microsoft.com/3ee4579c-22ab-494d-8b27-fbf8a3b342fb/Files/CREDIT_SCORING_MODEL"
)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
