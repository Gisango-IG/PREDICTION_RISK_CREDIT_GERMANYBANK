# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "1a6e6347-3612-44f5-9d2d-e84e90137cae",
# META       "default_lakehouse_name": "LK_SOURCE_CREDIT_SILVER",
# META       "default_lakehouse_workspace_id": "76eb933a-950e-4895-8f00-76ccb4a5f37d",
# META       "known_lakehouses": [
# META         {
# META           "id": "1a6e6347-3612-44f5-9d2d-e84e90137cae"
# META         },
# META         {
# META           "id": "8226a6f8-e06a-4cd1-aec0-2ae6e014c1c7"
# META         },
# META         {
# META           "id": "3ee4579c-22ab-494d-8b27-fbf8a3b342fb"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

# =========================================================
# NOTEBOOK SILVER — ANALYSE EXPLORATOIRE + PRÉPARATION ML
# =========================================================

# Spark functions pour manipuler les colonnes
from pyspark.sql.functions import (
    col, trim, when, isnan, count, desc, avg
)

# Pour les statistiques et les transformations futures
from pyspark.sql import DataFrame

# Pour afficher proprement
import pandas as pd


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# LECTURE DE LA TABLE BRONZE
bronze_path = "abfss://76eb933a-950e-4895-8f00-76ccb4a5f37d@onelake.dfs.fabric.microsoft.com/8226a6f8-e06a-4cd1-aec0-2ae6e014c1c7/Tables/bronze_credit"

df_bronze = spark.read.format("delta").load(bronze_path)

print("Aperçu des données Bronze :")
df_bronze.limit(5).toPandas()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import col, when, count, isnan

# check if there is null field(on vérifie si nous avons des valeur null)
missing_counts = (
    df_bronze.select([
        count(when(col(c).isNull() | isnan(c), c)).alias(c)
        for c in df_bronze.columns
    ])
    .toPandas()
    .T
    .rename(columns={0: "missing_count"})
)

missing_counts


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Identification des variables catégorielles et numériques
categorical_cols = [c for c, t in df_bronze.dtypes if t == "string" and c != "Class"]
numeric_cols = [c for c, t in df_bronze.dtypes if t != "string" and c != "Class"]

print("Variables catégorielles :", categorical_cols)
print("Variables numériques :", numeric_cols)



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Analyse exploratoire ==> analyse univariée en forme de tableaux
from pyspark.sql.functions import col, round, concat, lit

total_rows = df_bronze.count()

for variable in categorical_cols:
    print(f"Distribution de {variable}")
    
    df_univar = (
        df_bronze
        .groupBy(variable)
        .count()
        .withColumn("percentage",
                    concat(
                        round((col("count") / total_rows) * 100, 1),
                        lit("%")
                    )
        )
        .orderBy("count", ascending=False)
    )
    
    display(df_univar)



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# je check le schemas pour voir le nom de la variable cible ==> credit_risk
df_bronze.printSchema()



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import col, round, lit, concat

target_col = "credit_risk"

for variable in categorical_cols:
    print(f"=== Relation entre {variable} et {target_col} ===")
    
    # Tableau croisé (pivot)
    df_pivot = (
        df_bronze
        .groupBy(variable)
        .pivot(target_col)   # crée les colonnes 0 et 1
        .count()
        .fillna(0)
    )
    
    # Calcul du total par ligne
    df_pivot = df_pivot.withColumn("total", col("0") + col("1"))
    
    # Pourcentages formatés
    df_pivot = (
        df_pivot
        .withColumn("0_pct", concat(round((col("0") / col("total")) * 100, 1), lit("%")))
        .withColumn("1_pct", concat(round((col("1") / col("total")) * 100, 1), lit("%")))
        .orderBy(variable)
    )
    
    # Colonnes finales dans l'ordre souhaité
    df_final = df_pivot.select(variable, "0", "1", "0_pct", "1_pct")
    
    display(df_final)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# transformation of numerical variable to categorial variable
from pyspark.sql.functions import col, when

df_silver = df_bronze

# -------------------------
# duration
# -------------------------
df_silver = df_silver.withColumn(
    "duration_cat",
    when(col("duration") <= 12, "<=12") \
    .when((col("duration") >= 13) & (col("duration") <= 24), "13-24") \
    .when((col("duration") >= 25) & (col("duration") <= 36), "25-36") \
    .when((col("duration") >= 37) & (col("duration") <= 48), "37-48") \
    .otherwise(">48")
)

# -------------------------
# amount
# -------------------------
df_silver = df_silver.withColumn(
    "amount_cat",
    when(col("amount") <= 1000, "<=1000") \
    .when((col("amount") >= 1001) & (col("amount") <= 2000), "1001-2000") \
    .when((col("amount") >= 2001) & (col("amount") <= 5000), "2001-5000") \
    .when((col("amount") >= 5001) & (col("amount") <= 10000), "5001-10000") \
    .otherwise(">10000")
)

# -------------------------
# present_residence
# -------------------------
df_silver = df_silver.withColumn(
    "present_residence_cat",
    when(col("present_residence") <= 1, "<=1") \
    .when((col("present_residence") >= 2) & (col("present_residence") <= 3), "2-3") \
    .otherwise(">=4")
)

# -------------------------
# age
# -------------------------
df_silver = df_silver.withColumn(
    "age_cat",
    when(col("age") < 25, "<25") \
    .when((col("age") >= 25) & (col("age") <= 35), "25-35") \
    .when((col("age") >= 36) & (col("age") <= 45), "36-45") \
    .when((col("age") >= 46) & (col("age") <= 60), "46-60") \
    .otherwise(">60")
)

# -------------------------
# number_credits
# -------------------------
df_silver = df_silver.withColumn(
    "number_credits_cat",
    when(col("number_credits") <= 1, "0-1") \
    .when(col("number_credits") == 2, "2") \
    .otherwise(">=3")
)

# -------------------------
# people_liable
# -------------------------
df_silver = df_silver.withColumn(
    "people_liable_cat",
    when(col("people_liable") == 0, "0") \
    .when(col("people_liable") == 1, "1") \
    .otherwise(">=2")
)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# keeping numerical variable for analysis in Power BI
numeric_for_reporting = ["amount"]

# New categ columns created
categorical_binned = [
    "duration_cat",
    "amount_cat",
    "present_residence_cat",
    "age_cat",
    "number_credits_cat",
    "people_liable_cat"
]

# native categ columns
categorical_original = [
    col for col in df_bronze.columns
    if col not in ["duration", "amount", "present_residence", "age", "number_credits", "people_liable", "credit_risk"]
    and df_bronze.schema[col].dataType.simpleString() == "string"
]

# Target variable
target = ["credit_risk"]

# Construction of dataframe Silver
df_silver_powerbi = df_silver.select(
    *(categorical_binned + numeric_for_reporting + categorical_original + target)
)

display(df_silver_powerbi.limit(10))



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# creation of gold tables in  lakehouse Gold for reporting in PowerBi via Semantic Model and ML

# the Path for reporting
gold_credit_scoring_path="abfss://76eb933a-950e-4895-8f00-76ccb4a5f37d@onelake.dfs.fabric.microsoft.com/3ee4579c-22ab-494d-8b27-fbf8a3b342fb/Tables/dbo/credit_scoring"

# Saving the Delta table into lakehouse
df_silver_powerbi.write.format("delta").mode("overwrite").save(gold_credit_scoring_path)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Native numerical column to remove in order to create the final dataframe for ML(regression logistic)
cols_to_drop = [
    "duration",
    "amount",
    "present_residence",
    "age",
    "number_credits",
    "people_liable"
]

# creation of final dataframe 
df_model = df_silver.drop(*cols_to_drop)

display(df_model.limit(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Saving the Delta table into lakehouse for ML purpose

# the path for ML

gold_credit_scoring_ml_path="abfss://76eb933a-950e-4895-8f00-76ccb4a5f37d@onelake.dfs.fabric.microsoft.com/3ee4579c-22ab-494d-8b27-fbf8a3b342fb/Tables/dbo/credit_scoring_ml"
df_model.write.format("delta").mode("overwrite").save(gold_credit_scoring_ml_path)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
