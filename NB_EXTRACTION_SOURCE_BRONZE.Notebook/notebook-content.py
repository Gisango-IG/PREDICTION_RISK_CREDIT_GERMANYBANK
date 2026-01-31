# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "8226a6f8-e06a-4cd1-aec0-2ae6e014c1c7",
# META       "default_lakehouse_name": "LK_SOURCE_CREDIT_BRONZE",
# META       "default_lakehouse_workspace_id": "76eb933a-950e-4895-8f00-76ccb4a5f37d",
# META       "known_lakehouses": [
# META         {
# META           "id": "8226a6f8-e06a-4cd1-aec0-2ae6e014c1c7"
# META         },
# META         {
# META           "id": "1a6e6347-3612-44f5-9d2d-e84e90137cae"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

# =========================================================
# NOTEBOOK BRONZE — INGESTION + PREMIER NETTOYAGE
# Architecture Médaillon : Bronze → Silver → Gold
# =========================================================

import requests
import pandas as pd
import io
from pyspark.sql.functions import col, trim, when, isnan

# ---------------------------------------------------------
# 1. Télécharger le dataset German Credit
# ---------------------------------------------------------

url = "https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv"
response = requests.get(url)

pdf = pd.read_csv(io.BytesIO(response.content))
df_raw = spark.createDataFrame(pdf)

print("Aperçu des données brutes :")
#df_raw.show(5)
#df_raw.printSchema()
display(df_raw)

# ---------------------------------------------------------
# 2. Nettoyage léger (Bronze)
# ---------------------------------------------------------

df_clean = df_raw

# Trim sur les colonnes string
for c, t in df_clean.dtypes:
    if t == "string":
        df_clean = df_clean.withColumn(c, trim(col(c)))

# Remplacement simple des nulls dans les colonnes string
for c, t in df_clean.dtypes:
    if t == "string":
        df_clean = df_clean.fillna({c: "Unknown"})

# ---------------------------------------------------------
# 3. Écriture dans le Lakehouse Bronze (ABFS)
# ---------------------------------------------------------

bronze_path = "abfss://76eb933a-950e-4895-8f00-76ccb4a5f37d@onelake.dfs.fabric.microsoft.com/8226a6f8-e06a-4cd1-aec0-2ae6e014c1c7/Tables/bronze_credit"

df_clean.write.format("delta").mode("overwrite").save(bronze_path)

print("Table Bronze écrite avec succès dans le Lakehouse Bronze via ABFS.")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
