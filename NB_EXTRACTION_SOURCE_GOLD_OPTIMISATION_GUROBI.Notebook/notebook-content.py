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
# META           "id": "3ee4579c-22ab-494d-8b27-fbf8a3b342fb"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

# installation du gurobi au debut pour que quand je vais lancer le reste qu'il n y ait pas de casse 
# car le PySpark kernel sera redemmaré à l'issue de l'installation du moteur Gurobi
%pip install gurobipy

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Loading dataset scoring_credit from Lakehouse Gold used to build report in powerbi
#German Credit Dataset.

from pyspark.sql.functions import col, lit, when

df_silver = spark.read.format("delta").load(
    "abfss://76eb933a-950e-4895-8f00-76ccb4a5f37d@onelake.dfs.fabric.microsoft.com/3ee4579c-22ab-494d-8b27-fbf8a3b342fb/Tables/dbo/credit_scoring"
    
)

display(df_silver.limit(3))


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import col, lit, when

# -----------------------------
# 1) PD basé sur credit_history
# -----------------------------
df_silver = df_silver.withColumn(
    "PD_history",
    when(col("credit_history") == "no credits taken/ all credits paid back duly", lit(0.03))
    .when(col("credit_history") == "all credits at this bank paid back duly", lit(0.05))
    .when(col("credit_history") == "existing credits paid back duly till now", lit(0.10))
    .when(col("credit_history") == "delay in paying off in the past", lit(0.25))
    .when(col("credit_history") == "critical account/ other credits existing (not at this bank)", lit(0.45))
    .otherwise(lit(0.20))  # fallback
)

# PD final = max(PD_history, 0.60) si credit_risk == True
df_silver = df_silver.withColumn(
    "PD",
    when(col("credit_risk") == True, lit(0.60))
    .otherwise(col("PD_history"))
)

# -----------------------------
# 2) LGD basé sur purpose
# -----------------------------
df_silver = df_silver.withColumn(
    "LGD",
    when(col("purpose") == "car (new)", lit(0.25))
    .when(col("purpose") == "car (used)", lit(0.30))
    .when(col("purpose") == "furniture/equipment", lit(0.35))
    .when(col("purpose") == "retraining", lit(0.35))
    .when(col("purpose") == "radio/television", lit(0.40))
    .when(col("purpose") == "repairs", lit(0.40))
    .when(col("purpose") == "education", lit(0.30))
    .when(col("purpose") == "domestic appliances", lit(0.45))
    .when(col("purpose") == "business", lit(0.60))  # ta modification
    .otherwise(lit(0.45))  # others
)

# -----------------------------
# 3) Profit margin basé sur risque
# -----------------------------
df_silver = df_silver.withColumn(
    "profit_margin",
    when(col("PD") <= 0.05, lit(0.08))   # très bon historique
    .when(col("PD") <= 0.10, lit(0.10))  # bon
    .when(col("PD") <= 0.25, lit(0.12))  # moyen
    .when(col("PD") <= 0.45, lit(0.18))  # risqué
    .otherwise(lit(0.25))                # très risqué
)

# -----------------------------
# 4) Colonnes financières
# -----------------------------

# Montant du crédit
df_silver = df_silver.withColumn("credit_amount", col("amount"))

# Profit attendu
df_silver = df_silver.withColumn(
    "expected_profit",
    col("credit_amount") * col("profit_margin")
)

# Perte attendue
df_silver = df_silver.withColumn(
    "expected_loss",
    col("credit_amount") * col("PD") * col("LGD")
)

# Valeur attendue (profit net du risque)
df_silver = df_silver.withColumn(
    "expected_value",
    col("expected_profit") - col("expected_loss")
)

# Exposition au risque
df_silver = df_silver.withColumn(
    "exposure_at_risk",
    col("credit_amount") * col("PD")    
)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(
    df_silver.select(
        "credit_history", "purpose", "credit_risk",
        "PD", "LGD", "profit_margin",
        "expected_profit", "expected_loss", "expected_value"
    ).limit(20)
)



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Add unique id column very important for gurobi solver
from pyspark.sql.functions import monotonically_increasing_id

df_silver = df_silver.withColumn("id", monotonically_increasing_id())


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Sélection des colonnes utiles pour Gurobi
df_opt = df_silver.select(
    "id",                # identifiant unique du crédit
    "credit_amount",
    "expected_profit",
    "expected_loss",
    "expected_value",
    "exposure_at_risk",
    "PD",
    "LGD",
    "profit_margin"
)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(df_opt.limit(10))


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# convert to pandas for case of gurobi
pdf_opt = df_opt.toPandas()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

pdf_opt.shape

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# we import gurobi models
from gurobipy import Model, GRB

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#creation of gurobi model
m = Model("credit_portfolio_optimization")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Étape suivante : créer les variables binaires
# On ajoute une variable binaire x[i] pour chaque crédit :
# 
# x[i] = 1 → le crédit est sélectionné
# 
# x[i] = 0 → le crédit est exclu

# CELL ********************

x = m.addVars(
    pdf_opt.index,      # un index par ligne du dataframe
    vtype=GRB.BINARY,   # variable binaire
    name="x"
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Étape suivante: définir la fonction objectif
# Ici On dit à Gurobi : maximise la somme des expected_value des crédits sélectionnés.
# 
# Gurobi va chercher la combinaison de crédits qui maximise la valeur attendue totale.
# 
# <mark>Rien d’autre n’est ajouté pour l’instant.
# 
# Pas encore de contraintes.</mark>

# CELL ********************

m.setObjective(
    sum(x[i] * pdf_opt.loc[i, "expected_value"] for i in pdf_opt.index),
    GRB.MAXIMIZE
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Étape suivante : Ajouter la contrainte de risque
# On impose à Gurobi de ne pas dépasser un niveau de risque global.
# Pour rester simple et réaliste, on fixe un seuil égal à 30 % du risque total actuel.
# 
# risk_threshold = limite maximale de risque autorisée
# 
# la contrainte empêche Gurobi de sélectionner trop de crédits risqués

# CELL ********************

risk_threshold = 0.30 * pdf_opt["exposure_at_risk"].sum()

m.addConstr(
    sum(x[i] * pdf_opt.loc[i, "exposure_at_risk"] for i in pdf_opt.index)
    <= risk_threshold,
    name="risk_constraint"
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# <mark><gurobi.Constr *Awaiting Model Update*></mark>
# est exactement ce qu’on veut voir à cette étape.
# Ça signifie simplement :
# 
# * La contrainte a été ajoutée au modèle, mais le modèle n’a pas encore été optimisé.  
# * Aucune erreur.  
# * Tout est propre.

# MARKDOWN ********************

# Étape suivante : lancer l’optimisation
# 
# Gurobi va résoudre le modèle
# * Il va choisir les x[i] = 0/1 optimaux
# * Il va respecter la contrainte de risque
# * Il va maximiser la valeur attendue

# CELL ********************

m.optimize()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# <mark>En clair, ce que Gurobi dit ci-dessus </mark>:
# 
# Modèle :
# 
#     1000 variables binaires (1000 crédits possibles)
# 
#     1 contrainte (ta contrainte de risque globale)
# 
# Il a trouvé une solution optimale : “Optimal solution found” → le problème est bien posé, bien résolu
# 
# Objectif optimal ≈ 110 207 → c’est la valeur attendue totale du portefeuille optimisé
# 
# Plusieurs solutions testées, la meilleure a été gardée
# 
# Gap 0.0003 % → la solution est pratiquement parfaite
# 
# Donc :
# * Le modèle est sain
# * La contrainte de risque est active
# * L’optimisation a bien fonctionné

# CELL ********************

selected_ids = [pdf_opt.loc[i, "id"] for i in pdf_opt.index if x[i].X > 0.5]

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# On a parcouru toutes les variables x[i]
# 
# On a gardé les id des crédits pour lesquels Gurobi a mis x[i] = 1

# MARKDOWN ********************

# Prochaine étape : 
# On va filtrer le DataFrame Pandas pour ne garder que les crédits où x[i] = 1.
# 
# le dataframe selected_df devient le portefeuille optimisé.

# CELL ********************

selected_df = pdf_opt[pdf_opt["id"].isin(selected_ids)]

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

total_expected_value = selected_df["expected_value"].sum()
total_risk = selected_df["exposure_at_risk"].sum()
total_amount = selected_df["credit_amount"].sum()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

print("Valeur attendue totale optimisée :", total_expected_value)
print("Risque total consommé :", total_risk)
print("Montant total sélectionné :", total_amount)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# <u>**Interprétation métier des résultats**</u>
# 
#  1.** Valeur attendue totale optimisée : 110 207 €**
# 
# C’est le profit net attendu du portefeuille sélectionné par Gurobi.
# 
# En clair :
# 
# Si on finance exactement les crédits choisis par le modèle,en tenant compte du risque de défaut (PD),et de la perte en cas de défaut (LGD),alors le portefeuille devrait générer 110k € de profit net sur l’horizon considéré. C’est le “expected return” après ajustement du risque. C’est un indicateur clé :plus il est élevé, plus le portefeuille est rentable pour un niveau de risque donné.
# 
# 2.** Risque total consommé : 425 327 €**
# 
# C’est la somme :∑𝑖𝑥𝑖⋅(𝑃𝐷𝑖×𝐿𝐺𝐷𝑖×𝑐𝑟𝑒𝑑𝑖𝑡_𝑎𝑚𝑜𝑢𝑛𝑡𝑖) Autrement dit : le montant total que on  risque de perdre en moyenne si les défauts se réalisent selon les probabilités estimées.
# 
# Ce chiffre doit être comparé au seuil de risque (30 % du portefeuille initial dans notre modèle).
# Ici, Gurobi a utilisé presque tout le risque autorisé, ce qui est normal : un optimiseur exploite toujours la contrainte active jusqu’à la limite.
# 
# C’est exactement ce que est recommandé pour une bonne gestion de risque :
# <mark>maximiser le rendement tout en consommant le budget de risque disponible</mark>.
# 
# 3. **Montant total sélectionné : 1 577 583 €**
# 
# C’est le montant total des crédits que Gurobi a décidé de financer.Ce chiffre est important car il te dit :combien de capital est immobilisé,quelle taille de portefeuille on obtient après optimisation,si on respecte ou non un éventuel budget (ici on n’avait pas mis de contrainte de budget).
# 
# <mark>==> Le modèle a choisi 1,58 M€ de crédits parmi les 1000 disponibles.</mark>
# 
# Ce montant n’est pas optimisé en soi :
# il est une conséquence de la maximisation de la valeur attendue sous contrainte de risque.
# 
# En langage métier :    
# 
# <u><mark>“Le modèle a construit un portefeuille de 1,58 M€ qui maximise la valeur attendue (110k €) tout en respectant strictement le budget de risque (425k €).”</mark></u>


# CELL ********************

pd_mean = selected_df["PD"].mean()
lgd_mean = selected_df["LGD"].mean()
profit_mean = selected_df["expected_value"].mean()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

print(pd_mean)
print(lgd_mean)
print(profit_mean)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

comparison = {
    "PD_initial": pdf_opt["PD"].mean(),
    "PD_optim": selected_df["PD"].mean(),
    "LGD_initial": pdf_opt["LGD"].mean(),
    "LGD_optim": selected_df["LGD"].mean(),
    "EV_initial": pdf_opt["expected_value"].mean(),
    "EV_optim": selected_df["expected_value"].mean()
}


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

for k, v in comparison.items():
    print(k, ":", v)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Interprétation métier des résultats avant / après optimisation:**
# 
# 1. Indicateur	               Avant optimisation	Après optimisation
# 2. PD moyen	                        0.4603	                    0.3021
# 3. LGD moyen	                    0.36995	                    0.31815
# 4. Expected Value moyen	            131.40 €	                245.45 €
# 
# 
# 
# 🟦 1) PD_initial = 0.46 → PD_optim = 0.30
# 
# Le portefeuille optimisé est beaucoup moins risqué. Avant optimisation, le portefeuille moyen avait 46 % de probabilité de défaut. Après optimisation, on tombe à 30 %. C’est une réduction massive du risque de crédit.
# 
# <mark>En langage métier : Le modèle a éliminé une grande partie des crédits les plus risqués, tout en respectant la contrainte de risque globale.
# </mark>
# 
# Le modèle vient de transformer un portefeuille instable en portefeuille solide.
# 
# 🟧 2) LGD_initial = 0.37 → LGD_optim = 0.32
# 
# Le portefeuille optimisé perd moins en cas de défaut
# LGD = perte en cas de défaut.
# 
# Avant optimisation : 37 % de perte moyenne. Après optimisation : 32 %
# 
# Ça veut dire : On finance des crédits mieux garantis,on sélectionnes des clients avec de meilleures sûretés et on réduis la perte potentielle en cas de défaut
# 
# 🟩 3) EV_initial = 131 € → EV_optim = 245 €
# 
# Le portefeuille optimisé est presque 2 fois plus rentable. <u>**<mark>C’est le point le plus spectaculaire.
# </mark>**</u>
# 
# Avant optimisation : chaque crédit rapportait 131 € en moyenne. Après optimisation : 245 €
# 
# On  a doublé la rentabilité moyenne tout en réduisant le risque.
# 
# C’est le Graal du risk management : Moins de risque, plus de profit.
# 
# 
# Conclusion stratégique le modèle :
# 
# réduit le risque (PD ↓, LGD ↓)
# 
# augmente la rentabilité (EV ↑)
# 
# respecte la contrainte de risque globale
# 
# sélectionne un portefeuille plus sain et plus performant
# 
# En langage comité exécutif : **<mark>“L’optimisation améliore simultanément la qualité du portefeuille et sa rentabilité, démontrant une allocation optimale du capital sous contrainte de risque.</mark>**


# CELL ********************

selected_df.describe()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# creation of dataframe pandas with choice column from opt model
pdf_opt["optim_choice"] = pdf_opt["id"].isin(selected_ids).astype(int)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# convert into spark dataframe
df_optim = spark.createDataFrame(pdf_opt)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(df_optim.limit(3))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(df_silver.limit(3))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# construction of finale dataframe for reporting purpose

from pyspark.sql import functions as F

df_final_opt = (
    df_silver
    .join(
        df_optim.select("id", "optim_choice"),
        on="id",
        how="left"
    )
    .withColumn(
        "portfolio_type",
        F.when(F.col("optim_choice") == 1, "Optimized").otherwise("Baseline")
    )
)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df_final_opt = df_final_opt.withColumn(
    "credit_risk_niveau",
    F.when(F.col("credit_risk") == 1, "Risque élevé").otherwise("Risque faible")
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# creation of gold opt tables in  lakehouse Gold for reporting in PowerBi via Semantic Model and ML

# the Path for lakehouse(gold) reporting
gold_credit_scoring_path="abfss://76eb933a-950e-4895-8f00-76ccb4a5f37d@onelake.dfs.fabric.microsoft.com/3ee4579c-22ab-494d-8b27-fbf8a3b342fb/Tables/dbo/credit_scoring_gurobi_opt"

# Saving the Delta table into lakehouse
df_final_opt.write \
    .format("delta") \
    .mode("overwrite") \
    .option("mergeSchema", "true") \
    .save(gold_credit_scoring_path)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
