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

#Charger Spark et modèle depuis le Lakehouse Gold
from pyspark.ml import PipelineModel
#from notebookutils import parameters

# Chemin vers ton modèle dans le Lakehouse Gold
model_path = "abfss://76eb933a-950e-4895-8f00-76ccb4a5f37d@onelake.dfs.fabric.microsoft.com/3ee4579c-22ab-494d-8b27-fbf8a3b342fb/Files/CREDIT_SCORING_MODEL"

# Charger le modèle Spark ML
model = PipelineModel.load(model_path)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#Récupération des paramètres envoyés par Power Automate
status = "default"
credit_history = "default"
purpose = "default"
savings = "default"
employment_duration = "default"
personal_status_sex = "default"
other_debtors = "default"
property = "default"
other_installment_plans = "default"
housing = "default"
job = "default"
telephone = "default"
foreign_worker = "default"
duration_cat = "default"
amount_cat = "default"
present_residence_cat = "default"
age_cat = "default"
number_credits_cat = "default"
people_liable_cat = "default"


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#Définir les variables d’entrée (plus tard envoyées par Power Automate)
# EXEMPLE DE DONNÉES (à remplacer par les valeurs venant de Power Automate)
status = "no checking account"
credit_history = "existing credits paid back duly till now"
purpose = "car (new)"
savings = "... < 100 DM"
employment_duration = "1 <= ... < 4 years"
personal_status_sex = "male : single"
other_debtors = "none"
property = "car or other"
other_installment_plans = "none"
housing = "own"
job = "skilled employee/official"
telephone = "no"
foreign_worker = "yes"
duration_cat = "13-24"
amount_cat = "1001-2000"
present_residence_cat = "2-3"
age_cat = "25-35"
number_credits_cat = "0-1"
people_liable_cat = "1"


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

payload = {
    "status": status,
    "credit_history": credit_history,
    "purpose": purpose,
    "savings": savings,
    "employment_duration": employment_duration,
    "personal_status_sex": personal_status_sex,
    "other_debtors": other_debtors,
    "property": property,
    "other_installment_plans": other_installment_plans,
    "housing": housing,
    "job": job,
    "telephone": telephone,
    "foreign_worker": foreign_worker,
    "duration_cat": duration_cat,
    "amount_cat": amount_cat,
    "present_residence_cat": present_residence_cat,
    "age_cat": age_cat,
    "number_credits_cat": number_credits_cat,
    "people_liable_cat": people_liable_cat
}

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Créer un DataFrame Spark avec une seule ligne
df = spark.createDataFrame([payload])

# Appliquer le modèle
result = model.transform(df).select("probability", "prediction").collect()[0]

# Extraire les résultats
p_default = float(result["probability"][1])
prediction = int(result["prediction"])
decision = "ACCEPTE" if prediction == 0 else "REFUSE"


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

output = {
    "probability_default": p_default,
    "prediction": prediction,
    "decision": decision
}

print(output)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
