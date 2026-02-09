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

# Montant réel du crédit demandé (valeur numérique)
credit_amount = 2000   # EXEMPLE – remplacé par Power Automate

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 3. Construction du payload pour le modèle Spark ML
# ============================================================

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

# ============================================================
# 4. Fonctions métier : LGD et profit_margin
# ============================================================

def compute_LGD(purpose: str) -> float:
    if purpose == "car (new)": return 0.25
    if purpose == "car (used)": return 0.30
    if purpose == "furniture/equipment": return 0.35
    if purpose == "retraining": return 0.35
    if purpose == "radio/television": return 0.40
    if purpose == "repairs": return 0.40
    if purpose == "education": return 0.30
    if purpose == "domestic appliances": return 0.45
    if purpose == "business": return 0.60
    return 0.45


def compute_profit_margin(PD: float) -> float:
    if PD <= 0.05: return 0.08
    if PD <= 0.10: return 0.10
    if PD <= 0.25: return 0.12
    if PD <= 0.45: return 0.18
    return 0.25


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 5. Fonction principale : scoring enrichi
# ============================================================

def score_application(payload: dict, credit_amount: float) -> dict:
    """
    Fonction de scoring enrichi utilisée en production.
    Elle combine :
    - modèle ML (PD)
    - règles métier (LGD, profit_margin)
    - calculs financiers (EL, EV, etc.)
    - décision finale (ACCEPTE / REFUSE)
    """

    # 1) DataFrame Spark pour le modèle
    df = spark.createDataFrame([payload])

    # 2) Prédiction du modèle ML
    result = model.transform(df).select("probability", "prediction").collect()[0]
    PD = float(result["probability"][1])
    raw_prediction = int(result["prediction"])

    # 3) Paramètres métier
    LGD = compute_LGD(payload["purpose"])
    profit_margin = compute_profit_margin(PD)

    # 4) Calculs financiers
    exposure_at_risk = credit_amount
    expected_loss   = credit_amount * PD * LGD
    expected_profit = credit_amount * profit_margin
    expected_value  = expected_profit - expected_loss

    # 5) Décision binaire basée sur PD
    PD_threshold = 0.05
    credit_risk = 1 if PD > PD_threshold else 0

    # 6) Segment de risque
    if PD <= 0.03:
        risk_segment = "A"
    elif PD <= 0.10:
        risk_segment = "B"
    else:
        risk_segment = "C"

    decision = "ACCEPTE" if credit_risk == 0 else "REFUSE"

    # 7) Sortie finale
    return {
        "PD": PD,
        "raw_prediction": raw_prediction,
        "LGD": LGD,
        "profit_margin": profit_margin,
        "credit_amount": credit_amount,
        "exposure_at_risk": exposure_at_risk,
        "expected_loss": expected_loss,
        "expected_profit": expected_profit,
        "expected_value": expected_value,
        "credit_risk": credit_risk,
        "risk_segment": risk_segment,
        "decision": decision,
       
    }

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# 6. TEST MÉTIER : simulation d’un vrai client
# ============================================================

resultat = score_application(payload, credit_amount)
resultat

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# * PD = 0.6710 (67,10 %) : Probabilité de défaut estimée : 67 %
# 
# En langage métier : “Selon le modèle, ce client a environ 2 chances sur 3 de ne pas rembourser son crédit.” C’est extrêmement élevé pour un crédit conso.
# 
# * raw_prediction = 1 : Le modèle brut Spark ML classe ce client comme risqué. C’est cohérent avec la PD très élevée.
# 
# * LGD = 0.25 (25 %) : Si le client fait défaut, la banque perdrait 25 % du montant. C’est un LGD plutôt faible, car le produit “car (new)” est considéré comme ayant une certaine valeur de revente.
# 
# * profit_margin = 0.25 (25 %) : La banque appliquerait une marge de 25 % sur ce crédit. Pourquoi si élevée ?
# Parce que la PD est très élevée → le pricing automatique augmente la marge pour compenser le risque.
# 
# * credit_amount = 2000 : Le client demande 2000 €.
# 
# * exposure_at_risk = 2000 : La banque est exposée à 2000 € si le client ne rembourse pas.
# 
# * expected_loss = 335.52 € : Perte attendue = 335 €
# 
# Formule :𝐸𝐿=𝑃𝐷×𝐿𝐺𝐷×𝑚𝑜𝑛𝑡𝑎𝑛𝑡=0.671×0.25×2000 
# 
# En langage métier : “Statistiquement, ce crédit ferait perdre 335 € à la banque.”
# 
# * expected_profit = 500 € : Profit attendu si tout se passe bien = 500 €
# 
# * expected_value = 164.48 € : Valeur nette du crédit = 164 €. C’est positif, mais faible par rapport au risque.
# 
# * credit_risk = 1 : Le crédit dépasse le seuil PD de 5 % → classé risqué.
# 
# * risk_segment = C : Segment C ==> risque élevé.
# 
# Segments :
# 
#     A = très bon
# 
#     B = moyen
# 
#     C = risqué
# 
# * decision = REFUSE : Décision finale : REFUSÉ
# 
# <mark><u>**Interprétation métier globale**</u></mark>
# 
# “Le client présente un risque de défaut très élevé (67 %).
# Même si la marge appliquée est forte (25 %), la perte attendue reste importante (335 €).
# Le crédit tombe dans le segment C, considéré comme risqué.
# Conformément à la politique d’octroi, ce crédit doit être refusé.”
# 
# 
# “Le modèle estime une probabilité de défaut de 67 %, ce qui place ce client dans le segment C.
# La perte attendue est de 335 €, malgré une marge élevée.
# Le risque dépasse largement notre seuil d’acceptation.
# Je recommande de refuser ce crédit.”
# 
# **<mark><u>Reponse à donner au client : </u></mark>**
# 
# “Après analyse de votre dossier, certains éléments montrent que le risque de non‑remboursement est trop élevé selon nos critères internes.
# Pour cette raison, nous ne pouvons pas donner une suite favorable à votre demande.
# Nous pouvons cependant revoir ensemble d’autres solutions adaptées à votre situation.”
# 
# *** Conclusion en une phrase : 
# 
# **<u><mark>“Le crédit est refusé parce que le modèle estime que le client a 67 % de risque de ne pas rembourser, ce qui dépasse largement le seuil d’acceptation.”</mark></u>**

