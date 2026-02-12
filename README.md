# Credit Risk Optimization with Microsoft Fabric & Gurobi
*A full end‑to‑end decision intelligence pipeline built entirely inside Microsoft Fabric.*

## Overview

This project demonstrates how **Microsoft Fabric** can be used as a unified platform for **data engineering, machine learning, optimization, and business intelligence**.  
Although the use case focuses on **credit risk management**, the architecture and techniques presented here apply broadly across industries such as:

- **Logistics optimization** (routing, fleet allocation, delivery scheduling)  
- **Inventory management** (stock optimization, demand forecasting)  
- **Supply chain planning** (capacity planning, procurement optimization)  
- **Marketing analytics** (customer segmentation, churn prediction)  
- **Operational risk modeling**  
- **Fraud detection**  

The goal is to show how Fabric enables **real‑world analytical and optimization solutions** by combining ingestion, storage, transformation, ML, and decision optimization in a single environment.

The project uses the publicly available **German Credit dataset**  
<https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv>,  
a classic benchmark dataset containing 1,000 consumer loans with attributes such as purpose, age, credit amount, duration, and repayment behavior.  
It is widely used in credit scoring research and ML experimentation.

## What This Project Provides

This repository includes:

- **A complete medallion architecture** (Bronze → Silver → Gold) implemented in Fabric  
- **PySpark and SQL notebooks** for data cleaning, feature engineering, and ML  
- **A Gurobi optimization model** integrated directly into Fabric notebooks  
- **A Power BI report** showing baseline vs. optimized credit portfolio performance  
- **Reusable scripts and workflows** for ML, optimization, and reporting  
- **A fully reproducible Fabric environment** for end‑to‑end analytics

## Why Microsoft Fabric?

Fabric is used as the backbone of this project because it provides:

### Unified Data Ingestion
- Direct ingestion from online sources or APIs  
- Automatic schema inference  
- Zero‑copy access across engines (Spark, SQL, Power BI)

### Medallion Architecture

The project implements a clean, production‑grade medallion pipeline:

| Layer | Description |
|-------|-------------|
| **Bronze** | Raw ingestion of the German Credit dataset, mirroring production source data |
| **Silver** | Cleaned, transformed, feature‑engineered data for ML and optimization |
| **Gold** | Curated analytical tables optimized for Power BI dashboards |

### Multi‑Engine Processing

Fabric seamlessly integrates:

- **PySpark** for data engineering and optimization  
- **SQL** for analytics  
- **Power BI** for visualization  
- **OneLake** for unified storage  

### Integrated Deployment & Governance

All assets—pipelines, notebooks, ML models, semantic models, and reports—are deployed and governed inside Fabric.

## Machine Learning Pipeline

The ML workflow estimates **Probability of Default (PD)** and **Loss Given Default (LGD)**.

### Key Steps

#### 1. Data Assessment

The dataset exhibited a strong class imbalance:
- **70% high‑risk loans**
- **30% low‑risk loans**

#### 2. Statistical & ML Preprocessing

To address this imbalance, we applied recommended techniques such as:
- **Oversampling** the minority class  
- Feature scaling  
- Encoding categorical variables  
- Train/test split  

#### 3. Model Training

Multiple ML models were evaluated (logistic regression, tree‑based models, etc.) to estimate PD and LGD.

#### 4. Integration into Silver Layer

Model outputs are stored in the Silver layer and used as inputs for optimization.

## Gurobi Optimization in Fabric

A major highlight of this project is the integration of the **Gurobi solver** directly inside a **Fabric PySpark notebook**.

This demonstrates that Fabric is not only a BI or data engineering platform, but also a **decision optimization environment**.

### Optimization Objective

Select the best subset of credits that:

- Maximizes **Expected Value (EV)**  
- Minimizes **Expected Loss (EL)**  
- Reduces portfolio risk exposure  
- Respects business constraints  

### Baseline vs. Optimized Portfolio

| Metric | Baseline | Optimized |
|--------|----------|-----------|
| Number of Credits | 1,000 | 449 |
| Total Expected Loss | 5.16K | 1.23K |
| Exposure at Risk | 1.42M | 425K |
| Risk Efficiency | 0.25 | 0.90 |

The optimization dramatically improves portfolio quality and reduces risk.

## Power BI Reporting

The final Gold tables feed a Power BI dashboard built inside Fabric.

### Key Visuals

- **KPIs**: Expected Loss, Exposure at Risk, Expected Value, Risk Efficiency  
- **Risk Distribution**: Baseline vs. Optimized  
- **Expected Loss Reduction**: Waterfall chart  
- **Heatmaps using PBIVizEdit Heatmap Chart (Pro)**  
  - Expected Loss by Purpose & Age Group  
  - Optimized Expected Loss by Purpose & Age Group  

These heatmaps reveal:

- Which customer segments concentrate the most risk  
- How optimization shifts risk away from high‑loss segments  
- How purpose and age interact to influence portfolio risk  

## Project Structure

```text
├── notebooks/
│   ├── bronze_ingestion/      # Raw data ingestion
│   ├── silver_processing/     # Cleaning, feature engineering, ML
│   ├── optimization/          # Gurobi optimization model
│   └── gold_reporting/        # Tables for Power BI
├── pipelines/                 # Fabric Data Factory pipelines
├── powerbi/                   # Power BI report files
├── docs/                      # Documentation
└── README.md                  # Project overview
```
## Conclusion

This project demonstrates how **Microsoft Fabric** can serve as a complete platform for:

- Data ingestion  
- Data engineering  
- Machine learning  
- Optimization with Gurobi  
- Business intelligence  

All within a single, unified environment.

It highlights the strategic value of combining **ML predictions** with **mathematical optimization** to build smarter, safer, and more profitable credit portfolios.

Fabric’s integration of **Spark**, **SQL**, **OneLake**, and **Power BI** — combined with **Gurobi’s optimization power** — creates a modern, scalable, and production‑ready architecture for **credit risk management** and many other domains such as **logistics**, **supply chain**, and **operations research**.




# Optimisation du Risque Crédit avec Microsoft Fabric & Gurobi  
*Une chaîne décisionnelle complète construite entièrement dans Microsoft Fabric.*

## Présentation

Ce projet démontre comment **Microsoft Fabric** peut être utilisé comme une plateforme unifiée pour **l’ingestion de données, l’ingénierie de données, le machine learning, l’optimisation mathématique et la business intelligence**.  
Bien que le cas d’usage porte sur la **gestion du risque crédit**, l’architecture et les techniques présentées ici s’appliquent à de nombreux autres domaines métiers, tels que :

- **Logistique** (optimisation de tournées, allocation de flotte, planification des livraisons)  
- **Gestion des stocks** (optimisation des niveaux de stock, prévision de la demande)  
- **Supply chain** (planification capacitaire, optimisation des achats)  
- **Marketing analytique** (segmentation, prédiction du churn)  
- **Gestion des risques opérationnels**  
- **Détection de fraude**  

L’objectif est de montrer comment Fabric permet de construire des **solutions analytiques et décisionnelles réelles**, en combinant ingestion, stockage, transformation, machine learning et optimisation dans un environnement unique.

Le projet utilise le jeu de données public **German Credit**  
<https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv>,  
un dataset de référence contenant 1 000 crédits consommateurs avec des variables telles que le but du crédit, l’âge, le montant, la durée et le comportement de remboursement.  
Il est largement utilisé dans la recherche en scoring et dans les expérimentations ML.

## Ce que fournit ce projet

Le dépôt inclut :

- **Une architecture médaille complète** (Bronze → Silver → Gold) dans Fabric  
- **Des notebooks PySpark et SQL** pour le nettoyage, la préparation et le ML  
- **Un modèle d’optimisation Gurobi** intégré directement dans Fabric  
- **Un rapport Power BI** comparant le portefeuille Baseline vs Optimisé  
- **Des scripts réutilisables** pour le ML, l’optimisation et le reporting  
- **Un environnement Fabric entièrement reproductible** pour l’analyse de bout en bout

## Pourquoi Microsoft Fabric ?

Fabric est utilisé comme socle du projet car il offre :

### Ingestion unifiée des données
- Ingestion directe depuis des sources en ligne ou des API  
- Détection automatique des schémas  
- Accès unifié sans copie entre Spark, SQL et Power BI  

### Architecture Médaille

Le projet implémente une architecture médaille propre et industrialisable :

| Couche | Description |
|--------|-------------|
| **Bronze** | Données brutes issues de la source German Credit |
| **Silver** | Données nettoyées, enrichies et prêtes pour le ML et l’optimisation |
| **Gold** | Tables analytiques prêtes pour Power BI |

### Traitement multi‑moteurs

Fabric intègre de manière fluide :

- **PySpark** pour l’ingénierie et l’optimisation  
- **SQL** pour l’analyse  
- **Power BI** pour la visualisation  
- **OneLake** pour le stockage unifié  

### Déploiement & gouvernance intégrés

Tous les artefacts — pipelines, notebooks, modèles ML, modèles sémantiques, rapports — sont gérés et déployés dans Fabric.

## Pipeline Machine Learning

Le workflow ML estime la **Probabilité de Défaut (PD)** et la **Perte en cas de Défaut (LGD)**.

### Étapes clés

#### 1. Analyse des données

Le dataset présentait un fort déséquilibre :
- **70 % de crédits à risque**
- **30 % de crédits sains**

#### 2. Prétraitements statistiques & ML

Pour corriger ce déséquilibre, nous avons appliqué des techniques recommandées :
- **Sur‑échantillonnage** de la classe minoritaire  
- Normalisation  
- Encodage des variables catégorielles  
- Séparation train/test  

#### 3. Entraînement des modèles

Plusieurs modèles ML ont été testés (régression logistique, arbres, etc.) pour estimer PD et LGD.

#### 4. Intégration dans la couche Silver

Les prédictions ML alimentent ensuite le modèle d’optimisation.

## Optimisation Gurobi dans Fabric

L’un des points forts du projet est l’intégration du **solveur Gurobi** directement dans un **notebook PySpark Fabric**.

Cela montre que Fabric n’est pas seulement une plateforme BI ou data engineering, mais aussi un **environnement d’optimisation décisionnelle**.

### Objectif de l’optimisation

Sélectionner le meilleur sous‑ensemble de crédits qui :

- maximise la **valeur attendue (EV)**  
- minimise la **perte attendue (EL)**  
- réduit l’exposition au risque  
- respecte les contraintes métier  

### Résultats Baseline vs Optimisé

| Indicateur | Baseline | Optimisé |
|------------|----------|----------|
| Nombre de crédits | 1 000 | 449 |
| Perte attendue totale | 5.16K | 1.23K |
| Exposition au risque | 1.42M | 425K |
| Efficacité du risque | 0.25 | 0.90 |

L’optimisation améliore fortement la qualité du portefeuille.

## Reporting Power BI

Les tables Gold alimentent un rapport Power BI construit dans Fabric.

### Visuels clés

- **KPIs** : Expected Loss, Exposure at Risk, Expected Value, Risk Efficiency  
- **Distribution du risque** : Baseline vs Optimisé  
- **Réduction de la perte attendue** : graphique en cascade  
- **Heatmaps (PBIVizEdit Heatmap Chart Pro)**  
  - Expected Loss par Purpose & Age Group  
  - Optimized Expected Loss par Purpose & Age Group  

Ces heatmaps révèlent :

- les segments les plus risqués  
- comment l’optimisation redistribue le risque  
- l’interaction entre âge et finalité du crédit  

## Structure du projet

```text
├── notebooks/
│   ├── bronze_ingestion/
│   ├── silver_processing/
│   ├── optimization/
│   └── gold_reporting/
├── pipelines/
├── powerbi/
├── docs/
└── README.md
```

## Conclusion

Ce projet montre comment **Microsoft Fabric** peut servir de plateforme complète pour :

- l’ingestion de données  
- l’ingénierie de données  
- le machine learning  
- l’optimisation avec Gurobi  
- la business intelligence  

Le tout dans un environnement unifié.

Il met en évidence la valeur stratégique de combiner **prédictions ML** et **optimisation mathématique** pour construire des portefeuilles de crédit plus sûrs, plus performants et plus rentables.

L’intégration de **Spark**, **SQL**, **OneLake** et **Power BI**, associée à la puissance de **Gurobi**, constitue une architecture moderne, scalable et prête pour la production — applicable non seulement au risque crédit, mais aussi à la **logistique**, à la **supply chain**, et à l’**optimisation opérationnelle**.

