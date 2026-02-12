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


