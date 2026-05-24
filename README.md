# 🏦 Bank Customer Churn Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?style=flat-square&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-best%20model-green?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

> Predicting which bank customers are likely to churn using classical ML models and a deep neural network — with full EDA, preprocessing, SMOTE resampling, and model comparison.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Models & Results](#-models--results)
- [Key Findings](#-key-findings)
- [Getting Started](#-getting-started)
- [Dependencies](#-dependencies)

---

## 🔍 Overview

Customer churn is one of the most costly problems in retail banking. This project builds and compares multiple classification models to identify customers at risk of leaving the bank, enabling proactive retention strategies.

The pipeline covers:
- **Exploratory Data Analysis (EDA)** with distribution plots, correlation heatmaps, and feature-vs-churn breakdowns
- **Preprocessing** with `StandardScaler` and `OneHotEncoder` via `ColumnTransformer`
- **Class imbalance handling** using SMOTE (Synthetic Minority Oversampling Technique)
- **Multi-model benchmarking** across 5 classical classifiers
- **XGBoost** as the best-performing model
- **Deep Neural Network** with Dropout, Batch Normalization, and Early Stopping

---

## 📊 Dataset

**File:** `Bank_Churn_Modelling.xlsx`

| Property | Value |
|---|---|
| Records | 10,000 customers |
| Features | 14 columns |
| Target | `Exited` (1 = churned, 0 = stayed) |
| Churn Rate | ~20.4% |
| Geographies | France (50.1%), Germany (25.1%), Spain (24.8%) |

### Features

| Feature | Type | Description |
|---|---|---|
| `CreditScore` | Numerical | Customer credit score |
| `Geography` | Categorical | Country (France, Germany, Spain) |
| `Gender` | Categorical | Male / Female |
| `Age` | Numerical | Customer age |
| `Tenure` | Numerical | Years as a bank customer |
| `Balance` | Numerical | Account balance |
| `NumOfProducts` | Numerical | Number of bank products used |
| `HasCrCard` | Binary | Owns a credit card (1/0) |
| `IsActiveMember` | Binary | Active member status (1/0) |
| `EstimatedSalary` | Numerical | Estimated annual salary |
| `Exited` | Binary | **Target** — churned (1) or stayed (0) |

> `RowNumber`, `CustomerId`, and `Surname` are dropped as non-predictive identifiers.

---

## 📁 Project Structure

```
├── Bank_Churn_Modelling.xlsx    # Raw dataset
├── Churn_Classification.ipynb   # Main analysis notebook
├── dashboard.html               # Interactive project dashboard
└── README.md                    # This file
```

---

## 🔬 Methodology

### 1. Data Preprocessing & EDA
- Dropped identifier columns (`RowNumber`, `CustomerId`, `Surname`)
- Visualized churn distribution, feature histograms, boxplots
- Checked for duplicates and null values
- Outlier capping for `Age` using IQR method
- Correlation heatmap to understand feature relationships

### 2. Data Splitting
The dataset is split into three parts using stratified sampling:

```
Train  →  70%   (7,000 samples)
Val    →  15%   (1,500 samples)
Test   →  15%   (1,500 samples)
```

### 3. Feature Engineering
- **Numerical features** → `StandardScaler`
- **Categorical features** (`Geography`, `Gender`) → `OneHotEncoder(drop='first')`
- All via `sklearn.compose.ColumnTransformer`

### 4. Class Imbalance — SMOTE
With ~20% churn rate, SMOTE is applied to the training set to generate synthetic minority samples and prevent model bias toward the majority class.

### 5. Model Training & Evaluation
Models are evaluated on the validation set using accuracy, precision, recall, F1-score, and ROC-AUC.

---

## 🏆 Models & Results

### Classical Model Comparison

| Model | Validation Accuracy |
|---|---|
| Logistic Regression | ~81% |
| Logistic Regression (balanced) | ~76% |
| SVM | ~85% |
| Random Forest | ~86% |
| KNN | ~83% |
| **XGBoost** | **~85%** |

### ⭐ Best Model — XGBoost

| Metric | Value |
|---|---|
| **Test Accuracy** | **85.13%** |
| **ROC-AUC** | **0.847** |
| Churn Precision | 0.72 |
| Churn Recall | 0.44 |
| Churn F1-Score | 0.55 |

> XGBoost delivers reliable churn predictions with strong overall accuracy. The recall of 0.44 on the churn class reflects the inherent difficulty of the imbalanced problem — precision is prioritized to minimize false alarms.

### 🧠 Neural Network (TensorFlow/Keras)

A sequential deep network was also trained:

```
Input → Dense(128, relu) + L2 regularization
      → Dropout(0.2) + BatchNormalization
      → Dense(64, relu) + Dropout(0.2)
      → Dense(1, sigmoid)
```

- Optimizer: Adam
- Loss: Binary Crossentropy
- Early Stopping: patience=2 on `val_loss`

---

## 💡 Key Findings

- **Age** is the strongest predictor of churn — older customers churn significantly more.
- **German customers** have a disproportionately high churn rate compared to France and Spain.
- **Inactive members** (`IsActiveMember=0`) churn at much higher rates.
- **Customers with 3–4 products** have nearly 100% churn rate — a strong signal.
- **Balance** shows a bimodal distribution; customers with zero balance rarely churn.
- **Gender** (female) shows slightly higher churn, but is a weaker predictor overall.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/bank-churn-classification.git
cd bank-churn-classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

```bash
jupyter notebook Churn_Classification.ipynb
```

---

## 📦 Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
tensorflow
openpyxl
jupyter
```

Install all at once:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost tensorflow openpyxl jupyter
```

---
