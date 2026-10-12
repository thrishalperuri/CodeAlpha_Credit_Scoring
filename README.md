# CodeAlpha_Credit_Scoring

## 📌 Project Overview
This repository contains **Task 1: Credit Scoring Model** developed for the **CodeAlpha Machine Learning Internship**.

The goal of this project is to build an end-to-end machine learning system that predicts an individual's creditworthiness (credit risk) based on their historical financial behavior, income, debts, and credit profiles.

---

## 🎯 Problem Statement & Objectives
Financial institutions evaluate loan applicants to minimize defaults and financial risk. This project tackles credit scoring as a binary classification problem:
- **`1` (Creditworthy / Approved):** Low risk of default.
- **`0` (High Risk / Denied):** High likelihood of delinquency or non-repayment.

---

## 🔑 Key Features & Dataset Schema
The model uses key financial and behavioral variables:
| Feature | Description |
| :--- | :--- |
| **`Age`** | Applicant's age |
| **`Annual_Income`** | Annual pre-tax income in USD |
| **`Employment_Years`**| Length of continuous employment history |
| **`Total_Debt`** | Total outstanding debt obligations |
| **`DTI_Ratio`** | Debt-to-Income ratio |
| **`Open_Credit_Lines`**| Number of open credit lines/accounts |
| **`Delinquent_Accounts`**| Number of past late payments or defaulted accounts |
| **`Credit_Score`** | Standard FICO credit score (300 - 850) |
| **`Loan_Amount`** | Principal loan amount requested |
| **`Loan_Purpose`** | Purpose of credit (Debt Consolidation, Home Improvement, Business, etc.) |
| **`Creditworthy`** | **Target variable** (`1` = Approved, `0` = Denied) |

---

## ⚙️ Machine Learning Pipeline & Algorithms
Multiple supervised classification algorithms are evaluated and benchmarked:
1. **Logistic Regression** (baseline linear classifier)
2. **Decision Tree Classifier** (rule-based non-linear baseline)
3. **Random Forest Classifier** (bagging ensemble)
4. **Gradient Boosting Classifier** (boosting ensemble)

### 📈 Evaluation Metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC Score**
- **Confusion Matrix Analysis**

---

## 🚀 Project Structure
```text
CodeAlpha_Credit_Scoring/
├── app.py                     # Interactive Streamlit Web Application
├── train_model.py             # Model training, comparison & plot generation script
├── requirements.txt           # Python dependencies
├── README.md                  # Comprehensive project documentation
├── data/
│   ├── credit_data.csv        # Financial dataset
│   └── generate_data.py       # Data synthesis & distribution script
├── models/
│   └── credit_model.pkl       # Serialized best model pipeline
└── plots/
    ├── roc_curve_comparison.png # ROC curves across models
    ├── confusion_matrix.png     # Best model confusion matrix
    └── feature_importance.png   # Top risk drivers plot
```

---

## 💻 Installation & Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/your-username/CodeAlpha_Credit_Scoring.git
cd CodeAlpha_Credit_Scoring
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model & Generate Benchmark Plots
```bash
python train_model.py
```
This trains all models, compares their scores, saves the best model to `models/credit_model.pkl`, and exports visualization plots into `plots/`.

### 4. Launch the Interactive Web Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser to interact with the loan assessment tool!

---

## 📊 Visualizations & Output Previews
- **ROC-AUC Comparison Curve:** Compares classification separation capabilities across models.
- **Confusion Matrix:** Tracks True Positives vs False Positives/Negatives.
- **Feature Importance:** Highlights key factors (e.g. Credit Score, DTI Ratio, Delinquencies) influencing the decision.

---

## 👨‍💻 Internship Submission Details
- **Organization:** [CodeAlpha](https://www.codealpha.tech/)
- **Domain:** Machine Learning
- **Task 1:** Credit Scoring Model
