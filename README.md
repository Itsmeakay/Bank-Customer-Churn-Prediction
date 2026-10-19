# 🏦 Bank Customer Churn Prediction

An end-to-end Machine Learning project that predicts the probability of bank customer churn and classifies customers into Low, Medium, and High Risk categories.

## 🎯 Project Objective

The objective of this project is to:

- Predict whether a bank customer is likely to churn.
- Calculate individual customer churn probability.
- Identify important factors associated with customer churn.
- Categorize customers based on churn risk.
- Provide an interactive Streamlit dashboard for customer risk prediction.

## 📊 Dataset

The project uses the Churn Modelling dataset.

- Records: 10,000
- Features: 14
- Target Variable: `Exited`

### Target

- `0` → Customer stayed
- `1` → Customer churned

## 🔍 Exploratory Data Analysis

The project includes:

- Dataset structure analysis
- Missing-value analysis
- Duplicate detection
- Churn distribution
- Categorical analysis
- Age analysis
- Balance analysis
- Product analysis
- Active-member analysis
- Correlation analysis
- Outlier detection using IQR

### Key Findings

- Overall churn rate: approximately **20.37%**
- Germany showed higher observed churn than France and Spain.
- Older customers showed higher observed churn in several age groups.
- Customers with 3 or 4 products showed unusually high observed churn.
- Inactive members had higher observed churn than active members.

## 🛠️ Feature Engineering

The following features were created:

### Balance-Salary Ratio

```python
BalanceSalaryRatio = Balance / EstimatedSalary