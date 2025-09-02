# 🏦 Customer Churn Prediction (Logistic Regression)

## 📌 Project Overview

This project predicts **customer churn** (whether a customer leaves the bank) using a **Logistic Regression model**.  
The model is trained on customer demographic and financial data and outputs churn risk.

### 🔑 Key Steps:
1. Exploratory Data Analysis (EDA + Visuals)
2. Data Cleaning & Preprocessing
3. Feature Engineering  
4. Model Training & Evaluation  
5. Interpretation of Results  
6. Saving Model for Deployment  

---

## 📊 Dataset

- **Source:** [Bank Customer Churn Dataset (Kaggle)](https://www.kaggle.com/datasets/shubhammeshram579/bank-customer-churn-prediction)  
- **Target:** `Exited` → (1 = Churned, 0 = Stayed)  
- **Features:** Demographics, account information, and financial data.  

---

## ⚙️ Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/lashm4/Bank-Churn-Prediction.git
cd Bank-Churn-Prediction
pip install -r requirements.txt
```

---

## 🚀 Usage

Run the Jupyter Notebook:

```bash
jupyter notebook notebooks/churn_model.ipynb
```

Or load the pre-trained model for predictions:

```python
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("models/logistic_regression_churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Example usage
X_new = pd.DataFrame([[600, 40, 3, 60000]],
                     columns=["CreditScore","Age","NumOfProducts","EstimatedSalary"])
X_new_scaled = scaler.transform(X_new)
prediction = model.predict(X_new_scaled)
print("Churn Prediction:", prediction)
```

---

## 📈 Model Performance

- **Training Accuracy:** 81.5%  
- **Test Accuracy:** 80.6% → model generalizes reasonably well on unseen data  
- **ROC-AUC:** 0.578 → model is only slightly better than random at separating churners from non-churners

---

## 🔑 Top Features & Business Insights (Odds Ratios)

- **Age (OR = 2.11)** → older customers are ~2.1× more likely to churn per additional year  
- **IsActiveMember (OR = 0.60)** → active members are 40% less likely to churn  
- **Balance (OR = 1.17)** → higher balances slightly increase churn risk  
- **NumOfProducts (OR = 0.95)** → customers with more products are slightly less likely to churn  
- **Geography_Germany (OR = 1.43)** → German customers are 43% more likely to churn than French customers  
- **Gender (OR = 0.77)** → males are less likely to churn than females  
- **Geography_Spain (OR = 1.02)** → minimal impact, similar to France

**Business Insights:**

- Customers with high balance but fewer products are at higher risk of churn  
- Active members and customers with multiple products are safer  
- Targeted retention campaigns (e.g., personalized offers, loyalty incentives) could reduce churn in high-risk segments

---

## 📊 Dashboard

Interactive visual insights are available on Tableau Public:  
👉 [Churn Insights Dashboard](https://public.tableau.com/app/profile/lashmi.munante/viz/ChurnInsightsDashboard/Dashboard1#1)

---

## 📌 Next Steps

- Train advanced models (Random Forest, XGBoost, Gradient Boosting)  
- Build a Streamlit or Flask web app for deployment  
- Expand business intelligence integration with Tableau / Power BI

---

## 📜 Requirements

Dependencies are listed in `requirements.txt`, including:

- pandas  
- numpy  
- scikit-learn  
- seaborn  
- matplotlib  
- joblib  
- jupyter

---

## 👩‍💻 Author

Created by **Lashmi M.**,  feel free to reach out!

