# 🏦 Customer Churn Prediction (Logistic Regression)

## Project Overview
This project predicts **customer churn** (whether a customer leaves the bank) using a **Logistic Regression model**.  
The model is trained on customer demographic and financial data and outputs churn risk.

### Key Steps:
1. Data Cleaning & Preprocessing  
2. Exploratory Data Analysis (EDA + Visuals)  
3. Feature Engineering  
4. Model Training & Evaluation  
5. Interpretation of Results  
6. Saving Model for Deployment  

---

## 📊Dataset
- **Source:** [Bank Customer Churn Dataset (Kaggle)](https://www.kaggle.com/datasets/shubhammeshram579/bank-customer-churn-prediction)  
- **Target:** `Exited` → (1 = Churned, 0 = Stayed)  
- **Features:** Demographics, account information, and financial data.  

---

## ⚙️Installation
Clone this repository and install dependencies:

```bash
git clone https://github.com/yourusername/churn-prediction-logreg.git
cd churn-prediction-logreg
pip install -r requirements.txt

**## 🚀Usage**
Run the Jupyter Notebook:
jupyter notebook notebooks/churn_model.ipynb

Or load the pre-trained model for predictions:

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

📈 Model Performance

Training Accuracy: 81.5%

Test Accuracy: 81% → indicates the model generalizes reasonably well

ROC-AUC: 0.596 → model is not great at separating churners from non-churners

🔑 Key Insights:

Age, Activity Status, Balance, and Geography are the strongest churn predictors.

Older, inactive, high-balance customers (especially from Germany) are more likely to churn.

Active members and those with multiple products are less likely to leave.

📊 Dashboard

Interactive visual insights are available on Tableau Public:
👉 Churn Insights Dashboard

📌 Next Steps

Train advanced models (Random Forest, XGBoost, Gradient Boosting)

Build a Streamlit or Flask web app for deployment

Expand business intelligence integration with Tableau / Power BI

📜 Requirements

- Dependencies are listed in requirements.txt, including:

- pandas

- numpy

- scikit-learn

- seaborn

- matplotlib

- joblib

👩‍💻 Author

Created by Lashmi M., feel free to reach out!
