# insurance-risk-prediction
End-to-end Machine Learning pipeline using XGBoost to identify underpriced insurance policies

# 🚗 AI-Driven Insurance Risk Engine: Solving the "Profitability Gap"

### 📄 Project Overview
In the insurance industry, the "Profitability Gap" occurs when a policyholder's risk profile (expected claims) exceeds the premium collected. This project builds a machine learning engine to identify these "Silent Risks" and optimize the carrier's Loss Ratio.

### 🎯 The Business Problem
* **The Goal:** Predict the `Total Claim Amount` for auto insurance customers.
* **The Value:** Identify underpriced policies where `Predicted_Loss > Premium_Collected`.
* **The Impact:** Enable dynamic pricing strategies to reduce the Loss Ratio.

### 🛠️ Tech Stack
* **Language:** Python
* **Modeling:** XGBoost Regressor (Gradient Boosting)
* **Explainability:** SHAP (SHapley Additive exPlanations)
* **Libraries:** Pandas, Scikit-Learn, Seaborn, Matplotlib

### 📊 Key Results
* **Model Accuracy:** Achieved an **R² Score of 0.85** on unseen test data.
* **RMSE:** The model predicts claim amounts with an average error of ~$150.
* **Risk Discovery:** Identified that **Suburban Location** was the #1 strongest driver of high claim severity, outweighing Income or Vehicle Class.

### 🧠 Key Insights (SHAP Analysis)
The model opened the "Black Box" to reveal why certain customers were flagged as high risk:
1.  **Location Matters:** Suburban drivers showed significantly higher claim severities than urban drivers.
2.  **Vehicle Sensitivity:** "Luxury SUVs" have a non-linear impact on claims (exponentially higher repair costs).
3.  **Income Irrelevance:** Surprisingly, customer income was a weak predictor of claim severity compared to driving history.

### 🚀 How to Run
1.  Clone the repository.
2.  Install dependencies: `pip install pandas xgboost shap sklearn matplotlib`
3.  Run the Jupyter Notebook `Insurance_Risk_Prediction_Engine.ipynb`.

---
*Data Source: Kaggle (Auto Insurance Claims Updated 2024)*
