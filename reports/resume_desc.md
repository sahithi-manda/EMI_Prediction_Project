# Resume Description - EMIPredict AI

This document contains professional bullet points that can be added to a software engineer / data scientist resume.

## B.Tech CSE-AIML Portfolio Bullet Points

### Option 1: Detailed Project Description
**EMIPredict AI – Smart Credit & EMI Repayment Capability Estimator**
*Technologies: Python, Scikit-Learn, Streamlit, Pandas, NumPy, Matplotlib, Seaborn, Joblib*
* Developed an end-to-end Machine Learning pipeline to predict loan eligibility (binary classification) and estimate the maximum affordable monthly EMI obligation (continuous regression) for retail banking customers.
* Designed a unified preprocessing pipeline using scikit-learn `ColumnTransformer` with `StandardScaler` and `OneHotEncoder` to avoid data leakage and ensure seamless inference serialization.
* Evaluated multiple ML architectures, selecting **Logistic Regression** for classification (F1-Score: 73.6%, Accuracy: ~90%) and **Random Forest Regressor** for regression (R²: 81.9%, MAE: ~3,570 INR).
* Built a premium interactive **Streamlit dashboard** equipped with stateful profile forms, custom glassmorphism HTML elements, dynamic loan capacity simulators, and correlation/feature importance analytics.
* Serialized the trained pipelines with `joblib` and served predictions through the Streamlit app.

---

### Option 2: High-Impact Bullet Points (Short version)
* **Machine Learning Pipelines**: Built and serialized unified preprocessing and model pipelines (classification & regression) using scikit-learn for loan eligibility prediction and EMI limit estimation.
* **Model Benchmarking**: Compared Logistic Regression and Random Forest architectures on credit metrics, achieving a classification accuracy of ~90% and a regression R² score of 81.9%.
* **Interactive Deployment**: Developed a Streamlit web application featuring a loan tenure calculator, feature importance visualizations, and custom-styled financial health indicator cards.
