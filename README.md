# EMIPredict AI – EMI Eligibility and Affordability Predictor

An end-to-end machine learning project designed for B.Tech CSE-AIML portfolios. It features a dual-engine AI system that evaluates retail loan eligibility (classification) and calculates maximum comfortable monthly Equated Monthly Installments (EMIs) (regression) based on a customer's demographic and financial profile.

---

## 🚀 Key Features

* **Dual-Engine ML Inference**:
  * **Classification (Eligibility)**: Employs a **Logistic Regression** pipeline to classify profile eligibility with probability confidence rates and risk levels.
  * **Regression (Affordability)**: Employs a **Random Forest Regressor** pipeline to recommend the maximum affordable EMI.
* **Stateful Input Form**: Preserves user input values across the eligibility and affordability prediction calculators for seamless UX.
* **Simulated Principal Loan Capacity**: An amortization simulator calculating maximum principal borrowing limits based on predicted monthly repayment capacities, interest rates, and tenures.
* **Interactive Analytics**: Interactive distributions, a correlation matrix heatmap, and model feature weights.
* **Academic Assets**: Built-in resume descriptions, viva presentation guidelines, and academic reports.

---

## 🛠️ Tech Stack

* **Language**: Python 3.9+
* **Data Processing**: Pandas, NumPy
* **Machine Learning**: Scikit-Learn (Pipelines, ColumnTransformer)
* **Visualization**: Matplotlib, Seaborn
* **Deployment**: Streamlit
* **Model Serialization**: Joblib

---

## 📁 Project Structure

```text
EMIPredict-AI/
│
├── data/
│   ├── generate_data.py          # Script to generate synthetic credit records
│   └── emi_dataset.csv           # Generated customer dataset (2,500 records)
│
├── notebooks/
│   ├── generate_notebook.py      # Script to programmatically write the Jupyter notebook
│   └── emi_prediction_model.ipynb # Jupyter Notebook containing EDA and model benchmarking
│
├── models/
│   ├── train.py                  # Main script to train and save ML models
│   ├── best_classifier.joblib     # Serialized Logistic Regression pipeline
│   ├── best_regressor.joblib      # Serialized Random Forest Regressor pipeline
│   ├── classifier_stats.joblib    # Accuracy and confusion matrices data
│   ├── regressor_stats.joblib     # RMSE, MAE, R² score data
│   └── feature_importance.joblib  # Pre-computed feature weights
│
├── app/
│   ├── main.py                    # Main Streamlit dashboard application
│   └── utils.py                   # Custom glassmorphism components and styling
│
├── reports/
│   ├── project_report.md          # Full academic thesis report
│   ├── ppt_content.md             # Structure and slides for viva presentations
│   └── resume_desc.md             # Tailored resume descriptions
│
├── requirements.txt               # Dependencies list
└── README.md                      # Setup and usage guide (this file)
```

---

## 💻 Installation & Setup

Follow these steps to run the project locally:

### 1. Clone or Copy the Workspace
Navigate into the project workspace directory:
```bash
cd "EMIPredict AI"
```

### 2. Create and Activate a Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Populate and Train Models
Run the generation and training scripts to build the datasets and export model files:
```bash
# Generate synthetic dataset
python data/generate_data.py

# Train models and export joblib pipeline files
python models/train.py
```

### 5. Launch the Streamlit Dashboard
```bash
streamlit run app/main.py
```
This will start the application locally. Open `http://localhost:8501` in your browser.

---

## 📈 Model Performance Summary

### Classification (Eligibility Engine)
* **Best Model**: Logistic Regression
* **Accuracy**: ~89.8%
* **Precision**: ~86.6%
* **Recall**: ~64.0%
* **F1-Score**: ~73.6%

### Regression (Affordability Engine)
* **Best Model**: Random Forest Regressor
* **R² Score**: 81.9%
* **Mean Absolute Error (MAE)**: ~₹3,576
* **Root Mean Squared Error (RMSE)**: ~₹6,968
