# Academic Project Report: EMIPredict AI
**Project Title**: EMIPredict AI – EMI Eligibility and Affordability Predictor  
**Course Domain**: B.Tech CSE (Artificial Intelligence & Machine Learning)  
**Academic Year**: 2026  

---

## Abstract
Modern retail lending depends on credit-scoring mechanisms to determine whether a borrower qualifies for financial products. However, conventional credit scores (such as FICO or CIBIL) are primarily binary filters, failing to compute the actual monthly cash capacity of a borrower. This study presents **EMIPredict AI**, a dual-engine machine learning solution designed to predict:
1. **EMI Eligibility**: A classification system evaluating credit-worthiness norms using Logistic Regression.
2. **Maximum Affordable EMI**: A regression system calculating safe monthly repayments using Random Forest Regressor.

Evaluated on a synthetic dataset of 2,500 profiles, the classification pipeline achieved an accuracy of ~89.8% with an F1-score of 73.6%, and the regression pipeline achieved an $R^2$ score of 81.9% with a Mean Absolute Error (MAE) of ~3,570 INR. The models are deployed in a responsive Streamlit web application providing interactive credit evaluations, financial health diagnostics, and loan capacity simulations.

---

## CHAPTER 1: INTRODUCTION

### 1.1 Background
The growth of consumer finance has made Equated Monthly Installments (EMIs) the primary mechanism for purchasing vehicles, housing, and consumer electronics. For banks and Non-Banking Financial Companies (NBFCs), automated credit risk decisioning is vital to process large volumes of applications while minimizing Non-Performing Assets (NPAs). 

### 1.2 Problem Statement
Traditional lending criteria face two main challenges:
* **Over-reliance on Static Credit Scores**: Credit ratings are lagging indicators of credit behaviour, often not reflecting current monthly cash flow capacity.
* **Lack of Repayment Limit Guidelines**: Standard procedures assess eligibility (Yes/No) but do not recommend a concrete monthly budget limit. This leads to over-leverage, where borrowers take on multiple concurrent EMIs, triggering default cycles.

### 1.3 Project Objectives
* Generate a realistic credit profile dataset representing a variety of demographics.
* Construct an end-to-end Machine Learning pipeline utilizing preprocessing transformers.
* Train and benchmark classification models (Logistic Regression vs. Random Forest) for loan eligibility.
* Train and benchmark regression models (Linear Regression vs. Random Forest Regressor) for maximum affordable EMI.
* Deploy the serialized best models into an interactive, user-friendly Streamlit dashboard.

---

## CHAPTER 2: LITERATURE SURVEY
Credit risk evaluation has evolved from subjective qualitative reviews (using the 5 C's of credit: Character, Capacity, Capital, Collateral, and Conditions) to scoring models. 

1. **Credit Scoring Systems**: Altmann (1968) introduced the Z-score, demonstrating multivariate statistical techniques in bankruptcy prediction. Standard credit scoring systems (such as FICO) leverage logistic regression models built on historical repayment records to predict probability of default.
2. **Machine Learning in Credit Risk**: Recent studies have transitioned to ensemble methods. Random Forests, Gradient Boosted Trees (XGBoost, LightGBM), and Neural Networks have demonstrated high accuracy in default classification.
3. **Budget and Repayment Limits**: Standard bank guidelines use a Fixed Obligation to Income Ratio (FOIR), typically constraining total EMI outflow to 40%–50% of monthly income. The integration of continuous machine learning regressors to estimate this threshold dynamically remains an area of interest in financial technology (Fintech) applications.

---

## CHAPTER 3: SYSTEM DESIGN & METHODOLOGY

### 3.1 System Architecture
The pipeline consists of five major modules:
```
[User Inputs Form] -> [Preprocessing ColumnTransformer] -> [Pipeline Model Predictions] -> [Financial Dashboard Render]
                                                                        ^
                                                                        |
[Synthetic Data (2500 rows)] -> [Model Training & Comparison (Logistic Reg / Random Forest)] 
```

### 3.2 Dataset Formulation
A synthetic dataset of 2,500 customer rows was generated representing realistic financial distributions:
* **Monthly Income ($I$)**: Exponential distribution scaled between ₹15,000 and ₹250,000.
* **Monthly Expenses ($E$)**: Regulated as a ratio of income ($25\% - 70\%$) plus a random base value.
* **Credit Score ($CS$)**: Gaussian distribution centered at 650, clipped between 300 and 850.
* **Existing EMI ($EE$)**: Correlated with existing loan amounts, restricted by disposable income.
* **Employment Type**: Salaried (60%), Self-Employed (30%), Unemployed (10%).

#### 3.2.1 Eligibility Score (Classification Target)
The eligibility target is derived using a logit function representing credit score strength ($CS$), debt-to-income ($DTI$), employment risk, and dependent capacity:
$$\text{Logit} = -1.5 + 6.5 \times CS_{\text{scaled}} - 7.0 \times DTI + \beta_{\text{emp}} - 0.3 \times \text{Dependents}$$
Where:
* $CS_{\text{scaled}} = (CS - 300) / 550$
* $DTI = (E + EE) / I$
* $\beta_{\text{emp}} = 1.5$ for Salaried, $0.5$ for Self-Employed, $-5.0$ for Unemployed.

The classification target is:
$$\text{EMI\_Eligibility} = \begin{cases} 1 & \text{if } \sigma(\text{Logit}) \ge 0.5 \\ 0 & \text{otherwise} \end{cases}$$
With a $4\%$ random noise flip added to simulate real-world variance.

#### 3.2.2 Maximum Affordable EMI (Regression Target)
Repayment capacity is calculated as:
$$\text{Max\_Affordable\_EMI} = \max(0, (I \times \text{FOIR}_{\text{cap}}) - EE - (\text{Dependents} \times 1000)) + \epsilon$$
Where:
* $\text{FOIR}_{\text{cap}}$ is a capacity limit scaling from $10\%$ (poor credit/ineligible) up to $55\%$ (excellent credit score).
* $\epsilon \sim \mathcal{N}(0, 1500)$ represents random financial fluctuations.
* Unemployed profiles are set to ₹0.

### 3.3 Data Preprocessing & Pipelines
To prevent data leakage, a unified scikit-learn `Pipeline` was implemented.
* **Numerical Features**: Scaled using `StandardScaler` to zero mean and unit variance.
* **Categorical Features**: Encoded using `OneHotEncoder` (dropping the first category to avoid multicollinearity).
* **ColumnTransformer**: Combines scaling and encoding.

---

## CHAPTER 4: IMPLEMENTATION

### 4.1 Development Environment
* **Language**: Python 3.9+
* **Libraries**: Pandas, NumPy, Scikit-Learn, Joblib, Streamlit, Matplotlib, Seaborn
* **IDE**: Jupyter Notebook / VS Code

### 4.2 Model Training Configurations
1. **Logistic Regression**: Set with `max_iter=1000` and default L2 regularization.
2. **Random Forest Classifier**: Configuration of `n_estimators=100`, `max_depth=10`, and `random_state=42`.
3. **Linear Regression**: Standard Ordinary Least Squares (OLS) regression.
4. **Random Forest Regressor**: Configuration of `n_estimators=100`, `max_depth=12`, and `random_state=42`.

---

## CHAPTER 5: RESULTS & DISCUSSION

### 5.1 Classification Metrics
| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **89.80%** | **86.59%** | **63.96%** | **73.58%** |
| Random Forest Classifier | 88.40% | 90.77% | 53.15% | 67.05% |

*Logistic Regression outperformed Random Forest Classifier in balancing precision and recall (F1-Score), likely because the dataset's generative boundary contains linear relationships in logit space.*

### 5.2 Regression Metrics
| Model | RMSE (₹) | MAE (₹) | $R^2$ Score |
| :--- | :--- | :--- | :--- |
| Linear Regression | 8,756.99 | 6,161.54 | 71.36% |
| **Random Forest Regressor** | **6,968.81** | **3,576.68** | **81.86%** |

*Random Forest Regressor demonstrated superior prediction capability, capturing non-linear threshold effects (like credit score caps and dependent deductions) that Linear Regression struggled to model.*

### 5.3 Feature Importance
Feature importance plots from Random Forest models show:
1. **Credit Score** is the dominant factor determining **EMI Eligibility** (weight ~37%).
2. **Monthly Income** is the dominant factor determining **Maximum Affordable EMI** (weight ~51%).

---

## CHAPTER 6: CONCLUSION & FUTURE SCOPE
The **EMIPredict AI** system provides an end-to-end framework that helps borrowers evaluate their eligibility and repayment capacity. The use of unified scikit-learn pipelines ensures robust deployment without pipeline leakage, while the Streamlit application delivers a premium, highly interactive dashboard.

### Future Scope:
* **Alternative Data Scoring**: Incorporate utility bills and transactional SMS data for thinner-file scoring.
* **Deep Learning Models**: Benchmark against Multi-Layer Perceptrons (MLPs).
* **API Integration**: Integrate open banking protocols (such as Account Aggregators) for real-time customer income verification.
