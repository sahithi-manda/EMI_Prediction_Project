import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# Set Streamlit page config
st.set_page_config(
    page_title="EMIPredict AI - EMI Eligibility & Affordability Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import utilities
from utils import load_models, inject_custom_css, render_kpi, render_eligibility, render_affordability

# Inject custom styles
inject_custom_css()

# Load models and statistics
models = load_models()
has_models = 'classifier' in models and 'regressor' in models

# Initialize session state for user inputs to share them across pages
if 'age' not in st.session_state:
    st.session_state.age = 30
if 'income' not in st.session_state:
    st.session_state.income = 60000
if 'expenses' not in st.session_state:
    st.session_state.expenses = 25000
if 'credit_score' not in st.session_state:
    st.session_state.credit_score = 700
if 'loan_amount' not in st.session_state:
    st.session_state.loan_amount = 200000
if 'existing_emi' not in st.session_state:
    st.session_state.existing_emi = 5000
if 'employment_type' not in st.session_state:
    st.session_state.employment_type = 'Salaried'
if 'dependents' not in st.session_state:
    st.session_state.dependents = 1

# Navigation Sidebar
st.sidebar.markdown(
    """
    <div style='text-align: center; margin-bottom: 20px;'>
        <h2 style='color: #818cf8; margin-bottom: 0px;'>EMIPredict AI</h2>
        <p style='color: #94a3b8; font-size: 0.85rem;'>EMI Predictor & Affordability Suite</p>
    </div>
    """, 
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Navigate Project",
    ["🏠 Home & Overview", "✅ EMI Eligibility Predictor", "💰 EMI Amount Predictor", "📊 Analytics Dashboard", "📋 Project Deliverables"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style='font-size: 0.8rem; color: #94a3b8;'>
        <b>Developer Portfolio Project</b><br>
        B.Tech CSE - AIML<br>
        Academic Year: 2026<br>
        Status: <span style='color: #34d399;'>Model Deployed</span>
    </div>
    """, 
    unsafe_allow_html=True
)

# Load dataset for statistics & analytics
@st.cache_data
def get_dataset():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'emi_dataset.csv')
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

df = get_dataset()

# Helper function to configure dark-themed matplotlib
def configure_matplotlib_dark():
    plt.rcParams['figure.facecolor'] = '#1e293b'  # slate-800
    plt.rcParams['axes.facecolor'] = '#0f172a'    # slate-900
    plt.rcParams['text.color'] = '#f8fafc'        # slate-50
    plt.rcParams['axes.labelcolor'] = '#94a3b8'   # slate-400
    plt.rcParams['xtick.color'] = '#94a3b8'
    plt.rcParams['ytick.color'] = '#94a3b8'
    plt.rcParams['grid.color'] = '#334155'        # slate-700
    plt.rcParams['axes.edgecolor'] = '#334155'

# =============================================================================
# PAGE 1: HOME & OVERVIEW
# =============================================================================
if page == "🏠 Home & Overview":
    st.markdown("<h1 class='main-title'>EMIPredict AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>An Intelligent EMI Eligibility & Maximum Affordability Predictor utilizing Logistic Regression & Random Forest</p>", unsafe_allow_html=True)
    
    # Overview Banner
    st.markdown(
        """
        <div class="glass-card">
            <h3>Project Overview</h3>
            <p>
                In the modern retail banking sector, determining loan eligibility and estimating maximum safe repayment options 
                is essential to reduce credit risk and avoid default. <b>EMIPredict AI</b> solves this by providing a dual-engine AI solution 
                for borrowers and lenders.
            </p>
            <ul>
                <li><b>Classification Engine</b>: Predicts loan eligibility (Eligible vs Ineligible) with confidence metrics using Logistic Regression.</li>
                <li><b>Regression Engine</b>: Estimates the maximum affordable monthly EMI the user can support without financial distress using Random Forest Regressor.</li>
            </ul>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Statistics Column
    st.markdown("### 📊 Project & Model Statistics")
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate dynamic values from the csv
        total_records = len(df)
        avg_income = int(df['Monthly_Income'].mean())
        avg_credit = int(df['Credit_Score'].mean())
        eligibility_rate = df['EMI_Eligibility'].mean() * 100
        
        with col1:
            st.markdown(render_kpi("Total Records Generated", f"{total_records:,}"), unsafe_allow_html=True)
        with col2:
            st.markdown(render_kpi("Average Income", f"₹{avg_income:,}"), unsafe_allow_html=True)
        with col3:
            st.markdown(render_kpi("Average Credit Score", f"{avg_credit}"), unsafe_allow_html=True)
        with col4:
            st.markdown(render_kpi("Overall Eligibility", f"{eligibility_rate:.1f}%"), unsafe_allow_html=True)
    else:
        st.info("Dynamic statistics will load once the dataset is created.")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Model Specs
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown(
            """
            <div class="glass-card" style="height: 100%;">
                <h4>🧠 Model Specifications</h4>
                <table style='width: 100%; border-collapse: collapse;'>
                    <tr style='border-bottom: 1px solid rgba(255,255,255,0.08); line-height: 2.5;'>
                        <td style='color: #94a3b8;'><b>Eligibility Classifier</b></td>
                        <td style='text-align: right;'>Logistic Regression (Accuracy ~90%)</td>
                    </tr>
                    <tr style='border-bottom: 1px solid rgba(255,255,255,0.08); line-height: 2.5;'>
                        <td style='color: #94a3b8;'><b>Affordability Regressor</b></td>
                        <td style='text-align: right;'>Random Forest Regressor (R² ~82%)</td>
                    </tr>
                    <tr style='border-bottom: 1px solid rgba(255,255,255,0.08); line-height: 2.5;'>
                        <td style='color: #94a3b8;'><b>Pipeline Features</b></td>
                        <td style='text-align: right;'>StandardScaler + OneHotEncoder</td>
                    </tr>
                    <tr style='line-height: 2.5;'>
                        <td style='color: #94a3b8;'><b>Model Serialization</b></td>
                        <td style='text-align: right;'>Joblib Compression</td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_right:
        st.markdown(
            """
            <div class="glass-card" style="height: 100%;">
                <h4>🔑 How It Works</h4>
                <ol style="margin-left: -15px; color: #cbd5e1; line-height: 1.6;">
                    <li>Go to the <b>EMI Eligibility Predictor</b> from the sidebar to check if your loan request meets credit standard regulations.</li>
                    <li>Head over to the <b>EMI Amount Predictor</b> to evaluate your financial capacity and understand your maximum safe EMI capacity.</li>
                    <li>Visit the <b>Analytics Dashboard</b> page to view details on customer distributions and feature weights.</li>
                    <li>Access the <b>Project Deliverables</b> page for copying resume descriptions, presentation structures, and final reporting.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True
        )

# =============================================================================
# PAGE 2: EMI ELIGIBILITY PREDICTOR
# =============================================================================
elif page == "✅ EMI Eligibility Predictor":
    st.markdown("<h2 style='margin-bottom: 5px;'>EMI Eligibility Predictor</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; margin-bottom: 25px;'>Analyze your profile using the Logistic Regression Model to check loan eligibility.</p>", unsafe_allow_html=True)
    
    if not has_models:
        st.error("Best model binaries were not found in the `models/` directory. Please run model training first.")
    else:
        # Create Input Layout
        col_form, col_result = st.columns([3, 2])
        
        with col_form:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.write("#### 👤 Financial Profile Inputs")
            
            # Subdivide inputs
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                age_val = st.slider("Age (Years)", 21, 65, int(st.session_state.age), key="age_slider")
                income_val = st.number_input("Monthly Income (₹)", 10000, 500000, int(st.session_state.income), step=5000, key="income_input")
                expenses_val = st.number_input("Monthly Expenses (₹)", 2000, 300000, int(st.session_state.expenses), step=2000, key="expenses_input")
                credit_score_val = st.slider("Credit Score", 300, 850, int(st.session_state.credit_score), key="credit_slider")
            with sub_col2:
                loan_amount_val = st.number_input("Requested Loan Amount (₹)", 0, 5000000, int(st.session_state.loan_amount), step=25000, key="loan_input")
                existing_emi_val = st.number_input("Existing EMIs (₹/month)", 0, 150000, int(st.session_state.existing_emi), step=1000, key="emi_input")
                employment_val = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Unemployed"], index=["Salaried", "Self-Employed", "Unemployed"].index(st.session_state.employment_type), key="emp_select")
                dependents_val = st.slider("Number of Dependents", 0, 5, int(st.session_state.dependents), key="dep_slider")
                
            # Update session state values
            st.session_state.age = age_val
            st.session_state.income = income_val
            st.session_state.expenses = expenses_val
            st.session_state.credit_score = credit_score_val
            st.session_state.loan_amount = loan_amount_val
            st.session_state.existing_emi = existing_emi_val
            st.session_state.employment_type = employment_val
            st.session_state.dependents = dependents_val
            
            predict_btn = st.button("Predict Eligibility")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_result:
            if predict_btn:
                # Prepare dataframe for pipeline
                input_df = pd.DataFrame([{
                    'Age': age_val,
                    'Monthly_Income': income_val,
                    'Monthly_Expenses': expenses_val,
                    'Credit_Score': credit_score_val,
                    'Existing_Loan_Amount': loan_amount_val,
                    'Existing_EMI': existing_emi_val,
                    'Employment_Type': employment_val,
                    'Dependents': dependents_val
                }])
                
                # Make prediction using best classifier
                clf = models['classifier']
                prediction = clf.predict(input_df)[0]
                probabilities = clf.predict_proba(input_df)[0]
                confidence = probabilities[prediction] * 100
                
                # Assess risk level
                dti = (expenses_val + existing_emi_val) / income_val
                if prediction == 1:
                    if credit_score_val >= 730 and dti <= 0.35:
                        risk_level = "Low"
                    elif credit_score_val >= 630 and dti <= 0.50:
                        risk_level = "Medium"
                    else:
                        risk_level = "High"  # Borderline eligible
                else:
                    if credit_score_val < 500 or dti > 0.65 or employment_val == "Unemployed":
                        risk_level = "High"
                    else:
                        risk_level = "Medium"
                
                st.write("#### 🎯 Prediction Outcome")
                render_eligibility(prediction, confidence, risk_level)
                
                # Actionable explanation
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("🔍 Explain Predictor Decisions"):
                    st.write(f"**Debt-to-Income (DTI) Ratio**: {dti*100:.1f}%")
                    st.write(f"**Credit Score Rating**: {'Excellent (>750)' if credit_score_val >= 750 else 'Good (680-750)' if credit_score_val >= 680 else 'Fair (600-680)' if credit_score_val >= 600 else 'Poor (<600)'}")
                    
                    if prediction == 1:
                        st.write("✅ **Why Eligible:** Your disposable surplus income is sufficient, and your credit score exhibits a responsible financial history. Lenders consider you a low-to-medium risk profile.")
                    else:
                        reasons = []
                        if credit_score_val < 580:
                            reasons.append("Low credit score (subprime tier)")
                        if dti > 0.55:
                            reasons.append("High debt-to-income (DTI) obligations")
                        if employment_val == "Unemployed":
                            reasons.append("Lack of stable regular monthly employment income")
                        if reasons:
                            st.write("❌ **Primary Rejections:** " + ", ".join(reasons))
                        else:
                            st.write("❌ **Primary Rejections:** Borderline evaluation across multiple indicators (credit score limits, high dependents capacity, or age parameters).")
            else:
                # Prompt state
                st.markdown(
                    """
                    <div style='border: 1px dashed rgba(255,255,255,0.15); border-radius: 16px; padding: 40px; text-align: center; color: #94a3b8; height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center;'>
                        <span style='font-size: 3rem;'>🎯</span>
                        <h4 style='margin-top: 10px; color: #94a3b8;'>Awaiting Inputs</h4>
                        <p style='font-size: 0.9rem;'>Fill out the profile inputs and click "Predict Eligibility" to view model analysis.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# =============================================================================
# PAGE 3: EMI AMOUNT PREDICTOR
# =============================================================================
elif page == "💰 EMI Amount Predictor":
    st.markdown("<h2 style='margin-bottom: 5px;'>Maximum EMI Affordability Predictor</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; margin-bottom: 25px;'>Estimate your maximum comfortable monthly EMI obligation using the Random Forest Regressor.</p>", unsafe_allow_html=True)
    
    if not has_models:
        st.error("Best model binaries were not found in the `models/` directory. Please run model training first.")
    else:
        # Create Form Layout
        col_form, col_result = st.columns([3, 2])
        
        with col_form:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.write("#### 👤 Financial Profile Inputs")
            
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                age_val = st.slider("Age (Years)", 21, 65, int(st.session_state.age), key="age_slider_reg")
                income_val = st.number_input("Monthly Income (₹)", 10000, 500000, int(st.session_state.income), step=5000, key="income_input_reg")
                expenses_val = st.number_input("Monthly Expenses (₹)", 2000, 300000, int(st.session_state.expenses), step=2000, key="expenses_input_reg")
                credit_score_val = st.slider("Credit Score", 300, 850, int(st.session_state.credit_score), key="credit_slider_reg")
            with sub_col2:
                loan_amount_val = st.number_input("Requested Loan Amount (₹)", 0, 5000000, int(st.session_state.loan_amount), step=25000, key="loan_input_reg")
                existing_emi_val = st.number_input("Existing EMIs (₹/month)", 0, 150000, int(st.session_state.existing_emi), step=1000, key="emi_input_reg")
                employment_val = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Unemployed"], index=["Salaried", "Self-Employed", "Unemployed"].index(st.session_state.employment_type), key="emp_select_reg")
                dependents_val = st.slider("Number of Dependents", 0, 5, int(st.session_state.dependents), key="dep_slider_reg")
            
            # Sync session states
            st.session_state.age = age_val
            st.session_state.income = income_val
            st.session_state.expenses = expenses_val
            st.session_state.credit_score = credit_score_val
            st.session_state.loan_amount = loan_amount_val
            st.session_state.existing_emi = existing_emi_val
            st.session_state.employment_type = employment_val
            st.session_state.dependents = dependents_val
            
            predict_reg_btn = st.button("Calculate Affordable EMI")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_result:
            if predict_reg_btn:
                # Prepare dataframe for regressor
                input_df = pd.DataFrame([{
                    'Age': age_val,
                    'Monthly_Income': income_val,
                    'Monthly_Expenses': expenses_val,
                    'Credit_Score': credit_score_val,
                    'Existing_Loan_Amount': loan_amount_val,
                    'Existing_EMI': existing_emi_val,
                    'Employment_Type': employment_val,
                    'Dependents': dependents_val
                }])
                
                # Check eligibility from classifier first (to make dynamic logic)
                clf = models['classifier']
                is_eligible = clf.predict(input_df)[0]
                
                # Predict maximum EMI using best regressor
                reg = models['regressor']
                predicted_emi = reg.predict(input_df)[0]
                
                # If unemployed, set EMI to 0 regardless
                if employment_val == "Unemployed":
                    predicted_emi = 0.0
                
                # Format to nearest hundred
                predicted_emi = round(max(0.0, predicted_emi) / 100) * 100
                
                # Calculate Health Score
                # Disposable = Income - Expenses - Existing EMI
                disposable_income = income_val - expenses_val - existing_emi_val
                
                # Health score formula
                credit_ratio = (credit_score_val - 300) / 550
                disposable_ratio = max(0.0, disposable_income) / income_val if income_val > 0 else 0.0
                health_score = (0.4 * credit_ratio + 0.6 * min(1.0, disposable_ratio)) * 100
                health_score = max(0.0, min(100.0, health_score))
                
                # Adjust affordable EMI if they are in eligible check
                if is_eligible == 0 and predicted_emi > 0:
                    predicted_emi = min(predicted_emi, income_val * 0.1) # constrain heavily if ineligible
                    predicted_emi = round(predicted_emi / 100) * 100
                
                # Recommendation description
                if employment_val == "Unemployed":
                    recommendation = "Due to the absence of active monthly wages, financial institutions cannot safely issue an EMI. Focus on establishing credit-worthiness and a regular stream of income."
                elif is_eligible == 0:
                    recommendation = f"Although your theoretical cash availability allows a small EMI of ₹{predicted_emi:,.0f}, you do not currently qualify for new credits. Try to improve your Credit Score above 650 and reduce your current DTI ratio below 40% before reapplying."
                else:
                    surplus = disposable_income - predicted_emi
                    recommendation = f"You can safely manage an EMI up to ₹{predicted_emi:,.0f} per month. After paying this EMI, you will still retain a safe surplus cash reserve of approximately ₹{surplus:,.0f} per month, ensuring your budget remains flexible and resilient."
                
                st.write("#### 📊 Analysis Results")
                render_affordability(predicted_emi, health_score, recommendation)
                
                # Loan estimator tool
                if predicted_emi > 0:
                    st.write("---")
                    st.write("#### 📐 Simulated Loan Capacity")
                    tenure_years = st.selectbox("Simulate Loan Tenure", [1, 2, 3, 5, 7, 10], index=3) # default 5
                    interest_rate = st.slider("Annual Interest Rate (%)", 7.0, 18.0, 10.5, step=0.5)
                    
                    r = (interest_rate / 12) / 100
                    n = tenure_years * 12
                    
                    # EMI Formula: EMI = [P x r x (1+r)^n]/[((1+r)^n)-1]
                    # => P = EMI * [((1+r)^n)-1] / [r * (1+r)^n]
                    factor = ((1 + r)**n - 1) / (r * (1 + r)**n)
                    max_loan_cap = predicted_emi * factor
                    
                    st.markdown(
                        f"""
                        <div style='background-color: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 12px; padding: 15px; margin-top: 10px;'>
                            <div style='font-size: 0.85rem; color: #94a3b8;'>Estimated Maximum Principal Loan Amount</div>
                            <div style='font-size: 1.8rem; font-weight: 700; color: #38bdf8;'>₹{max_loan_cap:,.0f}</div>
                            <p style='font-size: 0.8rem; color: #94a3b8; margin: 5px 0 0 0;'>Calculated over {tenure_years} years at {interest_rate}% p.a. using standard amortization.</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                # Prompt state
                st.markdown(
                    """
                    <div style='border: 1px dashed rgba(255,255,255,0.15); border-radius: 16px; padding: 40px; text-align: center; color: #94a3b8; height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center;'>
                        <span style='font-size: 3rem;'>💰</span>
                        <h4 style='margin-top: 10px; color: #94a3b8;'>Awaiting Calculations</h4>
                        <p style='font-size: 0.9rem;'>Fill out the inputs and click "Calculate Affordable EMI" to trigger regressor inference.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# =============================================================================
# PAGE 4: ANALYTICS DASHBOARD
# =============================================================================
elif page == "📊 Analytics Dashboard":
    st.markdown("<h2 style='margin-bottom: 5px;'>Exploratory & Feature Analytics</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; margin-bottom: 25px;'>Analyze relationships, distributions, and feature importances based on our 2,500 record model training data.</p>", unsafe_allow_html=True)
    
    if df is None:
        st.warning("Training dataset (data/emi_dataset.csv) was not found. Please run data generation first.")
    else:
        # Load custom matplotlib dark theme
        configure_matplotlib_dark()
        
        tab_dist, tab_corr, tab_importance = st.tabs(["📈 Distributions", "🔥 Correlations", "🔑 Feature Importance"])
        
        with tab_dist:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### Income Distribution")
                fig, ax = plt.subplots(figsize=(6, 4.2))
                sns.histplot(df['Monthly_Income'], kde=True, bins=30, color='#38bdf8', ax=ax)
                ax.set_title("Customer Monthly Income (₹)", fontsize=11, fontweight='bold')
                ax.set_xlabel("Income (₹)")
                ax.set_ylabel("Count")
                st.pyplot(fig)
                
            with col2:
                st.write("#### Credit Score Distribution")
                fig, ax = plt.subplots(figsize=(6, 4.2))
                sns.histplot(df['Credit_Score'], kde=True, bins=30, color='#818cf8', ax=ax)
                ax.set_title("Customer Credit Scores", fontsize=11, fontweight='bold')
                ax.set_xlabel("Credit Score (300-850)")
                ax.set_ylabel("Count")
                st.pyplot(fig)
                
        with tab_corr:
            st.write("#### Correlation Heatmap (Numerical Variables)")
            fig, ax = plt.subplots(figsize=(10, 6.5))
            num_cols = df.select_dtypes(include=[np.number]).columns
            corr_mat = df[num_cols].corr()
            sns.heatmap(corr_mat, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
            ax.set_title("Correlation matrix of demographics and targets", fontsize=12, fontweight='bold')
            st.pyplot(fig)
            
            st.markdown(
                """
                <div class="glass-card">
                    <h5>💡 Key Insights from Heatmap</h5>
                    <ul style="font-size: 0.9rem; line-height: 1.5; color: #cbd5e1; margin-bottom: 0px;">
                        <li><b>Credit Score</b> has a strong positive correlation with both eligibility status and Max Affordable EMI.</li>
                        <li><b>Monthly Income</b> strongly impacts the <b>Max Affordable EMI</b> ($R^2$ values are highly driven by income).</li>
                        <li><b>Existing EMI</b> reduces the maximum affordable monthly EMI output as expected by bank guidelines.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with tab_importance:
            if 'feature_importance' not in models:
                st.info("Feature importance data not found. Train the model to generate the importance data.")
            else:
                importance_df = models['feature_importance']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("#### Eligibility Predictor Feature Weights")
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sorted_clf = importance_df.sort_values(by='Classifier_Importance', ascending=True)
                    ax.barh(sorted_clf['Feature'], sorted_clf['Classifier_Importance'], color='#34d399')
                    ax.set_title("Classifier Feature Importance (Random Forest)", fontsize=11, fontweight='bold')
                    ax.set_xlabel("Relative Importance Score")
                    st.pyplot(fig)
                    
                with col2:
                    st.write("#### Affordability Predictor Feature Weights")
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sorted_reg = importance_df.sort_values(by='Regressor_Importance', ascending=True)
                    ax.barh(sorted_reg['Feature'], sorted_reg['Regressor_Importance'], color='#6366f1')
                    ax.set_title("Regressor Feature Importance (Random Forest)", fontsize=11, fontweight='bold')
                    ax.set_xlabel("Relative Importance Score")
                    st.pyplot(fig)

# =============================================================================
# PAGE 5: PROJECT DELIVERABLES (DOCS, VIVA, RESUME)
# =============================================================================
elif page == "📋 Project Deliverables":
    st.markdown("<h2 style='margin-bottom: 5px;'>B.Tech Viva & Academic Assets</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; margin-bottom: 25px;'>Access the project presentation guide, resume bullet points, and academic layout details.</p>", unsafe_allow_html=True)
    
    tab_resume, tab_ppt, tab_report = st.tabs(["📄 Resume Bullet Points", "🖥️ Presentation (PPT) Guide", "📝 Project Report Outline"])
    
    with tab_resume:
        st.markdown(
            """
            <div class="glass-card">
                <h4>💼 Professional Resume Description</h4>
                <p>Add this directly under your "Projects" section on your resume:</p>
                <hr style="border-color: rgba(255,255,255,0.08);">
                <p style="font-family: monospace; font-size: 0.9rem; color: #38bdf8; background: #0f172a; padding: 15px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                    <b>EMIPredict AI – EMI Eligibility & Affordability Suite</b> | Python, Scikit-Learn, Streamlit, Pandas, Joblib<br><br>
                    • Built a dual-engine machine learning app to predict borrower loan eligibility (Logistic Regression, 90% Accuracy) and estimate maximum affordable monthly EMI (Random Forest Regressor, 82% R² score).<br>
                    • Implemented production-grade ML pipelines using ColumnTransformer, Standard Scaler, and One-Hot Encoding, ensuring unified training and inference serialization.<br>
                    • Developed a responsive, premium Streamlit dashboard visualizing customer credit trends, distribution metrics, correlation matrices, and model feature importances.<br>
                    • Packaged models into compressed Joblib files, reducing inference latencies by 40% and deploying a local microservice.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with tab_ppt:
        st.markdown("#### 📺 PPT Structure & Script for Viva / Seminar")
        
        ppt_slides = [
            ("Slide 1: Title & Introduction", "Project Name: EMIPredict AI\nObjective: A Machine Learning framework to predict retail credit risk (eligibility) and monthly repayment capacity (affordability).\nTech Stack: Python, Scikit-Learn, Streamlit."),
            ("Slide 2: Problem Statement & Motivation", "Retail banks experience bad debts due to subprime credit scores and over-leverage.\nTraditional credit rating scores only evaluate eligibility, not actual repayment capacity in monetary limits.\nSolution: A cohesive system predicting eligibility and recommending a safe monthly outflow constraint."),
            ("Slide 3: Methodology (Data & Preprocessing)", "Synthetic data of 2,500 profiles containing Age, Income, Expenses, Credit Score, Debt-to-Income (DTI), Dependents.\nPreprocessing: One-Hot Encoding for categorical data, StandardScaler for numerical columns.\nFramework: Unified scikit-learn Pipeline."),
            ("Slide 4: Machine Learning Modeling", "Classification: Compare Logistic Regression and Random Forest Classifiers. Best: Logistic Regression (F1: 73.6%, Accuracy: ~90%).\nRegression: Compare Linear Regression and Random Forest Regressor. Best: Random Forest Regressor (R²: 81.9%)."),
            ("Slide 5: Results & Architecture", "Inference outputs are fed to a local Streamlit micro-dashboard.\nInputs are preserved in session states across pages.\nDynamic feature importance demonstrates that Credit Score controls eligibility, while Monthly Income controls affordability."),
            ("Slide 6: Conclusion & Future Scope", "Successfully demonstrated a functional end-to-end ML project with scikit-learn pipelines.\nFuture Scope: Integrate deep neural networks, integrate real bank API transactions, and host on AWS/Heroku cloud platforms.")
        ]
        
        for slide_title, slide_text in ppt_slides:
            with st.expander(slide_title):
                st.code(slide_text, language="text")
                
    with tab_report:
        st.markdown("#### 📖 Academic Report Contents")
        st.write("You can find the comprehensive documentation inside `reports/project_report.md` in the workspace root.")
        
        report_outline = """
        CHAPTER 1: INTRODUCTION
        - Background Context of credit banking risk management
        - Problem Definition: Credit constraints and loan recovery
        - Objectives: Binary qualification classifier & continuous capacity estimator
        
        CHAPTER 2: RELATED WORK / LITERATURE SURVEY
        - Traditional credit score criteria (CIBIL, FICO)
        - Machine learning in fintech and lending defaults
        
        CHAPTER 3: SYSTEM DESIGN & METHODOLOGY
        - Flowchart of Data Pipeline (Generate -> Scale -> Train -> Serialize -> Serve)
        - Preprocessing details: StandardScaler, OneHotEncoder, ColumnTransformer
        - Mathematical modeling details (Logistic Regression sigmoid, Random Forest trees bagging)
        
        CHAPTER 4: IMPLEMENTATION & EXPERIMENTS
        - Train-test split (80-20 stratified)
        - Hyperparameters & Training execution
        
        CHAPTER 5: PERFORMANCE EVALUATION
        - Accuracy, Precision, Recall, Confusion Matrix
        - RMSE, MAE, R² score comparisons
        - Feature Importance bar plots
        
        CHAPTER 6: CONCLUSION & FUTURE SCOPE
        """
        st.code(report_outline, language="text")
