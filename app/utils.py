import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Cache model loading to optimize performance
@st.cache_resource
def load_models():
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    clf_path = os.path.join(models_dir, 'best_classifier.joblib')
    reg_path = os.path.join(models_dir, 'best_regressor.joblib')
    clf_stats_path = os.path.join(models_dir, 'classifier_stats.joblib')
    reg_stats_path = os.path.join(models_dir, 'regressor_stats.joblib')
    importance_path = os.path.join(models_dir, 'feature_importance.joblib')
    
    models = {}
    if os.path.exists(clf_path):
        models['classifier'] = joblib.load(clf_path)
    if os.path.exists(reg_path):
        models['regressor'] = joblib.load(reg_path)
    if os.path.exists(clf_stats_path):
        models['clf_stats'] = joblib.load(clf_stats_path)
    if os.path.exists(reg_stats_path):
        models['reg_stats'] = joblib.load(reg_stats_path)
    if os.path.exists(importance_path):
        models['feature_importance'] = joblib.load(importance_path)
        
    return models

def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    
    /* Header and title styling */
    h1, h2, h3 {
        color: #f8fafc !important;
        font-weight: 700 !important;
    }
    
    .main-title {
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        text-align: center;
    }
    
    .sub-title {
        color: #94a3b8;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Custom cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        border-color: rgba(99, 102, 241, 0.3);
    }
    
    .kpi-container {
        display: flex;
        justify-content: space-between;
        gap: 15px;
        flex-wrap: wrap;
    }
    
    .kpi-card {
        flex: 1;
        min-width: 200px;
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 5px 0;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .kpi-label {
        color: #94a3b8;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Result styling */
    .result-eligible {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.25) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.1);
    }
    
    .result-ineligible {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.25) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.1);
    }
    
    .result-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 10px;
    }
    
    .badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-top: 5px;
    }
    
    .badge-low { background-color: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .badge-medium { background-color: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .badge-high { background-color: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.4); }
    
    /* Input Form Styling */
    div[data-baseweb="input"], div[data-baseweb="select"] {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #6366f1 0%, #4f46e5 100%) !important;
        color: white !important;
        border: none !important;
        padding: 10px 24px !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6) !important;
    }
    
    /* Radial Health Score representation */
    .health-score-container {
        margin: 20px 0;
        text-align: center;
    }
    
    .health-bar-bg {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        height: 12px;
        width: 100%;
        overflow: hidden;
        margin-top: 8px;
    }
    
    .health-bar-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease-in-out;
    }
    </style>
    """, unsafe_allow_html=True)

def render_kpi(label, value):
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """

def render_eligibility(eligible, confidence, risk_level):
    if eligible == 1:
        bg_class = "result-eligible"
        title = "ELIGIBLE"
        text_color = "#34d399"
        desc = "Congratulations! You meet the standard criteria for EMI-based financing."
    else:
        bg_class = "result-ineligible"
        title = "NOT ELIGIBLE"
        text_color = "#f87171"
        desc = "Sorry, you do not meet the eligibility thresholds based on your credit profile."
        
    risk_badge = ""
    if risk_level.lower() == "low":
        risk_badge = '<span class="badge badge-low">Low Risk</span>'
    elif risk_level.lower() == "medium":
        risk_badge = '<span class="badge badge-medium">Medium Risk</span>'
    else:
        risk_badge = '<span class="badge badge-high">High Risk</span>'
        
    st.markdown(f"""
    <div class="{bg_class}">
        <div style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.1em; color: #94a3b8; margin-bottom: 5px;">Eligibility Prediction</div>
        <div class="result-title" style="color: {text_color};">{title}</div>
        <div style="font-size: 1.1rem; font-weight: 500; margin-bottom: 12px;">Confidence: {confidence:.1f}%</div>
        <div style="margin-bottom: 15px;">{risk_badge}</div>
        <p style="color: #cbd5e1; margin: 0; font-size: 0.95rem;">{desc}</p>
    </div>
    """, unsafe_allow_html=True)

def render_affordability(emi, health_score, recommendation):
    # Determine health score color
    if health_score >= 70:
        health_color = "linear-gradient(90deg, #34d399, #10b981)"
        health_text = "Strong"
        health_text_color = "#34d399"
    elif health_score >= 45:
        health_color = "linear-gradient(90deg, #fbbf24, #f59e0b)"
        health_text = "Moderate"
        health_text_color = "#fbbf24"
    else:
        health_color = "linear-gradient(90deg, #f87171, #ef4444)"
        health_text = "Weak"
        health_text_color = "#f87171"
        
    formatted_emi = f"₹{emi:,.2f}" if emi > 0 else "₹0"
    
    st.markdown(f"""
    <div class="glass-card" style="border-left: 5px solid #6366f1;">
        <div style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.1em; color: #94a3b8; margin-bottom: 5px;">Recommended Maximum EMI</div>
        <div style="font-size: 3rem; font-weight: 800; color: #818cf8; margin-bottom: 15px;">{formatted_emi} <span style="font-size: 1.2rem; font-weight: 400; color: #94a3b8;">/ month</span></div>
        
        <div class="health-score-container">
            <div style="display: flex; justify-content: space-between; font-size: 0.95rem; font-weight: 600;">
                <span>Financial Health Score</span>
                <span style="color: {health_text_color};">{health_score:.1f}% ({health_text})</span>
            </div>
            <div class="health-bar-bg">
                <div class="health-bar-fill" style="width: {health_score}%; background: {health_color};"></div>
            </div>
        </div>
        
        <div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.08);">
            <div style="font-weight: 600; color: #f8fafc; margin-bottom: 5px;">Recommendation Details:</div>
            <p style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.5; margin: 0;">{recommendation}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
