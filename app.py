# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Stock Return Prediction Dashboard", layout="wide")
st.title("📊 Stock Return Prediction Dashboard")

# Top-level tabs
main_tabs = st.tabs(["📘 Introduction", "📊 Dataset", "📈 Analysis"])

# --- Introduction ---
with main_tabs[0]:
    st.header("📘 Project Overview")
    st.markdown("""
    Welcome to the Stock Return Prediction Dashboard!

    This project explores several financial hypotheses:
    - Can composite signals beat individual predictors?
    - Do weighted predictors improve accuracy?
    - How do signals decay over time?
    - Can we estimate when a stock might 'die' using survival analysis?
    - Can engineered signals reveal new insights?
    - Do regime-aware models outperform static ones?

    Navigate through the tabs to explore each question.
    """)

# --- Dataset ---
with main_tabs[1]:
    st.header("📊 Upload Dataset")
    uploaded = st.file_uploader("Upload your CSV file with financial signals", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.session_state["df"] = df
        st.success("File uploaded successfully!")
        st.dataframe(df.head())
    elif "df" in st.session_state:
        st.info("Using previously uploaded dataset.")
        st.dataframe(st.session_state["df"].head())
    else:
        st.warning("Please upload a CSV file to continue.")


with main_tabs[2]:
    st.header("📈 Hypothesis-Driven Analysis")

    analysis_tab = st.selectbox("Choose Analysis", [
        "Composite Beats Individual?",
        "Feature Importance",
        "Survival Analysis - When will a stock die?",
        "Signal Decay",
        "Signal Engineering",
        "Regime-Aware Models"
    ])