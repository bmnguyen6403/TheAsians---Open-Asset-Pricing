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
    st.header("📘 Project Introduction")

    # Problem Statement & Core Question
    st.subheader("🎯 Objective")
    st.markdown("""
    This project explores advanced machine learning approaches for forecasting stock returns using:
    - **Survival analysis**
    - **Signal decay analysis**
    - **Composite signal construction**

    The goal is to build reliable predictive signals that remain robust across different market regimes.
    """)

    st.subheader("❓ Core Research Question")
    st.markdown("""
    > *Can many individually weak but statistically significant signals be combined into powerful predictors that outperform classic factors like value, momentum, and quality?*
    """)

    # Research Focus (Expandable)
    with st.expander("🔬 Research Focus"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("- Which signals help stocks avoid large crashes?")
            st.markdown("- How does predictive power fade over time?")
            st.markdown("- Do weighted signals outperform raw ones?")
        with col2:
            st.markdown("- How do models behave across market regimes?")
            st.markdown("- Can survival models reveal hidden risks?")

    # Hypotheses (Stylized with badges)
    st.subheader("📑 Hypotheses")
    st.markdown("""
    - <span style='color:#2E86C1'><strong>H1</strong></span>: Composite signals outperform individual ones.  
    - <span style='color:#28B463'><strong>H2</strong></span>: Weighted signals improve accuracy.  
    - <span style='color:#F1C40F'><strong>H3</strong></span>: Composite signals are more robust to decay.  
    - <span style='color:#AF7AC5'><strong>H4</strong></span>: Regime-aware models outperform static models.  
    """, unsafe_allow_html=True)

    st.success("Use the tabs above to explore each hypothesis and upload your data to analyze real results.")



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
        "Cumulative vs Predicted",
        "Feature Importance",
        "Survival Analysis - When will a stock die?",
        "Signal Decay",
        "Signal Engineering",
        "Regime-Aware Models"
    ])