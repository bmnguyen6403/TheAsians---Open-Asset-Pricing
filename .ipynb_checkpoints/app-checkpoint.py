jupyter nbconvert --to script best_model_10_random_good.ipynb
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shap
from lifelines import CoxPHFitter
from best_model_10_random_good.ipnyb import (
    signal_df,
    composite_df,
    model_performance_df,
    feature_importance_df,
    survival_df,
    decay_df,
    regime_df,
)



st.set_page_config(page_title="Stock Return Prediction Dashboard", layout="wide")
st.title("📊 Stock Return Prediction Full Dashboard")

# Create Tabs
tabs = st.tabs([
    "Signal Selection", "Composite Signal", "Model Performance",
    "Feature Importance", "Survival Analysis", "Signal Decay",
    "Signal Engineering", "Regime Detection"
])

# Signal Selection Tab
with tabs[0]:
    st.header("🔍 Signal Selection")
    good_signals = signal_df[(signal_df["Quality"] == "good") & (signal_df["T-Stat"] > 3)]
    st.dataframe(good_signals)
    st.metric(label="Number of Good Signals", value=len(good_signals))

# Composite Signal Tab
with tabs[1]:
    st.header("⚡ Composite Signal vs Actual Returns")
    fig, ax = plt.subplots()
    ax.plot(composite_df["Date"], composite_df["Composite"], label="Composite Prediction")
    ax.plot(composite_df["Date"], composite_df["Actual"], label="Actual Market", linestyle="--")
    ax.legend()
    ax.set_title("Composite vs Actual Returns")
    st.pyplot(fig)

# Model Performance Tab
with tabs[2]:
    st.header("🏆 Model Performance")
    st.dataframe(model_performance_df)
    fig, ax = plt.subplots()
    ax.plot(composite_df["Date"], composite_df["Composite"], label="Predicted")
    ax.plot(composite_df["Date"], composite_df["Actual"], label="Actual", linestyle="--")
    ax.legend()
    ax.set_title("Cumulative Actual vs Predicted")
    st.pyplot(fig)

# Feature Importance Tab
with tabs[3]:
    st.header("🔍 Top 20 Feature Importance")
    fig, ax = plt.subplots()
    ax.barh(feature_importance_df["Feature"], feature_importance_df["Importance"])
    ax.set_title("Feature Importance from XGBoost")
    st.pyplot(fig)

# Survival Analysis Tab
with tabs[4]:
    st.header("💀 Survival Analysis")
    st.dataframe(survival_df)
    st.markdown("*Model: Cox Proportional Hazard using financial signals*")
    fig, ax = plt.subplots()
    ax.hist(survival_df['Duration'], bins=10)
    ax.set_title("Survival Durations")
    st.pyplot(fig)

# Signal Decay Tab
with tabs[5]:
    st.header("📉 Signal Decay Over Time")
    fig, ax = plt.subplots()
    ax.plot(decay_df["Months"], decay_df["Decay"], marker='o')
    ax.set_xlabel("Months Forward")
    ax.set_ylabel("Predictive Strength")
    ax.set_title("Signal Decay")
    st.pyplot(fig)

# Signal Engineering Tab
with tabs[6]:
    st.header("🛠️ Signal Engineering")
    st.markdown("*Features built from OpenAP signals by transformations, interactions, ratios.*")

# Regime Detection Tab
with tabs[7]:
    st.header("📈 Regime Detection")
    st.dataframe(regime_df)
    fig, ax = plt.subplots()
    regime_colors = regime_df["Regime"].map({"Bull": "green", "Bear": "red"})
    ax.scatter(regime_df["Date"], np.random.randn(len(regime_df)).cumsum(), c=regime_colors)
    ax.set_title("Market Regime Over Time")
    st.pyplot(fig)

st.sidebar.info("Built by The Asians Team | FIN 377 Project")