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

st.sidebar.info("Built by The Asians Team | FIN 377 Project")
