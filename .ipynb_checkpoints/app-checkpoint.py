# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Stock Return Hypotheses Explorer", layout="wide")
st.title("📊 Stock Return Prediction: Hypothesis-Based Analysis")

# Create tabs per hypothesis
tabs = st.tabs([
    "Introduction",
    "Composite Beats Individual?",
    "Feature Importance",
    "Survival Analysis - When will a stock die?",
    "Signal Decay",
    "Signal Engineering"
    "Regime-Aware Models"
])