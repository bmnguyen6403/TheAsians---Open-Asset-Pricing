# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Stock Return Hypotheses Explorer", layout="wide")
st.title("📊 Stock Return Prediction: Hypothesis-Based Analysis")

# Create tabs per hypothesis
tabs = st.tabs([
    "📈 H1: Composite Beats Individual",
    "📊 H2: Weighted Signals",
    "⏳ H3: Signal Decay",
    "🧠 H4: Regime-Aware Models"
])