import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Config ---
st.set_page_config(page_title="Signal Engineering Results", layout="wide")

# --- Title ---
st.title("\U0001F4C8 Signal Engineering Project Overview")

# --- Introduction Text ---
st.markdown("""
This app presents the results of a project that evaluated financial signals using machine learning and signal engineering.
It compares the top 10 original signals versus top 10 engineered signals based on their feature importance in predicting returns.
""")

# --- Data: Manually Inserted Based on Your Results ---
original_data = {
    'Feature': ['retConglomerate', 'CustomerMomentum', 'betaVIX', 'IntanEP', 'MomSeason16YrPlus',
                'roaq', 'IndMom', 'TrendFactor', 'MomOffSeason06YrPlus', 'DelDRC'],
    'Importance': [0.2712, 0.1244, 0.0800, 0.0502, 0.0481, 0.0441, 0.0417, 0.0374, 0.0358, 0.0284]
}

engineered_data = {
    'Feature': ['retConglomerate_MA3', 'MomSeason16YrPlus_MA3', 'CustomerMomentum_MA3', 'TrendFactor_MA6',
                'TrendFactor_MA3', 'retConglomerate_MA6', 'IndMom_MA6', 'IndMom_MA3', 'betaVIX_MA6', 'betaVIX_MA3'],
    'Importance': [0.0995, 0.0749, 0.0573, 0.0478, 0.0383, 0.0357, 0.0352, 0.0352, 0.0323, 0.0311]
}

original_df = pd.DataFrame(original_data)
engineered_df = pd.DataFrame(engineered_data)

# --- Sidebar ---
st.sidebar.header("Choose View")
view = st.sidebar.radio("Select Feature Set:", ("Original Signals", "Engineered Signals", "Comparison"))

# --- Main Area ---
if view == "Original Signals":
    st.subheader("Top 10 Original Features by Importance")
    st.dataframe(original_df)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=original_df, palette='Blues_r', ax=ax)
    ax.set_title("Top 10 Original Feature Importances")
    st.pyplot(fig)

elif view == "Engineered Signals":
    st.subheader("Top 10 Engineered Features by Importance")
    st.dataframe(engineered_df)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=engineered_df, palette='Greens_r', ax=ax)
    ax.set_title("Top 10 Engineered Feature Importances")
    st.pyplot(fig)

else:
    st.subheader("Comparison of Original vs Engineered Feature Importances")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original Features**")
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        sns.barplot(x='Importance', y='Feature', data=original_df, palette='Blues_r', ax=ax1)
        ax1.set_title("Original Features")
        st.pyplot(fig1)

    with col2:
        st.markdown("**Engineered Features**")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.barplot(x='Importance', y='Feature', data=engineered_df, palette='Greens_r', ax=ax2)
        ax2.set_title("Engineered Features")
        st.pyplot(fig2)

# --- Footer ---
st.markdown("""
---
Made with \U0001F9E0 by [Your Name]
""")
