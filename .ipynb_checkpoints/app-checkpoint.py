import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Set up the page layout
st.set_page_config(page_title="Stock Return Prediction Dashboard", layout="wide")
st.title("📊 Stock Return Prediction Dashboard")

# Top-level tabs
main_tabs = st.tabs(["📘 Introduction", "📊 Dataset", "📈 Analysis"])

# --- Introduction Tab ---
with main_tabs[0]:
    st.header("📘 Project Introduction")
    st.markdown("""
    **Problem Statement**  
    This project applies machine learning to forecast stock returns using survival analysis, signal decay, and composite signal construction. It focuses on building strong predictors from multiple financial signals and evaluating their performance in different market regimes.

    **Core Question**  
    Can many individually weak but statistically significant signals be combined into powerful predictors that outperform classic factors like value, momentum, and quality?

    **Research Focus**
    - Which signals help stocks avoid large crashes?
    - How does predictive power fade over time?
    - Do weighted signals outperform raw ones?
    - How do models behave across market regimes?
    - Can survival models reveal hidden risks?

    **Hypotheses**
    - **H1**: Composite signals outperform individual ones.  
    - **H2**: Weighted signals improve accuracy.  
    - **H3**: Composite signals are more robust to decay.  
    - **H4**: Regime-aware models outperform static models.
    """)

# --- Dataset Tab ---
with main_tabs[1]:
    st.header("📊 Dataset Overview")

    # Paths to the datasets
    signaldoc_path = "signaldoc_head10.csv"  # Adjust path if needed
    merged_path = "merged_df_head10.csv"  # Adjust path if needed

    # Load the first dataset - signaldoc_head10.csv
    signaldoc_df = pd.read_csv(signaldoc_path)
    st.subheader("📑 Signal Documentation")
    st.markdown("""
    **`signaldoc_head10.csv`**: This dataset contains a list of **financial signals** with their respective **signal names**, **quality ratings**, and **t-statistics**. It is used to identify the most statistically significant signals based on quality and reliability for stock prediction.
    """)
    st.dataframe(signaldoc_df.head())

    # Load the second dataset - merged_df_head10.csv
    merged_df = pd.read_csv(merged_path)
    st.subheader("📈 Merged Data (Filtered by Top Features)")
    st.markdown("""
    **`merged_df_head10.csv`**: This dataset is the **final merged dataset**, filtered based on the **top 20 features** selected from the signal quality and t-statistics. It contains **monthly returns** and **lagged explanatory variables**, making it ready for modeling and analysis of stock predictions.
    """)
    st.dataframe(merged_df.head())

# --- Hypothesis-Driven Analysis Tab ---
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

    # Sidebar will only be available when "Analysis" tab is selected
    if analysis_tab == "Cumulative vs Predicted":
        # Show the sidebar for model selection
        model_choice = st.sidebar.radio("Choose a Model", [
            "Composite Signal",
            "Linear Regression",
            "MLP Regressor",
            "Random Forest Regressor",
            "SVR Regressor",
            "XGBoost Regressor"
        ])

        st.subheader("📊 Cumulative and Predicted Returns")
        st.markdown("""
        This section compares cumulative returns across various models:
        - Composite Signal vs Actual Returns
        - Linear Regression
        - MLP Regressor
        - Random Forest Regressor
        - Support Vector Regressor (SVR)
        - XGBoost Regressor
        """)
        st.success("Use the sidebar to the left to navigate models")

        # Display corresponding image based on model choice
        if model_choice == "Composite Signal":
            st.image("composite_signal.png", caption="Composite Signal", use_container_width=True)
            st.markdown("""
            **Summary**:  
            The **Composite Signal** significantly underperforms compared to the **Actual Market** returns. The blue line represents the **Composite Signal**, which seems to lag behind the market, especially during periods of sharp growth. The green dashed line represents the **Actual Market**, which shows much higher returns, particularly after 2010. This highlights potential areas where the composite signal might be missing market trends or signals. The gap indicates that additional features or adjustments may be necessary for this model to better track the market.
            """)
        elif model_choice == "Linear Regression":
            st.image("LinearRegression_cumulative_return.png", caption="Linear Regression", use_container_width=True)
            st.markdown("""
            **Summary**:  
            **Linear Regression** captures the general trend of the **Actual Returns** well, but it struggles with larger fluctuations. The predicted returns (blue line) tend to smooth out market extremes, failing to track steep drops or spikes accurately. The model does a good job of reflecting the overall upward trend but lacks precision in volatile periods, such as the 2008 financial crisis. This suggests that linear models may not fully capture market complexities. Further improvements in model complexity, such as adding more features, might improve its predictive power.
            """)
        elif model_choice == "MLP Regressor":
            st.image("MLPRegressor_cumulative_return.png", caption="MLP Regressor", use_container_width=True)
            st.markdown("""
            **Summary**:  
            The **MLP Regressor** (Multi-Layer Perceptron) provides a better fit than **Linear Regression**, as it captures more of the volatility and trends in the actual returns. However, it still shows some lag in more turbulent periods, particularly around 2008, suggesting that the model might need further tuning. The MLP is smoother and more adaptable, but it still misses sharp fluctuations, which may limit its performance in extreme market scenarios. This model shows promise but needs further improvement in handling market volatility.
            """)
        elif model_choice == "Random Forest Regressor":
            st.image("RandomForestRegressor_cumulative_return.png", caption="Random Forest Regressor", use_container_width=True)
            st.markdown("""
            **Summary**:  
            The **Random Forest Regressor** performs well, following the **Actual Returns** (green line) with relatively few discrepancies. It tracks the major market trends and fluctuations, including during the 2008 crisis, better than both **Linear Regression** and **MLP** models. The blue line (predicted values) shows reasonable alignment with the market, but it slightly lags during periods of rapid change. Overall, **Random Forest** is more robust, though it could still be improved for sudden market shocks.
            """)
        elif model_choice == "SVR Regressor":
            st.image("SVR_cumulative_return.png", caption="Support Vector Regressor", use_container_width=True)
            st.markdown("""
            **Summary**:  
            The **SVR model** seems to struggle with capturing the real sharp downturns (e.g., 2008) but generally follows the market trend well. The predictions from **SVR** appear smoother and less responsive to extreme market volatility, which might be due to how **SVR** handles outliers and its regularization methods. While it captures the overall trend, it does not react sharply to extreme movements. This could indicate a need for a more dynamic model or tuning to capture short-term shocks.
            """)
        elif model_choice == "XGBoost Regressor":
            st.image("XGBRegressor_cumulative_return.png", caption="XGBoost Regressor", use_container_width=True)
            st.markdown("""
            **Summary**:  
            The **XGBoost Regressor** performs well overall, tracking the actual returns more accurately, especially during volatile periods. It shows superior predictive power compared to other models, such as **Linear Regression** and **SVR**, making it a stronger choice for financial modeling. However, the model might still struggle with predicting extreme market shifts, like the sudden crash in 2008. Despite this, **XGBoost** is a top performer in terms of predictive accuracy and handling complex market conditions.
            """)
