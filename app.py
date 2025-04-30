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
    signaldoc_path = "dashboard_ref/signaldoc_head10.csv"  # Adjust path if needed
    merged_path = "dashboard_ref/merged_df_head10.csv"  # Adjust path if needed

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
            st.image("dashboard_ref/composite_signal.png", caption="Composite Signal", use_container_width=True)
            st.markdown("""
            **Summary**:  
            **Composite Signal** significantly underperforms compared to the **Actual Market** returns, as indicated by the gap between the predicted (blue line) and actual (green dashed line) values. The **Composite Signal** lags the market, especially during periods of sharp growth, suggesting that the model might be missing critical market trends. The actual returns, particularly after 2010, surpass the predictions made by the composite signal. This performance discrepancy highlights that additional features, adjustments, or improvements in model complexity are necessary to better capture market dynamics and improve predictive accuracy.\n
            The **low correlation** suggests that the model is not capturing the market's behavior accurately, and the **Sharpe Ratio** indicates that while the model has positive returns, they are not well-adjusted for risk. Further tuning and potentially the introduction of more relevant features could improve the model's performance.
            """)
        elif model_choice == "Linear Regression":
            st.image("dashboard_ref/LinearRegression_cumulative_return.png", caption="Linear Regression", use_container_width=True)
            st.markdown("""
            **Summary**:  
            The **Linear Regression** model captures the general trend of the **Actual Returns** well, but it struggles with larger fluctuations, particularly during volatile market periods like the 2008 financial crisis. The blue line representing the predicted returns tends to smooth out market extremes, failing to track sharp drops or spikes accurately. While **Linear Regression** reflects the overall upward trend of the market, it lacks precision in more turbulent times, such as during market crashes. This suggests that linear models may not fully capture the complexities and nonlinearities of the market, and adding more features could improve the model's predictive power.\n
            The **correlation** shows a reasonable fit to the market data, though there is room for improvement. The **Sharpe Ratio** suggests the model's returns are not well-adjusted for volatility, and the **T-Statistic** indicates that the model's significance is moderate.
            """)
        elif model_choice == "MLP Regressor":
            st.image("dashboard_ref/MLPRegressor_cumulative_return.png", caption="MLP Regressor", use_container_width=True)
            st.markdown("""
            **Summary**:  
            The **MLP Regressor** provides a better fit compared to **Linear Regression**, as it captures more of the volatility and trends in the actual returns. It adapts well to the changing dynamics of the market, but still exhibits some lag during extreme periods, particularly in 2008. While the **MLP Regressor** does a better job of handling market fluctuations compared to simpler models, it still misses sharp market movements, which can limit its effectiveness during times of market stress. The model shows promise, but fine-tuning its architecture or adding more features could improve its performance in volatile markets.\n
            With a high **correlation**, the model provides a strong fit to the actual market returns. The **Sharpe Ratio** reflects an improved risk-adjusted return, though there’s room for better performance during market shocks. The **T-Statistic** is reasonably significant, indicating that the model’s predictions are statistically meaningful.
            """)
        elif model_choice == "Random Forest Regressor":
            st.image("dashboard_ref/RandomForestRegressor_cumulative_return.png", caption="Random Forest Regressor", use_container_width=True)
            st.markdown("""
            **Summary**:  
            The **Random Forest Regressor** performs well in tracking the major trends of the **Actual Returns**, including significant market fluctuations such as the 2008 financial crisis. The predicted values (blue line) show a reasonable alignment with the actual returns (green line), though it slightly lags during periods of rapid market change. Despite this, the **Random Forest** model demonstrates higher robustness compared to **Linear Regression** and **MLP**, and it handles sudden market shifts better. Overall, **Random Forest** is a strong performer, but could still benefit from improvements to better capture extreme market movements.\n
            The **correlation** of 1.000000 indicates a perfect fit, which is expected given how **Random Forest** works by aggregating multiple decision trees. However, the **Sharpe Ratio** suggests that the model’s returns are not very well-adjusted for risk, and further tuning could improve its performance.
            """)
        elif model_choice == "SVR Regressor":
            st.image("dashboard_ref/SVR_cumulative_return.png", caption="Support Vector Regressor", use_container_width=True)
            st.markdown("""
            **Summary**:  
            The **SVR Regressor** seems to struggle with capturing sharp downturns, particularly during significant market crises like the 2008 financial crisis. While it follows the general market trend well, the predicted returns appear smoother and less responsive to extreme market movements. This suggests that **SVR** may be too conservative and not sufficiently sensitive to market shocks. The model could be improved by adjusting its regularization parameters or incorporating more dynamic features to capture sudden market fluctuations.\n
            The **correlation** indicates a fairly good fit, but the model may be too smooth to react sharply to volatile periods. The **Sharpe Ratio** suggests that **SVR** offers a decent risk-adjusted return, but could be further optimized to capture extreme market changes.
            """)
        elif model_choice == "XGBoost Regressor":
            st.image("dashboard_ref/XGBRegressor_cumulative_return.png", caption="XGBoost Regressor", use_container_width=True)
            st.markdown("""
            **Summary**:  
            The **XGBoost Regressor** performs well overall, tracking the **Actual Returns** more accurately than most other models, especially during volatile periods. It provides superior predictive power, handling complex market conditions effectively. While it performs well overall, the model may still face challenges in predicting sudden market shifts, like the sharp crash in 2008. Nevertheless, **XGBoost** is a top performer in terms of predictive accuracy and managing market dynamics, making it one of the most reliable models for stock return prediction.\n
            The **correlation** demonstrates a perfect fit to the market returns, making **XGBoost** one of the strongest models for stock prediction. However, the **Sharpe Ratio** suggests that the model's returns could be more adjusted for risk, and additional tuning may improve its performance during extreme market conditions.
            """)
# Load and display the "model_summary.csv" table
        model_summary_df = pd.read_csv("dashboard_ref/model_summary.csv"  )
        st.subheader("📊 Model Summary")
        st.markdown("Here is the summary of various models' performance metrics.")
        st.dataframe(model_summary_df)
        st.markdown("""
        **Random Forest** and **XGBoost** models are the most promising in terms of **predictive accuracy** (high correlation with actual returns) but still need improvements in **risk-adjusted returns** (Sharpe Ratio).\n
        **Linear Regression** and **SVR** struggle to capture extreme market fluctuations and have lower **Sharpe Ratios**, suggesting they need further enhancements.\n
        The **Composite Signal** lags behind the actual market and may benefit from additional features or adjustments to better capture market trends.
                    """)

    if analysis_tab == "Feature Importance":
        st.header("📊 Feature Importance")
        st.markdown("""
        **Top 20 Feature Importance:**  
        The bar plot ranks the top 20 financial signals by their relative importance in the predictive model.

        ### Key Observations:
        - **XFIN** is by far the most important feature, contributing the most to the model’s predictions.
        - **TrendFactor** and **NetEquityFinance** follow, but with noticeably lower impact compared to XFIN.
        - Other moderately important features include **TotalAccruals**, **grcapx**, and **RDS**.
        - Features like **DelLTI**, **IntanEP**, and **betaVIX** contribute the least among the top 20 but still add marginal predictive value.

        ### Implications:
        - **XFIN** plays a dominant role and should be a focal point for model interpretation and strategy construction.
        - Mid-importance features may offer opportunities for complementary signal interactions and diversification within composite signals.
        """)

    if analysis_tab == "Survival Analysis - When will a stock die?":
        surv_analysis_df = pd.read_csv("dashboard_ref/surv_analysis.csv")
        st.header("📊 Survival Analysis")
        st.dataframe(surv_analysis_df)
        st.markdown("""
The following analysis presents the results from a Cox Proportional Hazards model. We categorize signals based on whether their hazard ratios (**exp(coef)**) are greater than or less than 1, indicating whether they are associated with increased hazard (higher risk) or positive survival (growth). **p-values** greater than 0.05 indicate statistical significance.

A hazard ratio (**exp(coef)**) greater than 1 suggests the signal increases the risk of being delisted (higher "death risk"), while a hazard ratio less than 1 suggests it reduces the risk ("still alive").

### Signals Associated with Increased Hazard (Higher Risk)

| Acronym | coef | exp(coef) | p-value |
|:--------|-----:|----------:|--------:|
| NetEquityFinance | 0.6431 | 1.9023 | 0.5094 |
| TotalAccruals | 0.3062 | 1.3582 | 0.0726 |
| grcapx | 0.0321 | 1.0326 | 0.0770 |
| NetDebtFinance | 1.0352 | 2.8158 | 0.2352 |
| InvestPPEInv | 0.4018 | 1.4945 | 0.1162 |

**Analysis:**
- These signals have hazard ratios greater than 1, indicating a directional association with increased risk.
- However, all of them have p-values above 0.05, meaning their association with hazard is statistically weak.
- **Notable Signals:**
  - **NetEquityFinance** shows a relatively high hazard ratio (1.9023), suggesting firms with more equity financing activity could face higher risk, although the weak p-value suggests this should be interpreted cautiously.
  - **NetDebtFinance** has an even larger hazard ratio (2.8158), indicating a possible increased risk from debt financing behaviors, albeit without statistical confirmation.
- Despite being less reliable individually, these signals might still provide valuable information when combined or interacted with other signals through feature engineering.

### Signals Associated with Reduced Hazard (Positive Growth)

| Acronym | coef | exp(coef) | p-value |
|:--------|-----:|----------:|--------:|
| XFIN | -1.3061 | 0.2709 | 0.2035 |
| TrendFactor | -0.2044 | 0.8151 | 0.0658 |
| RDS | -0.000016 | 0.999984 | 0.2238 |
| hire | -0.1547 | 0.8567 | 0.2147 |
| MomSeason16YrPlus | -0.2352 | 0.7904 | 0.2950 |
| IndMom | -0.0091 | 0.9910 | 0.9415 |
| betaVIX | -3.5647 | 0.0283 | 0.1155 |

**Analysis:**
- These signals have hazard ratios less than 1, suggesting a protective or growth-enhancing effect.
- Although their p-values are not statistically strong, their directional indication toward positive survival could be explored further.
- **Notable Signals:**
  - **XFIN** (exp(coef) = 0.2709) indicates a potentially strong protective effect from external financing net flows, although not statistically significant.
  - **TrendFactor** suggests firms aligned with trending financial factors may experience modest survival advantages.
  - **betaVIX** (exp(coef) = 0.0283) implies an extreme protective effect for stocks less sensitive to market volatility (VIX), highlighting a potential defensive property worth future investigation.
- These signals could contribute to a composite low-risk score if properly validated or transformed in future modeling.

### Implications for Modeling

Signals showing weaker individual significance could still be useful in aggregated models or after applying feature engineering. By combining both risk-enhancing and protective signals, we can design more nuanced models that better account for multiple dimensions of survival dynamics.
        """)
