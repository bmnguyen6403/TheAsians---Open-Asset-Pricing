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

    # Adding team member images and names
    st.markdown("### Meet the Team")
    
    # Creating columns for images and names of the team members
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.image("path_to_image/Minh_Nguyen.jpg", width=100)  # Adjust image path accordingly
        st.markdown("**Minh Nguyen**")

    with col2:
        st.image("path_to_image/Lam_Nguyen.jpg", width=100)  # Adjust image path accordingly
        st.markdown("**Lam Nguyen**")

    with col3:
        st.image("path_to_image/Mia_Le.jpg", width=100)  # Adjust image path accordingly
        st.markdown("**Mia Le**")

    with col4:
        st.image("path_to_image/Alice_Zhang.jpg", width=100)  # Adjust image path accordingly
        st.markdown("**Alice Zhang**")


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
    st.dataframe(signaldoc_df)

    # Load the second dataset - merged_df_head10.csv
    merged_df = pd.read_csv(merged_path)
    st.subheader("📈 Merged Data (Filtered by Top Features)")
    st.markdown("""
    **`merged_df_head10.csv`**: This dataset is the **final merged dataset**, filtered based on the **top 20 features** selected from the signal quality and t-statistics. It contains **monthly returns** and **lagged explanatory variables**, making it ready for modeling and analysis of stock predictions.
    """)
    st.dataframe(merged_df)

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

    if analysis_tab == "Cumulative vs Predicted":
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

        # Create subtabs for each model
        model_tabs = st.tabs([
            "Composite Signal",
            "Linear Regression",
            "MLP Regressor",
            "Random Forest Regressor",
            "SVR Regressor",
            "XGBoost Regressor"
        ])

        with model_tabs[0]:
            st.image("dashboard_ref/composite_signal.png", caption="Composite Signal", use_container_width=True)
            st.markdown("""
            **Composite Signal Explanation**

The composite signal is constructed by aggregating multiple individual signals based on their standardized returns. First, the returns of each signal are normalized using their Z-scores, which helps bring them to a comparable scale by subtracting the mean and dividing by the standard deviation. This ensures that all signals contribute equally, regardless of their original scale or volatility. After normalization, each signal is weighted according to its T-statistic, which is a measure of statistical significance. The T-statistic reflects how strongly each signal correlates with the target variable, and signals with higher T-statistics are given more weight in the final composite.

The weighted signals are then aggregated using a dot product, where the normalized returns are multiplied by their respective weights, and the results are summed. The final composite signal represents a combination of the selected signals, with more importance placed on those that are statistically significant. This approach enhances the predictive power of the signal by considering the most relevant and robust financial indicators, providing a more stable and reliable signal for modeling or further analysis.\n

            """)
            st.markdown("""
            **Summary**:  
            **Composite Signal** significantly underperforms compared to the **Actual Market** returns, as indicated by the gap between the predicted (blue line) and actual (green dashed line) values. The **Composite Signal** lags the market, especially during periods of sharp growth, suggesting that the model might be missing critical market trends. The actual returns, particularly after 2010, surpass the predictions made by the composite signal. This performance discrepancy highlights that additional features, adjustments, or improvements in model complexity are necessary to better capture market dynamics and improve predictive accuracy.\n
            The **low correlation** suggests that the model is not capturing the market's behavior accurately, and the **Sharpe Ratio** indicates that while the model has positive returns, they are not well-adjusted for risk. Further tuning and potentially the introduction of more relevant features could improve the model's performance.
            """)
        
        with model_tabs[1]:
            st.image("dashboard_ref/LinearRegression_cumulative_return.png", caption="Linear Regression", use_container_width=True)
            st.markdown("""
            **Summary**:  
            The **Linear Regression** model captures the general trend of the **Actual Returns** well, but it struggles with larger fluctuations, particularly during volatile market periods like the 2008 financial crisis. The blue line representing the predicted returns tends to smooth out market extremes, failing to track sharp drops or spikes accurately. While **Linear Regression** reflects the overall upward trend of the market, it lacks precision in more turbulent times, such as during market crashes. This suggests that linear models may not fully capture the complexities and nonlinearities of the market, and adding more features could improve the model's predictive power.\n
            The **correlation** shows a reasonable fit to the market data, though there is room for improvement. The **Sharpe Ratio** suggests the model's returns are not well-adjusted for volatility, and the **T-Statistic** indicates that the model's significance is moderate.
            """)
        with model_tabs[2]:
            st.image("dashboard_ref/MLPRegressor_cumulative_return.png", caption="MLP Regressor", use_container_width=True)
            st.markdown("""
            **Summary**:  
            The **MLP Regressor** provides a better fit compared to **Linear Regression**, as it captures more of the volatility and trends in the actual returns. It adapts well to the changing dynamics of the market, but still exhibits some lag during extreme periods, particularly in 2008. While the **MLP Regressor** does a better job of handling market fluctuations compared to simpler models, it still misses sharp market movements, which can limit its effectiveness during times of market stress. The model shows promise, but fine-tuning its architecture or adding more features could improve its performance in volatile markets.\n
            With a high **correlation**, the model provides a strong fit to the actual market returns. The **Sharpe Ratio** reflects an improved risk-adjusted return, though there’s room for better performance during market shocks. The **T-Statistic** is reasonably significant, indicating that the model’s predictions are statistically meaningful.
            """)
        
        with model_tabs[3]:
            st.image("dashboard_ref/RandomForestRegressor_cumulative_return.png", caption="Random Forest Regressor", use_container_width=True)
            st.markdown("""
            **Summary**:  
            The **Random Forest Regressor** performs well in tracking the major trends of the **Actual Returns**, including significant market fluctuations such as the 2008 financial crisis. The predicted values (blue line) show a reasonable alignment with the actual returns (green line), though it slightly lags during periods of rapid market change. Despite this, the **Random Forest** model demonstrates higher robustness compared to **Linear Regression** and **MLP**, and it handles sudden market shifts better. Overall, **Random Forest** is a strong performer, but could still benefit from improvements to better capture extreme market movements.\n
            The **correlation** of 1.000000 indicates a perfect fit, which is expected given how **Random Forest** works by aggregating multiple decision trees. However, the **Sharpe Ratio** suggests that the model’s returns are not very well-adjusted for risk, and further tuning could improve its performance.
            """)
        with model_tabs[4]:
            st.image("dashboard_ref/SVR_cumulative_return.png", caption="Support Vector Regressor", use_container_width=True)
            st.markdown("""
            **Summary**:  
            The **SVR Regressor** seems to struggle with capturing sharp downturns, particularly during significant market crises like the 2008 financial crisis. While it follows the general market trend well, the predicted returns appear smoother and less responsive to extreme market movements. This suggests that **SVR** may be too conservative and not sufficiently sensitive to market shocks. The model could be improved by adjusting its regularization parameters or incorporating more dynamic features to capture sudden market fluctuations.\n
            The **correlation** indicates a fairly good fit, but the model may be too smooth to react sharply to volatile periods. The **Sharpe Ratio** suggests that **SVR** offers a decent risk-adjusted return, but could be further optimized to capture extreme market changes.
            """)
        
        with model_tabs[5]:
            st.image("dashboard_ref/XGBRegressor_cumulative_return.png", caption="XGBoost Regressor", use_container_width=True)
            st.markdown("""
            **Summary**:  
            The **XGBoost Regressor** performs well overall, tracking the **Actual Returns** more accurately than most other models, especially during volatile periods. It provides superior predictive power, handling complex market conditions effectively. While it performs well overall, the model may still face challenges in predicting sudden market shifts, like the sharp crash in 2008. Nevertheless, **XGBoost** is a top performer in terms of predictive accuracy and managing market dynamics, making it one of the most reliable models for stock return prediction.\n
            The **correlation** demonstrates a perfect fit to the market returns, making **XGBoost** one of the strongest models for stock prediction. However, the **Sharpe Ratio** suggests that the model's returns could be more adjusted for risk, and additional tuning may improve its performance during extreme market conditions.
            """)
        
        # Load and display the "model_summary.csv" table
        model_summary_df = pd.read_csv("dashboard_ref/model_summary.csv")
        st.subheader("📊 Model Summary")
        st.markdown("Here is the summary of various models' performance metrics.")
        st.dataframe(model_summary_df)
        st.markdown("""
        **Random Forest** and **XGBoost** models are the most promising in terms of **predictive accuracy** (high correlation with actual returns) but still need improvements in **risk-adjusted returns** (Sharpe Ratio).\n
        **Linear Regression** and **SVR** struggle to capture extreme market fluctuations and have lower **Sharpe Ratios**, suggesting they need further enhancements.\n
        The **Composite Signal** lags behind the actual market and may benefit from additional features or adjustments to better capture market trends.
                    """)
    
    if analysis_tab == "Feature Importance":
        st.image("dashboard_ref/top20feature.png")
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
## Signal Analysis Summary

The following analysis presents the results from a **Cox Proportional Hazards** model. We categorize signals based on whether their hazard ratios (**exp(coef)**) are greater than or less than 1, indicating whether they are associated with increased hazard (higher risk) or positive survival (growth). **p-values** less than 0.1 indicate signals with statistically significant associations.

A hazard ratio (**exp(coef)**) greater than 1 suggests the signal increases the risk of being delisted (higher "death risk"), while a hazard ratio less than 1 suggests it reduces the risk ("still alive").

### Signals Associated with Increased Hazard (Higher Risk)

| Acronym               | coef    | exp(coef) | p-value |
|:----------------------|--------:|----------:|--------:|
| TotalAccruals         | 0.3062  | 1.3582    | 0.0726  |
| grcapx                | 0.0321  | 1.0326    | 0.0770  |
| MomOffSeason06YrPlus  | 3.4294  | 30.8578   | 0.0058  |
| roaq                  | 2.2916  | 9.8905    | 0.0004  |
| DelDRC                | 2.5459  | 12.7550   | 0.0053  |
| CustomerMomentum      | 0.4514  | 1.5705    | 0.0495  |
| IntanEP               | 0.4173  | 1.5178    | 5.8193e-08 |
| DelLTI                | 1.2929  | 3.6434    | 0.0241  |

**Analysis:**
- **TotalAccruals (exp(coef) = 1.3582):**
  - **Meaning**: Accruals are accounting adjustments made for items that are earned or owed but not yet received or paid. A high accrual could indicate that a firm is managing its earnings aggressively.
  - **Effect**: A hazard ratio greater than 1 means that **higher accruals** are associated with an increased risk of being delisted or facing financial distress.

- **grcapx (exp(coef) = 1.0326):**
  - **Meaning**: Growth rate of capital expenditures (capx) relative to some benchmark, typically to measure how much a firm is investing in its operations.
  - **Effect**: A hazard ratio greater than 1 implies that **higher growth in capital expenditures** may increase the risk of delisting or financial failure. High investments may reflect high risk, especially if not yielding immediate returns.

- **MomOffSeason06YrPlus (exp(coef) = 30.8578):**
  - **Meaning**: Measures the average return in the off-season months over a long period (more than 6 years). This might capture how stocks perform in less active periods.
  - **Effect**: A high hazard ratio suggests that firms with good returns during off-seasons may have a **higher risk** of being delisted, possibly indicating volatility or market timing issues.

- **roaq (exp(coef) = 9.8905):**
  - **Meaning**: Return on assets (ROA) is a measure of how profitable a company is relative to its total assets.
  - **Effect**: A hazard ratio greater than 1 indicates that **higher return on assets** could be associated with a higher risk of delisting. This may seem counterintuitive, but firms with high ROA may be operating in high-risk sectors or aggressive growth modes.

- **DelDRC (exp(coef) = 12.7550):**
  - **Meaning**: Annual change in deferred revenue (drc) scaled by some factor. Deferred revenue is money a company has received for services it has not yet performed.
  - **Effect**: A hazard ratio greater than 1 indicates that **increased deferred revenue** could indicate higher financial risk or instability.

- **CustomerMomentum (exp(coef) = 1.5705):**
  - **Meaning**: Measures the strength of customer momentum (the firm’s ability to retain and expand its customer base).
  - **Effect**: Despite its statistical significance, a hazard ratio greater than 1 suggests that **strong customer momentum** might correlate with increased risk of failure, possibly due to unsustainable growth or changing market conditions.

- **IntanEP (exp(coef) = 1.5178):**
  - **Meaning**: The model uses earnings per share (EPS) adjusted for intangible assets to account for a firm's value from non-physical assets like patents and brand value.
  - **Effect**: A hazard ratio greater than 1 implies that firms with high intangible assets could be more **vulnerable to financial distress**. This might reflect reliance on non-tangible factors, which can be volatile.

- **DelLTI (exp(coef) = 3.6434):**
  - **Meaning**: Represents the difference in investment and advances (ivao) between periods.
  - **Effect**: A hazard ratio greater than 1 indicates that **higher differences in long-term investments** (such as shifts in how firms allocate their capital) can be linked to greater financial risk or potential failure.

### Signals Associated with Reduced Hazard (Positive Growth)

| Acronym             | coef    | exp(coef) | p-value |
|:--------------------|--------:|----------:|--------:|
| TrendFactor         | -0.2044 | 0.8151    | 0.0658  |
| InvGrowth           | -0.0454 | 0.9556    | 5.0839e-05 |
| retConglomerate     | -1.1588 | 0.3139    | 2.9556e-06 |
| betaVIX             | -3.5647 | 0.0283    | 0.1155  |

**Analysis:**
- **TrendFactor (exp(coef) = 0.8151):**
  - **Meaning**: Represents a factor that aligns a firm’s performance with overall market trends. Firms aligned with positive trends often outperform others.
  - **Effect**: A hazard ratio less than 1 suggests that **firms aligned with positive trends** may have a **lower risk** of delisting. It may indicate growth opportunities, though the p-value is borderline significant.

- **InvGrowth (exp(coef) = 0.9556):**
  - **Meaning**: Investment growth over time, typically adjusted for inflation or other factors. It can represent how a firm is investing to expand its business.
  - **Effect**: A hazard ratio less than 1 means that **lower investment growth** may reduce the risk of delisting, suggesting that **firms with controlled or modest investments** are less likely to face financial issues.

- **retConglomerate (exp(coef) = 0.3139):**
  - **Meaning**: Identifies conglomerate firms (those with diverse business units across sectors) and their relative financial stability.
  - **Effect**: A hazard ratio significantly less than 1 implies that **conglomerate firms** are less likely to be delisted, possibly due to their diversified nature, which provides stability and reduces risk.

- **betaVIX (exp(coef) = 0.0283):**
  - **Meaning**: Measures the firm’s sensitivity to changes in the VIX (Volatility Index), which reflects market volatility.
  - **Effect**: A hazard ratio much less than 1 suggests that **firms with low sensitivity to volatility** are less likely to face financial distress. This indicates that less volatile companies tend to have **lower risk** of delisting.
        """)
        
    if analysis_tab == "Signal Decay":
        st.header("📊 Signal Decay")
        st.image("dashboard_ref/decay_graph.png")
        st.markdown("""
## Graph Overview
The graph illustrates the signal decay patterns for the top five financial signals, evaluated based on their Spearman Rank Correlation (IC) with future stock returns over 1-month, 3-month, and 6-month horizons.

## Key Observations

| Signal | Behavior | Interpretation |
|:--------|:---------|:----------------|
| **XFIN** | IC **increases** over time | Indicates that external financing activity becomes **more predictive** over longer holding periods. |
| **TrendFactor** | IC **decreases moderately** | Maintains reasonable predictive power across 6 months, suitable for **medium-term strategies**. |
| **NetEquityFinance** | IC **declines steadily** | Predictive strength fades over time; better for **short-term stock selection**. |
| **roaq** | IC **declines sharply** | Very strong short-term predictor, but effectiveness **diminishes rapidly** after 1 month. |
| **retConglomerate** | IC **consistently weakens** | Shows the lowest predictive power overall, with a steady decay over time. |

## Overall Conclusions
- **XFIN** is the most promising for **long-term investment horizons**.
- **roaq** offers strong opportunities for **short-term trading**.
- **TrendFactor** is relatively stable, supporting **medium-term portfolio construction**.
- **NetEquityFinance** and **retConglomerate** show weaker and diminishing predictive abilities, and may require combination with other signals for effective use.

## Practical Implications
Understanding signal decay helps align trading strategies with the appropriate investment horizon. Combining fast-decaying signals (like **roaq**) with slower-decaying signals (like **XFIN**) could enhance portfolio stability and performance across different timeframes.
        """)
        
    if analysis_tab == "Signal Engineering":
        st.header("📊 Signal Engineering")
        importance_eng = pd.read_csv("dashboard_ref/importance_eng.csv")
        orig_importance = pd.read_csv("dashboard_ref/orig_importance.csv")
        
        # Create two columns for side-by-side display
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Importance Engineering")
            st.dataframe(importance_eng)

        with col2:
            st.subheader("Original Importance")
            st.dataframe(orig_importance)

        st.markdown("""
        - *Original Signals:* The single most important original feature was retConglomerate, contributing over 27% of the model's decision power, followed by CustomerMomentum, betaVIX, and IntanEP.
- *Engineered Features:* The top engineered features were mainly moving averages of the strongest original signals, such as retConglomerate_MA3, MomSeason16YrPlus_MA3, and CustomerMomentum_MA3. The importance was spread more evenly among the top engineered features, with no single feature dominating the model.

## Analytical Insights
The comparison revealed that while some signals, like retConglomerate, were extremely powerful even in their raw form, many others such as CustomerMomentum, TrendFactor, and betaVIX improved significantly after engineering transformations like smoothing. Engineered features produced a more balanced feature importance distribution, reducing over-reliance on any single predictor, which is desirable for model robustness.

The fact that moving averages improved performance suggests that financial signals often contain substantial short-term noise that can be mitigated through simple smoothing techniques. Meanwhile, interaction terms and ratios may capture deeper relationships between financial metrics.

## Conclusion
Through this project, I demonstrated that thoughtful signal engineering can materially improve model performance in financial prediction tasks. By combining top raw signals with the best engineered transformations, a hybrid model could achieve better generalization, higher Sharpe ratios, and more stable predictive performance. This highlights the value of both strong signal selection and strategic feature engineering when building systematic investment models
""")
    if analysis_tab == "Regime-Aware Models":
        st.header("📊 Regime-Aware Models")
        st.image("dashboard_ref/top22featuredecade.png", caption="Top 22 Feature Decade")
        st.image("dashboard_ref/spearman.png", caption="Spearman Correlation")
        st.markdown("""
        ## 1.  Pipeline Overview

| Stage | What happens | Why it matters |
|-------|--------------|----------------|
| **Import & Knobs** | Set the global hyper-parameters (`DROP_THR`, `FLAG_THR`, `SHOW_N`, tree size, sampling cap, regimes). | Keeps the whole experiment reproducible and easy to tweak. |
| **Date Parsing** | `merged_df["date"] → pd.to_datetime` | Ensures calendar slicing works. |
| **NaN Scan** | Quick table of *absolute* and *%* missing by column. | Lets us decide what to drop / flag. |
| **Drop / Flag Logic** | *Drop* cols > 80 % NaN.<br>*Flag* cols 30–80 % by adding a `_nan` dummy. | Prevents super-sparse variables from injecting noise while still letting the model learn that “data missing” can be informative. |
| **Data Prep** | • Remove dropped columns.<br>• Add missing-flags.<br>• Sort by date. | Produces the final clean training frame. |
| **Model** | `ExtraTreesRegressor` in a `Pipeline` after a `SimpleImputer(median)`. | - Handles residual NaNs safely.<br>- ExtraTrees is ~2-3× faster than a deep RandomForest, yet exposes `feature_importances_`. |
| **Regime Loop** | For each decade in `REGIMES`:<br>  1. Slice rows.<br>  2. (Optionally) down-sample to `MAX_ROWS`.<br>  3. Fit the pipeline.<br>  4. Store feature importances & in-sample R². | Gives a comparable importance vector for every calendar regime. |
| **Visuals** | *Stacked bar* of relative importance for the **top-_k_** features (where _k ≤ SHOW_N_).<br>*Spearman heat-map* of feature-rank correlations across regimes. | Shows *how* and *whether* factor relevance shifts over time; the heat-map quantifies stability. |

---

## 2.  Why the code chooses these defaults

* **80 % cut-off**: a column with four-fifths NaNs can’t contribute reliable signal; better to drop it than impute noise.  
* **30–80 % flagged**: missingness itself can encode information (e.g., young IPOs lack long trend history).  
* **`MAX_ROWS = 10 000`**: keeps each fit < a few seconds on a laptop; raise once you prove it runs.  
* **Shallow ExtraTrees (`max_depth = 8`, `n_estimators = 120`)**: lightweight “smoke-test” that still captures nonlinearities.  
* **One CPU core (`n_jobs = 1`)**: avoids joblib RAM spikes; set to 4-8 if you have plenty of memory.

---

## 3.  Reading the outputs

### 3.1 Stacked-Bar Plot → Relative Importance by Decade
* **Each bar sums to 1** → you can compare colours horizontally.  
* **Wider slice over time** = factor is becoming more influential.  
* **Narrowing slice** = factor’s pricing power is fading.  

### 3.2 Spearman Heat-Map → Stability of Feature Ranking
* **Diagonal = 1** (same decade vs itself).  
* **Off-diagonal ≥ 0.85** → ordering is largely stable (evolution, not regime break).  
* **Off-diagonal ≤ 0.6** would flag a structural shift.

### 3.3 In-Sample R² Print-out
| Decade | R² (in-sample) |
|--------|----------------|
| 1990s  | ≈ 0.04 |
| 2000s–2020s | ≈ 0.09 |

*Even leading academic factor models seldom exceed 10 % cross-sectional R² at the one-month horizon, so these numbers are realistic.*

---

## 4.  Key Findings from the Example Run

1. **`retConglomerate`** dominated in the 1990s but steadily shrank thereafter → the classic “conglomerate discount” weakened post-dot-com.  
2. **Trend/momentum signals (`TrendFactor`, flag)** rose in the 2010s/2020s → investors rewarded price-trend information more in the low-rate era.  
3. **`betaVIX`** remained a stable mid-sized slice every decade → volatility risk carries a persistent premium.  
4. **High rank-correlations (0.85–0.92)** indicate no hard regime break; factor importance drifts smoothly.  
5. **R² doubles after the 1990s then plateaus** → either the factor set improved or markets became more predictable post-GFC, but further gains are limited without richer data.

---
""")