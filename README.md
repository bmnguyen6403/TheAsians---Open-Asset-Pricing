# Research Question

## Problem Statement
The goal of this study is to develop advanced machine learning models aimed at predicting stock returns by combining survival analysis, signal decay analysis, and composite signal construction. Specifically, this study aims to construct robust composite signals using a number of financial predictors and assess their predictive stability in varying market states and investment horizons.

## Big Picture Question
Is the stock market dominated by a number of dominant drivers, i.e., value, momentum, and quality, or can the combination of numerous individually weak but statistically significant signals into composite indicators substantially increase return predictability, investment stability, and timing across various economic regimes?

## Specific Research Questions
- Which financial signals most significantly affect a stock’s survival probability (i.e., avoiding large crashes)?
- How does the predictive power of individual signals decay over 1-month, 3-month, and 6-month horizons?
- Can dynamically weighted composite signals outperform individual signals in terms of average returns and Sharpe ratio?
- How does the predictive strength of composite signals vary across different time periods and market regimes (e.g., pre-crisis, post-crisis)?
- Does survival modeling identify different risk factors compared to traditional return prediction?
- Are certain machine learning models better suited for extracting predictive power from composite signals in different regimes?

## Hypothesis
- **H1**: Combined signals created by aggregating several financial predictors will achieve higher average returns, superior Sharpe ratios, and lower portfolio volatility compared to single signals.
- **H2**: Allocating more importance to each indicator by their t-statistics, hazard ratios, or survival risk scores will increase the predictive precision overall.
- **H3**: The predictive power of the individual signals will decrease over time, but properly built composite signals will exhibit higher robustness on longer time horizons (up to 12 months).
- **H4**: Machine learning models that adapt dynamically to various market regimes (e.g., XGBoost) will provide more consistent predictive performance than static models.

# Necessary Data

## Final Dataset Results
The ultimate dataset will be sorted by firm and month from August 31, 2001, through December 29, 2023. The dataset will only consist of firms associated with signals that have a t-statistic above 3 and a high reputation for signal quality. Key variables will be:
- Signal name
- Quality rating of the signal
- T-statistic of the signal
- Returns on the portfolio associated with each signal
- Corresponding dates for the portfolio

The dataset must include:
- Monthly returns as the dependent (Y) variable
- A number of possible independent (X) variables (all explanatory variables must be lagged by one month)

Additionally, the final dataset will be enhanced to support four major areas of analysis:
- **Signal Engineering**: Creation of new engineered features by interacting, transforming, and combining existing signals to uncover non-linear and higher-order effects.
- **Regime Detection**: Splitting the dataset into distinct economic periods (such as 2001–2007, 2008–2015, and 2016–2023) to evaluate how the effectiveness of signals and composite models varies across different market regimes.
- **Survival Analysis**: Labeling firms with survival outcomes (such as a crash exceeding 50% or delisting) and measuring duration until the event, enabling the use of survival modeling techniques like the Cox Proportional Hazards model.
- **Signal Decay Analysis**: Calculating forward cumulative returns over 1, 3, and 6 months to assess how the predictive strength of signals and composites deteriorates over longer forecast horizons.

This comprehensive data structure will ensure robust testing of predictive power, model stability, signal resilience, and investment strategy performance over time.

# Current Data

We utilize the Open Source Asset Pricing (OpenAP) project dataset developed by Andrew Y. Chen and Tom Zimmermann. Using the OpenAP Python package, we download cross-sectional predictors organized into decile portfolios, with particular interest in continuous signals and long-short (LS) returns.

- The "Deciles (cts only)" approach has historically provided robust in-sample performance, with a mean monthly return of approximately 0.80%. 
- Recent underperformance indicates that more robust composite strategies must be developed and structural shifts over time in predictive ability must be researched.

Along with the portfolio-level data, OpenAP also offers datasets containing firm-specific signal values for financial predictors, monitored on a monthly basis. The availability of these firm-level signals allows:
- Sophisticated modeling techniques beyond simple portfolio classification
- Bespoke feature engineering
- Stock-by-stock-level survival modeling
- More accurate assessment of signal decay across various time horizons

In addition to OpenAP signals, we also use data from the Center for Research in Security Prices (CRSP) database, providing:
- Firm-month observations
- Stock returns
- Share prices
- Number of outstanding shares
- Delisting information

These data are necessary for calculating expected returns with precision, identifying survival outcomes (e.g., stock crashes or delistings), and constructing investment strategies at the individual stock level.

# Description of Output

The final output will be an interactive dashboard aimed at:
- Assessing predictive performance of composite signals
- Identifying the most impactful financial attributes
- Showing how signal strength decays across various horizons
- Offering survival probabilities for companies
- Monitoring model performance under various market regimes

The dashboard will combine:
- Signal engineering
- Regime detection
- Survival analysis
- Signal decay assessment

# Resources
- Textbook page: [OpenAP Anomaly Plot](https://ledatascifi.github.io/ledatascifi-2025/content/05/05e_OpenAP_anomaly_plot.html)
- Website: [Open Asset Pricing Project](https://www.openassetpricing.com/)
- Signals & Info: [Google Drive Link](https://drive.google.com/file/d/1Sev9s6cPFUGgxp1pFiej0lGzpsMqJCI2/view)
- Proposal instruction: [Project Proposal Template](https://ledatascifi.github.io/ledatascifi-2025/content/assignments/project_prop_template.html)
