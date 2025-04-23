# Research Question

## Problem Statement
We aim to develop robust machine learning models that predict stock returns by leveraging a diverse set of high-significance financial signals. Using five different machine learning approaches, our objective is to construct composite metrics from portfolios of these signals. This work seeks to improve the efficiency and predictive accuracy of stock return models, thereby providing enhanced investment performance over strategies based on individual signals.

## Big Picture Question
Is the stock market predominantly driven by a few dominant factors, such as value, momentum, or quality, or does a combination of many weak but significant signals offer genuine incremental predictive power?

Can combining multiple predictive financial signals into a single composite metric enhance portfolio return predictability and overall investment performance compared to using individual signals alone?

## Specific Research Questions
- **Which signals contribute most significantly to predictive performance?**
- **How does a composite signal, created by aggregating multiple financial predictors, compare with individual signals?**
- **Does weighting individual signals by their t-statistics or quality metrics further improve the performance of the composite signal?**
- **How sensitive are the composite signals to the choice of machine learning models and hyperparameters? Are certain models consistently better at extracting predictive information from the combination of signals?**

## Hypothesis
Our hypotheses focus on three key aspects: overall performance, the impact of weighting schemes, and the temporal stability of the composite strategy.

- **H1:** A composite signal, constructed by aggregating multiple financial predictors, will achieve higher average monthly returns and a superior Sharpe ratio compared to most individual signals.
- **H2:** Weighting the individual signals by t-statistics or quality metrics will further enhance the performance of the composite signal.
- **H3:** The composite strategy will demonstrate greater stability over time, exhibiting lower volatility than strategies based solely on single signals.

## Necessary Data

### Final Dataset Results
- The final dataset should be structured at the firm-month level from **August 31, 2001, to December 29, 2023**.
- It should include only those firms with a signal t-statistic greater than 3 and a high reputation for signal quality.
- Key variables include:
  - **Signal Name**
  - **Quality Rating**
  - **T-statistic of the Signal**
  - **Returns on the Portfolio for Each Signal**
  - **Portfolio Dates**
- The dataset must comprise monthly returns as the dependent (Y) variable alongside a wide range of plausible independent (X) variables.
- **Important:** All explanatory variables must be lagged one month; for example, the return in December 2023 must be matched only with information available prior to November 2020.

## Current Data
Our data is sourced from the **Open Source Asset Pricing (OpenAP)** project by Andrew Y. Chen and Tom Zimmermann, which provides cross-sectional stock return predictors and related data.

- The `signaldoc.csv` file is used to filter for high-quality, statistically significant signals.
- Using the OpenAP Python package, we download the performance of these signals sorted into decile portfolios, focusing on long-short (LS) returns, which measure the return spread between the top and bottom deciles.
- The rows labeled "Deciles" and "DecilesVW" represent these portfolios, while "cts only" refers to continuous signals.
- The "Deciles (cts only)" strategy shows the best in-sample performance with an average monthly return of 0.80%.
- **Note:** Although decile sorting historically generates strong returns, its effectiveness has declined over time, likely due to strategy saturation or changing market conditions.

## Description of Output
The final output will be presented in a dashboard that includes:
- The accuracy of the composite signals.
- The identification of the best feature with the highest impact on the model.

## Resources
- **Textbook Page:** [LEDatascifi - OpenAP Anomaly Plot](https://ledatascifi.github.io/ledatascifi-2025/content/05/05e_OpenAP_anomaly_plot.html)
- **Website:** [Open Asset Pricing](https://www.openassetpricing.com/)
- **Signals & Info:** [Google Drive File](https://drive.google.com/file/d/1Sev9s6cPFUGgxp1pFiej0lGzpsMqJCI2/view)
- **Proposal Instruction:** [Project Proposal Template](https://ledatascifi.github.io/ledatascifi-2025/content/assignments/project_prop_template.html)
