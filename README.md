<div align="center">
  <h1>📈 PrimeTrade Data Science Intern Project</h1>
  <p><strong>Uncovering the Mathematical Edge Between Market Sentiment and Trader Behavior</strong></p>
  
  [![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
  [![Pandas](https://img.shields.io/badge/Pandas-Data_Wrangling-150458.svg)](https://pandas.pydata.org/)
  [![XGBoost](https://img.shields.io/badge/scikit--learn-Machine_Learning-F7931E.svg)](https://scikit-learn.org/)
  [![Jupyter](https://img.shields.io/badge/Jupyter-Interactive_Analysis-F37626.svg)](https://jupyter.org/)
</div>

<br/>

## 🎯 Project Overview
Most market analyses stop at simple correlation charts. This project adopts a strictly rigorous quantitative approach to evaluate how market sentiment (Fear/Greed) influences trader behavior on Hyperliquid:
1. **Data Engineering**: Resolving UTC epoch misalignments, extracting lagging momentum, computing normalized directional bias (`long_pct`), and eliminating data leakage.
2. **Statistical Rigor**: Utilizing Mann-Whitney U tests paired with Bootstrapped Confidence Intervals and Effect Sizes to prove the counter-intuitive "Fear Premium".
3. **Machine Learning**: Deploying Silhouette-optimized K-Means to discover organic Trader Archetypes, and a Walk-Forward Random Forest (TimeSeriesSplit) + SHAP values to predict next-day profitability.
4. **Validation**: Stress-testing actionable strategies through 1,000-path Monte Carlo simulations designed strictly without look-ahead bias.

---

## 📂 Repository Structure
```text
PrimeTrade/
├── data/                       # Raw datasets (Sentiment.csv, Trader_Data.csv)
├── images/                     # Saved charts from the Jupyter Notebook
├── output/                     # Generated intermediate files (merged_trader_sentiment.csv)
├── notebooks/                  
│   └── Final_Analysis.ipynb    # The core analytical engine (Phases 1 through 3)
├── WRITEUP.md                  # Detailed 1-page business memo with insights & strategy rules
└── README.md                   # Project overview and setup guide
```

---

## 🚀 Setup & Execution

### 1. Environment Setup
Create a virtual environment and install the required packages:
```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy scipy matplotlib seaborn scikit-learn shap jupyterlab
```

### 2. Running the Pipeline
Open the primary Jupyter Notebook to execute the entire pipeline step-by-step:
```bash
jupyter lab notebooks/Final_Analysis.ipynb
```
*Note: Ensure the `data/` directory contains both CSV files before running the notebook.*

---

## 🔮 What I'd Build With 2 More Weeks
While this notebook serves as a strong quantitative foundation, here is how I would productionize this analysis if given more time:

1. **Survival Analysis via Cox Proportional Hazards:** I would move away from static "Days Survived" and model the true hazard rate of trader liquidations dynamically at the tick level, testing if entering a trade during Greed exponentially increases the risk of blow-up.
2. **Hidden Markov Models (HMM) for Regime Detection:** Instead of relying on the external "Fear/Greed" index, I would build an HMM directly on the Hyperliquid order flow to identify latent market regimes organically.
3. **Automated Trading Bot (Streamlit + API):** I would build a live Streamlit dashboard hooked into the Hyperliquid API that flags when Cluster 2 ("The Gamblers") begins accumulating 100% Long positions during Greed—serving as an early warning counter-indicator to go Short.
4. **Advanced SHAP Dashboards:** Build an interactive Streamlit UI where portfolio managers can click on a specific trader Account ID and instantly see their personalized SHAP force-plot explaining their risk profile.
