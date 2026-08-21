<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=28&pause=1000&color=00D4FF&center=true&vCenter=true&width=700&lines=Hyperliquid+Sentiment+Intelligence;Quantitative+Behavioral+Finance;Fear+%26+Greed+Alpha+Engine" alt="Typing SVG" />

<br/>

# Fear × Greed × Alpha
### *A Rigorous Quantitative Dissection of Trader Behavior on Hyperliquid*

<br/>

![Python](https://img.shields.io/badge/Python_3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-blueviolet?style=for-the-badge)

<br/>

> **"Markets are driven by two primal forces: Fear and Greed. This project quantifies exactly what each costs you."**

</div>

---

##  TL;DR — What Was Found

| Signal | Finding |
|--------|---------|
| **Fear Premium** | Traders statistically earn more during Fear than Greed (p < 0.05, Bootstrap CI validated) |
| **Greed Trap** | 100% Long directional bias during Greed = full exposure to liquidation cascades |
| **3 Archetypes** | Snipers, Algorithms, and Gamblers — each with a radically different risk signature |
| **Predictive Edge** | Walk-Forward Random Forest achieves ~90% accuracy on next-day profitability |
| **Strategy Alpha** | Sentiment-conditioned sizing beats the baseline in 1,000-path Monte Carlo simulations |

---

##  Architecture

```
market-sentiment-alpha-analysis/
│
├──  data/                       # Data files
│   ├── sentiment.csv              # Bitcoin Fear/Greed Index (2018–2025, public)
│   ├── trader_data.csv            #  Not included (proprietary Hyperliquid data)
│   └── README.md                  # Data schema and provenance documentation
│
├──  Output/                     # Pipeline artifacts
│   └── merged_trader_sentiment.csv # Phase 1 pre-merged dataset (77 rows, ready to use)
│
├──  images/                     # Auto-generated chart exports
│   ├── pnl_distribution.png       # Phase 2: PnL by regime boxplot
│   ├── long_short_bias.png        # Phase 2: Directional bias analysis
│   ├── trader_archetypes.png      # Phase 2: K-Means cluster scatter
│   └── monte_carlo.png            # Phase 4: Strategy fan chart
│
├──  Notebook/
│   └── Final_Notebook.ipynb       # Core analytical engine (all 4 phases, pre-run)
│
├──  src/
│   └── pipeline.py                # Extracted pipeline functions (reusable, testable)
│
├── streamlit_app.py               #  Interactive dashboard (Fear/Greed Explorer + Strategy Sim)
├── save_images.py                 # Standalone chart regeneration script
├── requirements.txt               # All Python dependencies
├── WRITEUP.md                     # Business memo: insights + strategy rules
└── README.md                      # This file
```

---

##  Analytical Pipeline

```
Raw Data
   │
   ├─► Phase 1 ── Data Engineering & Alignment
   │             UTC epoch → daily dates │ long_pct normalization │ lag feature creation
   │
   ├─► Phase 2 ── Statistical & Behavioral Analysis
   │             Mann-Whitney U + Bootstrap CI │ Regime behavioral shifts │ K-Means archetypes
   │
   ├─► Phase 3 ── Predictive Modeling
   │             Walk-Forward Random Forest │ TimeSeriesSplit │ SHAP explainability
   │
   └─► Phase 4 ── Strategy Validation
                 1,000-path Monte Carlo │ Honest strategy (zero look-ahead bias)
```

---

##  Quick Start

```bash
# 1. Clone and enter the repo
git clone https://github.com/arpitdhaka05/market-sentiment-alpha-analysis
cd market-sentiment-alpha-analysis

# 2. Create isolated environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter lab Notebook/Final_Notebook.ipynb

# 5. (Optional) Launch the interactive dashboard
streamlit run streamlit_app.py

# 6. (Optional) Regenerate chart exports
python save_images.py
```

> **Note:** Ensure `data/sentiment.csv` is present. The merged analysis dataset is pre-computed in `Output/merged_trader_sentiment.csv`. The raw `trader_data.csv` (211K rows, proprietary Hyperliquid data) is not redistributed — replace with your own export from Hyperliquid.

---

##  Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Mann-Whitney U over t-test** | PnL data is non-normal; parametric tests would be invalid |
| **`long_pct` over `long_short_ratio`** | Bounded [0,1] metric; ratio produces values of 3B when shorts = 0 |
| **TimeSeriesSplit over random split** | Prevents future data leaking into training — standard in quant finance |
| **Silhouette score to pick K** | Evidence-based cluster count instead of arbitrary choice |
| **Bootstrap CI alongside p-value** | Confirms effect size, not just statistical significance |

---

##  What I'd Build Next

```
Month 2 Roadmap
│
├── Hidden Markov Models          → Detect latent regimes from order flow itself
├── Cox Proportional Hazards      → Model liquidation hazard rate at the tick level
├── Live Streamlit Dashboard      → Real-time Greed Trap alert for Cluster 2 (Gamblers)
└── Interactive SHAP Explorer     → Per-trader risk profile force plots for PMs
```

---

##  Related Projects — The Quant Ecosystem

This project is part of a connected quantitative finance research portfolio:

- **[Monte Carlo Finance Simulator](https://github.com/arpitdhaka05/monte-carlo-finance-simulator)** — The 1,000-path Monte Carlo in Phase 4 of this project validates the strategy rules. The full MC simulator (GBM, Heston, GARCH, Bootstrap, Jump Diffusion) is its own standalone tool.
- **[NIFTY-DDPM](https://github.com/arpitdhaka05/nifty-regime-ddpm)** — Regime-Conditioned Diffusion Model. The Fear/Greed regimes identified here directly inform the conditioning variable for the DDPM's regime-switching architecture.

---

<div align="center">

Built as an independent <strong>Quantitative Finance Research Project</strong> — analytical framework inspired by institutional quant research methodology.

</div>
