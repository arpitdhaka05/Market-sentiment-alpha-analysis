# 🗺️ Roadmap

## ✅ Phase 1 — Research Pipeline (Completed)
- [x] 4-phase analytical pipeline (Data Engineering → Statistical → Predictive → Strategy)
- [x] Mann-Whitney U + Bootstrap CI for Fear Premium validation
- [x] K-Means archetype clustering (Snipers / Algorithms / Gamblers)
- [x] Walk-Forward Random Forest with TimeSeriesSplit (zero look-ahead bias)
- [x] 1,000-path Monte Carlo strategy validator
- [x] SHAP explainability for Random Forest
- [x] Interactive Streamlit dashboard

## 🔧 Phase 2 — Advanced Research (In Progress)
- [ ] **Hidden Markov Models (HMM)** — Detect latent behavioral regimes from order flow itself (not just the external Fear/Greed label)
  - *Directly connects to [NIFTY-DDPM](https://github.com/arpitdhaka05/nifty-regime-ddpm) — the HMM regime labels would become the conditioning variable*
- [ ] **Cox Proportional Hazards** — Model liquidation hazard rate at the tick level for Gambler archetype
- [ ] **SHAP Force Plots Per Trader** — Individual risk profile explanations for portfolio managers

## 🔮 Phase 3 — Live Integration
- [ ] **Live Streamlit Dashboard** — Real-time Greed Trap alert system (Hyperliquid API integration)
- [ ] **Multi-exchange generalization** — Test findings on Binance/Bybit order flow to validate beyond Hyperliquid
- [ ] **Monte Carlo → Full Simulator** — Connect regime labels to [Monte Carlo Finance Simulator](https://github.com/arpitdhaka05/monte-carlo-finance-simulator) for regime-conditioned path generation
