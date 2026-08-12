"""
streamlit_app.py
Market Sentiment Alpha — Interactive Dashboard
4 tabs: Overview, Fear/Greed Explorer, Trader Archetypes, Strategy Simulator

Run: streamlit run streamlit_app.py
"""

import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.simplefilter(action='ignore', category=FutureWarning)

from src.pipeline import (
    load_merged_data,
    load_sentiment_data,
    run_statistical_analysis,
    analyze_long_short_bias,
    run_clustering,
    run_walk_forward_rf,
    run_monte_carlo_strategy,
    ARCHETYPE_NAMES,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fear × Greed × Alpha",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Market Sentiment Alpha Analysis — Quantitative Finance Research Project by Arpit Dhaka",
    }
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — Premium Dark Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif !important; }

.stApp {
    background: linear-gradient(135deg, #060d18 0%, #0a1628 50%, #060d18 100%);
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1020 0%, #111827 100%);
    border-right: 1px solid #1e2840;
}
section[data-testid="stSidebar"] * { color: #c8d4e8 !important; }

.main-header {
    background: linear-gradient(135deg, #0d1f3c, #1a2848);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 28px;
}
.main-header h1 { color: #00D4FF; font-size: 2.2rem; font-weight: 700; margin: 0; }
.main-header p { color: #8892b0; margin: 8px 0 0 0; font-size: 1rem; }

.stat-card {
    background: linear-gradient(135deg, #111827, #1a2235);
    border: 1px solid #1e2840;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: border-color 0.2s, transform 0.2s;
}
.stat-card:hover { border-color: #00D4FF; transform: translateY(-2px); }
.stat-card .value { color: #00D4FF; font-size: 1.8rem; font-weight: 700; }
.stat-card .label { color: #8892b0; font-size: 0.85rem; margin-top: 4px; }

[data-testid="metric-container"] {
    background: linear-gradient(135deg, #111827, #1a2235) !important;
    border: 1px solid #1e2840 !important;
    border-radius: 10px !important;
    padding: 16px !important;
}
[data-testid="stMetricLabel"] { color: #8892b0 !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"] { color: #e0e8f8 !important; font-weight: 600 !important; }

.stTabs [data-baseweb="tab-list"] {
    background: #0d1220 !important;
    border-radius: 8px !important;
}
.stTabs [data-baseweb="tab"] { color: #8892b0 !important; font-weight: 500 !important; }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1a3a6e, #0d5c9e) !important;
    color: #00D4FF !important;
}
.stButton > button {
    background: linear-gradient(135deg, #1a3a6e, #0d5c9e) !important;
    border: 1px solid #00D4FF !important;
    color: #e0f4ff !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
hr { border-color: #1e2840 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Color Palettes
# ─────────────────────────────────────────────────────────────────────────────
REGIME_COLORS = {
    "Extreme Fear": "#b71c1c",
    "Fear": "#e53935",
    "Neutral": "#78909c",
    "Greed": "#43a047",
    "Extreme Greed": "#1b5e20",
}
ARCHETYPE_COLORS = {
    "Snipers": "#1565C0",
    "Algorithms": "#2E7D32",
    "Gamblers": "#C62828",
}
PLOTLY_DARK = dict(
    paper_bgcolor="#080c14",
    plot_bgcolor="#0e1520",
    font=dict(color="#c8d4e8"),
    xaxis=dict(gridcolor="#1e2840"),
    yaxis=dict(gridcolor="#1e2840"),
)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📊 Fear × Greed × Alpha</h1>
    <p>Quantitative dissection of 211,224 Hyperliquid trades against the Bitcoin Fear/Greed Index · 4-phase pipeline</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Data Loading (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    df = load_merged_data()
    sentiment_full = load_sentiment_data()
    return df, sentiment_full

with st.spinner("Loading datasets..."):
    df, sentiment_full = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    st.markdown("---")
    st.markdown("### 📊 Dataset Summary")
    st.metric("Total Rows", f"{len(df):,}")
    st.metric("Unique Traders", f"{df['Account'].nunique()}")
    st.metric("Date Range", f"{df['date'].min().strftime('%b %Y')} → {df['date'].max().strftime('%b %Y')}")

    st.markdown("---")
    st.markdown("### 🎨 Regime Colors")
    for regime, color in REGIME_COLORS.items():
        st.markdown(
            f'<span style="background:{color}33; border:1px solid {color}; '
            f'color:{color}; padding:3px 10px; border-radius:20px; '
            f'font-size:0.75rem; font-weight:600;">{regime}</span>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown(
        '<div style="color:#4a5568; font-size:0.75rem; text-align:center;">'
        'Market Sentiment Alpha v1.0<br>'
        'Quant Finance Research · 2025'
        '</div>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_overview, tab_regimes, tab_archetypes, tab_strategy = st.tabs([
    "📈 Overview",
    "😰 Fear/Greed Explorer",
    "👥 Trader Archetypes",
    "🎲 Strategy Simulator",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.markdown("## 🔬 Project Summary")

    # Key findings metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Fear Premium", "Statistically Significant", "p < 0.05")
    c2.metric("Greed Trap", "100% Long Bias", "During Extreme Greed")
    c3.metric("Trader Archetypes", "3 Clusters", "K-Means validated")
    c4.metric("RF Accuracy", "~90%", "Walk-Forward")
    c5.metric("MC Strategy", "1,000 paths", "30-day horizon")

    st.markdown("---")
    st.markdown("### 📅 Fear/Greed Index — Full History (2018–2025)")

    # Full sentiment history chart
    fig_sentiment = go.Figure()
    for regime, color in REGIME_COLORS.items():
        mask = sentiment_full["sentiment_class"] == regime
        if mask.sum() == 0:
            continue
        fig_sentiment.add_trace(go.Scatter(
            x=sentiment_full[mask]["date"],
            y=sentiment_full[mask]["sentiment_score"],
            mode="markers",
            name=regime,
            marker=dict(color=color, size=3, opacity=0.7),
            showlegend=True,
        ))

    fig_sentiment.add_trace(go.Scatter(
        x=sentiment_full["date"],
        y=sentiment_full["sentiment_score"].rolling(30, min_periods=1).mean(),
        mode="lines",
        name="30-day MA",
        line=dict(color="#00D4FF", width=1.5),
    ))
    fig_sentiment.add_hline(y=25, line=dict(color="#e53935", dash="dot"), annotation_text="Fear/Extreme Fear")
    fig_sentiment.add_hline(y=75, line=dict(color="#43a047", dash="dot"), annotation_text="Greed/Extreme Greed")
    fig_sentiment.update_layout(
        title="Bitcoin Fear & Greed Index — 2018 to 2025",
        height=400,
        **PLOTLY_DARK,
    )
    st.plotly_chart(fig_sentiment, use_container_width=True)

    st.markdown("### 🔗 Analytical Pipeline")
    st.markdown("""
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
    """)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — FEAR/GREED EXPLORER
# ═══════════════════════════════════════════════════════════════════════════
with tab_regimes:
    st.markdown("## 😰 Fear/Greed Regime Explorer")

    if st.button("🔬 Run Statistical Analysis", type="primary", key="run_stats"):
        with st.spinner("Running Mann-Whitney U + Bootstrap CI (5,000 resamples)..."):
            stats_result = run_statistical_analysis(df)
            st.session_state["stats_result"] = stats_result

    if "stats_result" not in st.session_state:
        st.info("Click **Run Statistical Analysis** to compute the Fear Premium test.")
    else:
        r = st.session_state["stats_result"]
        st.markdown("### 📐 Statistical Test Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mann-Whitney p-value", f"{r['p_value']:.4f}",
                  "✅ Significant" if r['statistically_significant'] else "❌ Not significant")
        m2.metric("Rank-Biserial r (effect size)", f"{r['rank_biserial_r']:.3f}")
        m3.metric("Fear Premium", f"${r['fear_premium']:,.2f}")
        ci_l, ci_u = r['bootstrap_ci_95']
        m4.metric("95% Bootstrap CI", f"[${ci_l:,.0f}, ${ci_u:,.0f}]",
                  "✅ Excludes zero" if r['ci_excludes_zero'] else "⚠️ Includes zero")

    st.markdown("---")
    st.markdown("### 📊 PnL Distribution by Regime")

    regime_order = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
    present_regimes = [r for r in regime_order if r in df["sentiment_class"].unique()]

    fig_box = go.Figure()
    for regime in present_regimes:
        sub = df[df["sentiment_class"] == regime]["daily_pnl"].dropna()
        # Clip to IQR × 3 for readability
        q1, q3 = sub.quantile(0.25), sub.quantile(0.75)
        iqr = q3 - q1
        clipped = sub[(sub >= q1 - 3*iqr) & (sub <= q3 + 3*iqr)]
        fig_box.add_trace(go.Box(
            y=clipped,
            name=regime,
            marker_color=REGIME_COLORS.get(regime, "#78909c"),
            boxmean=True,
        ))

    fig_box.add_hline(y=0, line=dict(color="#8892b0", dash="dash"), annotation_text="Break-even")
    fig_box.update_layout(title="Daily PnL Distribution (outliers clipped to IQR×3)", height=450, **PLOTLY_DARK)
    st.plotly_chart(fig_box, use_container_width=True)

    # Behavioral summary bar chart
    st.markdown("### 📊 Directional Bias — Long Position % by Regime")
    bias_df = analyze_long_short_bias(df)
    fig_bias = px.bar(
        bias_df,
        x="sentiment_class",
        y="avg_long_pct",
        color="sentiment_class",
        color_discrete_map=REGIME_COLORS,
        labels={"avg_long_pct": "Avg Long Position %", "sentiment_class": "Regime"},
        title="The Greed Trap — Average Long Position % by Sentiment Regime",
    )
    fig_bias.add_hline(y=0.5, line=dict(color="#8892b0", dash="dot"), annotation_text="50% neutral")
    fig_bias.update_layout(height=380, showlegend=False, **PLOTLY_DARK)
    st.plotly_chart(fig_bias, use_container_width=True)

    with st.expander("💡 Interpretation"):
        st.markdown("""
        - **The Fear Premium:** Traders earn more during Fear (statistically confirmed, p < 0.05).
        - **The Greed Trap:** During Greed/Extreme Greed, the aggregate book approaches 100% Long — a completely unhedged position against a reversal.
        - **Bootstrap CI:** The 95% interval for (Fear PnL − Greed PnL) excludes zero — ruling out that this is driven by lucky outlier days.
        """)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — TRADER ARCHETYPES
# ═══════════════════════════════════════════════════════════════════════════
with tab_archetypes:
    st.markdown("## 👥 Trader Archetype Explorer")

    if st.button("🔍 Run Clustering Analysis", type="primary", key="run_cluster"):
        with st.spinner("Running K-Means clustering (K=3 — validated by Elbow + Silhouette)..."):
            lifetime_df, km_model = run_clustering(df, k=3)
            st.session_state["lifetime_df"] = lifetime_df

    if "lifetime_df" not in st.session_state:
        st.info("Click **Run Clustering Analysis** to identify trader archetypes.")
    else:
        lifetime_df = st.session_state["lifetime_df"]

        # Summary metrics per archetype
        st.markdown("### 🏷️ Archetype Profiles")
        cols = st.columns(3)
        for i, (archetype, group) in enumerate(lifetime_df.groupby("archetype")):
            color = ARCHETYPE_COLORS.get(archetype, "#78909c")
            with cols[i]:
                st.markdown(f"""
                <div style="background:#111827; border:1px solid {color}; border-radius:12px; padding:16px; text-align:center;">
                    <div style="color:{color}; font-size:1.1rem; font-weight:700;">{archetype}</div>
                    <div style="color:#8892b0; font-size:0.75rem; margin-top:4px;">n = {len(group)} traders</div>
                    <div style="margin-top:12px; color:#c8d4e8; font-size:0.85rem;">
                        Avg Win Rate: <b>{group['avg_win_rate'].mean()*100:.1f}%</b><br>
                        Avg Trades: <b>{group['total_trades'].mean():.0f}</b><br>
                        Avg Trade Size: <b>${group['avg_trade_size'].mean():,.0f}</b><br>
                        Total PnL: <b style="color:{'#4caf50' if group['total_pnl'].sum() > 0 else '#ef5350'};">${group['total_pnl'].sum():,.0f}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        # Scatter plot
        st.markdown("### 📈 Risk vs Reward by Archetype")
        fig_scatter = px.scatter(
            lifetime_df,
            x="avg_win_rate",
            y="total_pnl",
            color="archetype",
            size="avg_trade_size",
            size_max=25,
            color_discrete_map=ARCHETYPE_COLORS,
            labels={"avg_win_rate": "Average Win Rate", "total_pnl": "Total PnL (USD)", "archetype": "Archetype"},
            title="Trader Risk-Reward Profile by Cluster",
            hover_data=["total_trades"],
        )
        fig_scatter.add_hline(y=0, line=dict(color="#8892b0", dash="dash"))
        fig_scatter.add_vline(x=0.5, line=dict(color="#8892b0", dash="dot"))
        fig_scatter.update_traces(marker=dict(opacity=0.8, line=dict(width=1, color="#0a1020")))
        fig_scatter.update_layout(height=480, **PLOTLY_DARK)
        st.plotly_chart(fig_scatter, use_container_width=True)

        with st.expander("💡 Interpretation"):
            st.markdown("""
            | Archetype | Behavioral Profile | Risk Signature |
            |-----------|-------------------|----|
            | 🎯 **Snipers** | High win rate, low frequency | Tight risk management, consistent PnL |
            | 🤖 **Algorithms** | Massive trade count | Machine-like consistency, good win rate |
            | 🎲 **Gamblers** | Low win rate, giant position sizes | Highest volatility — dangerous during Greed |
            
            **Key insight:** Gamblers carry the highest blow-up risk during Greed — 100% Long bias + oversized positions = maximum liquidation exposure.
            """)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — STRATEGY SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════
with tab_strategy:
    st.markdown("## 🎲 Monte Carlo Strategy Simulator")
    st.markdown("Stress-test the sentiment-conditioned sizing strategy against a baseline.")

    # Controls
    s1, s2, s3 = st.columns(3)
    with s1:
        n_paths = st.select_slider(
            "Simulation Paths",
            options=[100, 500, 1000, 5000],
            value=1000,
            key="mc_paths",
        )
    with s2:
        n_days = st.select_slider(
            "Horizon (Trading Days)",
            options=[10, 20, 30, 60, 90],
            value=30,
            key="mc_days",
        )
    with s3:
        fear_mult = st.slider(
            "Fear Multiplier (×)",
            min_value=1.0, max_value=4.0, value=2.0, step=0.5,
            key="fear_mult",
            help="Position size multiplier during Fear/Extreme Fear regimes",
        )
    greed_mult = st.slider(
        "Greed Multiplier (×)",
        min_value=0.1, max_value=1.0, value=0.5, step=0.1,
        key="greed_mult",
        help="Position size multiplier during Greed/Extreme Greed regimes",
    )

    if st.button(f"🚀 Run {n_paths:,}-Path Simulation", type="primary", key="run_mc"):
        with st.spinner(f"Simulating {n_paths:,} paths × {n_days} days..."):
            mc_result = run_monte_carlo_strategy(
                df, n_paths=n_paths, n_days=n_days,
                fear_multiplier=fear_mult, greed_multiplier=greed_mult,
            )
            st.session_state["mc_result"] = mc_result

    if "mc_result" not in st.session_state:
        st.info("Configure parameters above and click **Run Simulation**.")
    else:
        mc = st.session_state["mc_result"]
        bs = mc["baseline_stats"]
        ss = mc["strategy_stats"]

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Baseline Median Final PnL", f"${bs['final_median']:,.0f}")
        m2.metric("Strategy Median Final PnL", f"${ss['final_median']:,.0f}",
                  f"${ss['final_median'] - bs['final_median']:+,.0f} vs baseline")
        m3.metric("Strategy Beats Baseline", f"{mc['strategy_beats_baseline_pct']:.1f}% of paths")
        m4.metric("Paths Simulated", f"{mc['n_paths']:,}")

        # Fan chart
        days = list(range(mc["n_days"] + 1))
        fig_mc = go.Figure()

        # Baseline fan
        fig_mc.add_trace(go.Scatter(
            x=days + days[::-1],
            y=list(bs["p95"]) + list(bs["p5"][::-1]),
            fill="toself", fillcolor="rgba(229,57,53,0.1)",
            line=dict(color="rgba(0,0,0,0)"), name="Baseline 5th–95th", showlegend=True,
        ))
        fig_mc.add_trace(go.Scatter(
            x=days, y=bs["p50"],
            mode="lines", name="Baseline Median",
            line=dict(color="#e53935", width=2),
        ))

        # Strategy fan
        fig_mc.add_trace(go.Scatter(
            x=days + days[::-1],
            y=list(ss["p95"]) + list(ss["p5"][::-1]),
            fill="toself", fillcolor="rgba(67,160,71,0.1)",
            line=dict(color="rgba(0,0,0,0)"), name="Strategy 5th–95th", showlegend=True,
        ))
        fig_mc.add_trace(go.Scatter(
            x=days, y=ss["p50"],
            mode="lines", name="Strategy Median",
            line=dict(color="#43a047", width=2),
        ))

        # Regime overlay
        regime_sched = mc["regime_schedule"]
        for d, regime in enumerate(regime_sched):
            color = REGIME_COLORS.get(regime, None)
            if color and regime in ["Fear", "Extreme Fear", "Greed", "Extreme Greed"]:
                fig_mc.add_vrect(x0=d, x1=d+1, fillcolor=color, opacity=0.04, line_width=0)

        fig_mc.add_hline(y=0, line=dict(color="#8892b0", dash="dash"), annotation_text="Break-even")
        fig_mc.update_layout(
            title=f"Monte Carlo Strategy Validation — {mc['n_paths']:,} Paths × {mc['n_days']}d | Fear×{fear_mult} Greed×{greed_mult}",
            xaxis_title="Trading Days",
            yaxis_title="Cumulative PnL (USD)",
            height=480,
            legend=dict(bgcolor="#111827", bordercolor="#1e2840"),
            **PLOTLY_DARK,
        )
        st.plotly_chart(fig_mc, use_container_width=True)

        with st.expander("💡 How to read this chart"):
            st.markdown("""
            - **Red** = Baseline strategy (no sentiment conditioning)
            - **Green** = Sentiment strategy (size × fear_mult during Fear, × greed_mult during Greed)
            - **Shaded bands** = 5th–95th percentile across all simulation paths
            - **Background tints** = Regime overlay (red = Fear days, green = Greed days)
            
            **Strategy Logic:**
            - Fear Premium: Size × fear_multiplier during Fear/Extreme Fear (historically better outcomes)
            - Greed Hedge: Size × greed_multiplier during Greed/Extreme Greed (reduce unhedged long exposure)
            - Zero look-ahead bias: regime classification is known at day-start, no future information used
            """)
