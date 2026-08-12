"""
src/pipeline.py
Market Sentiment Alpha Analysis — Core Pipeline Functions.

Extracted from Final_Notebook.ipynb for reusability and testability.
Operates on the pre-merged dataset (Output/merged_trader_sentiment.csv)
since raw trader_data.csv is proprietary and not redistributed.

Usage:
    from src.pipeline import load_merged_data, run_statistical_analysis, run_clustering
    df = load_merged_data()
    stats = run_statistical_analysis(df)
    clusters = run_clustering(df)
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import shap

warnings.simplefilter(action='ignore', category=FutureWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Data Loading (from merged output)
# ─────────────────────────────────────────────────────────────────────────────

def load_merged_data(path: str = "Output/merged_trader_sentiment.csv") -> pd.DataFrame:
    """
    Load the pre-merged trader × sentiment dataset.

    Args:
        path: Path to merged CSV (Phase 1 output).

    Returns:
        DataFrame with DatetimeIndex on 'date' column.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["Account", "date"]).reset_index(drop=True)
    print(f"✅ Loaded {len(df):,} rows | {df['Account'].nunique()} unique traders")
    print(f"   Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    return df


def load_sentiment_data(path: str = "data/sentiment.csv") -> pd.DataFrame:
    """Load Bitcoin Fear/Greed Index from raw CSV."""
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    df = df.sort_values('date').reset_index(drop=True)
    df = df.rename(columns={'value': 'sentiment_score', 'classification': 'sentiment_class'})
    return df[['date', 'sentiment_score', 'sentiment_class']]


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Statistical Analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_statistical_analysis(df: pd.DataFrame, n_bootstrap: int = 5000) -> Dict:
    """
    Run Phase 2 statistical analysis: Fear Premium test.

    Mann-Whitney U + Bootstrap Confidence Interval for
    mean PnL difference (Fear regimes vs Greed regimes).

    Args:
        df:          Merged dataset with 'sentiment_class' and 'daily_pnl'.
        n_bootstrap: Number of bootstrap resamples (default 5,000).

    Returns:
        {
          "fear_pnl": array,
          "greed_pnl": array,
          "mann_whitney_stat": float,
          "p_value": float,
          "rank_biserial_r": float,
          "bootstrap_ci_95": (lower, upper),
          "fear_premium": float,
        }
    """
    fear_pnl = df[df["sentiment_class"].isin(["Fear", "Extreme Fear"])]["daily_pnl"].dropna()
    greed_pnl = df[df["sentiment_class"].isin(["Greed", "Extreme Greed"])]["daily_pnl"].dropna()

    u_stat, p_value = stats.mannwhitneyu(fear_pnl, greed_pnl, alternative="greater")

    # Rank-biserial effect size
    n1, n2 = len(fear_pnl), len(greed_pnl)
    rank_biserial_r = 1 - (2 * u_stat) / (n1 * n2)

    # Bootstrap CI for mean difference (Fear - Greed)
    boot_diffs = []
    for _ in range(n_bootstrap):
        f_sample = np.random.choice(fear_pnl, size=len(fear_pnl), replace=True)
        g_sample = np.random.choice(greed_pnl, size=len(greed_pnl), replace=True)
        boot_diffs.append(np.mean(f_sample) - np.mean(g_sample))

    ci_lower, ci_upper = np.percentile(boot_diffs, [2.5, 97.5])
    fear_premium = float(np.mean(fear_pnl) - np.mean(greed_pnl))

    return {
        "fear_pnl": fear_pnl.values,
        "greed_pnl": greed_pnl.values,
        "mann_whitney_stat": float(u_stat),
        "p_value": float(p_value),
        "rank_biserial_r": float(rank_biserial_r),
        "bootstrap_ci_95": (float(ci_lower), float(ci_upper)),
        "fear_premium": fear_premium,
        "statistically_significant": p_value < 0.05,
        "ci_excludes_zero": ci_lower > 0,
    }


def analyze_long_short_bias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze directional positioning bias (long_pct) by sentiment regime.

    Returns a DataFrame with one row per sentiment class showing
    avg trades, avg long position %, avg trade size.
    """
    regime_order = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
    summary = df.groupby("sentiment_class").agg(
        avg_trades=("num_trades", "mean"),
        avg_long_pct=("long_pct", "mean"),
        avg_trade_size=("avg_trade_size_usd", "mean"),
        sample_size=("daily_pnl", "count"),
    ).reset_index()

    summary["sentiment_class"] = pd.Categorical(
        summary["sentiment_class"], categories=regime_order, ordered=True
    )
    return summary.sort_values("sentiment_class").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Clustering
# ─────────────────────────────────────────────────────────────────────────────

ARCHETYPE_NAMES = {0: "Snipers", 1: "Algorithms", 2: "Gamblers"}

def run_clustering(df: pd.DataFrame, k: int = 3) -> Tuple[pd.DataFrame, KMeans]:
    """
    K-Means clustering on trader lifetime metrics.

    Features: avg_win_rate, total_trades, avg_trade_size_usd

    Args:
        df: Merged dataset.
        k:  Number of clusters (default 3 — validated by Elbow + Silhouette in notebook).

    Returns:
        (lifetime_df_with_clusters, fitted_km_model)
    """
    lifetime = df.groupby("Account").agg(
        avg_win_rate=("win_rate", "mean"),
        total_trades=("num_trades", "sum"),
        avg_trade_size=("avg_trade_size_usd", "mean"),
        total_pnl=("daily_pnl", "sum"),
    ).dropna()

    scaler = StandardScaler()
    X = scaler.fit_transform(lifetime[["avg_win_rate", "total_trades", "avg_trade_size"]])

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lifetime["cluster"] = km.fit_predict(X)
    lifetime["archetype"] = lifetime["cluster"].map(ARCHETYPE_NAMES)

    return lifetime.reset_index(), km


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Walk-Forward Random Forest
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "num_trades_lag1", "win_rate_lag1",
    "avg_trade_size_usd_lag1", "long_pct_lag1",
    "total_volume_usd_lag1", "sentiment_score",
]


def run_walk_forward_rf(df: pd.DataFrame, n_splits: int = 5) -> Dict:
    """
    Walk-forward Random Forest predicting next-day profitability.

    Target: is_profitable_today (1 if daily_pnl > 0, else 0)
    Features: only _lag1 (yesterday's) features + current sentiment_score
              (no look-ahead bias — today's sentiment is known at day-open)

    Args:
        df:       Merged dataset with lag features.
        n_splits: Number of TimeSeriesSplit folds (default 5).

    Returns:
        {
          "fold_accuracies": [float, ...],
          "avg_accuracy": float,
          "all_y_true": [...],
          "all_y_pred": [...],
          "feature_importances": DataFrame,
          "shap_values": ndarray,
          "X_test_last": DataFrame,
        }
    """
    # Build feature matrix
    df_model = df.copy().dropna(subset=FEATURE_COLS)
    df_model["is_profitable"] = (df_model["daily_pnl"] > 0).astype(int)
    df_model = df_model.sort_values("date").reset_index(drop=True)

    X = df_model[FEATURE_COLS]
    y = df_model["is_profitable"]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_accuracies = []
    all_y_true, all_y_pred = [], []
    X_test_last, y_test_last = None, None

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if len(y_train) < 10 or len(y_test) < 2:
            continue

        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(acc)
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        X_test_last = X_test
        y_test_last = y_test

    # Train final model on all data for SHAP
    rf_final = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_final.fit(X, y)
    feature_importances = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": rf_final.feature_importances_,
    }).sort_values("importance", ascending=False)

    # SHAP values on last test fold
    shap_values = None
    if X_test_last is not None:
        try:
            explainer = shap.TreeExplainer(rf_final)
            shap_values = explainer.shap_values(X_test_last)
        except Exception:
            shap_values = None

    return {
        "fold_accuracies": fold_accuracies,
        "avg_accuracy": float(np.mean(fold_accuracies)) if fold_accuracies else 0.0,
        "all_y_true": all_y_true,
        "all_y_pred": all_y_pred,
        "feature_importances": feature_importances,
        "shap_values": shap_values,
        "X_test_last": X_test_last,
        "rf_model": rf_final,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Monte Carlo Strategy Validation
# ─────────────────────────────────────────────────────────────────────────────

def run_monte_carlo_strategy(
    df: pd.DataFrame,
    n_paths: int = 1000,
    n_days: int = 30,
    fear_multiplier: float = 2.0,
    greed_multiplier: float = 0.5,
    random_seed: int = 42,
) -> Dict:
    """
    Monte Carlo simulation validating the sentiment-conditioned sizing strategy.

    Strategy A (Fear Premium): When Fear/Extreme Fear → size × fear_multiplier
    Strategy B (Greed Hedge):  When Greed/Extreme Greed → size × greed_multiplier

    Args:
        df:               Merged dataset.
        n_paths:          Number of simulation paths.
        n_days:           Simulation horizon in trading days.
        fear_multiplier:  Position size multiplier during Fear. Default 2.0.
        greed_multiplier: Position size multiplier during Greed. Default 0.5.
        random_seed:      NumPy random seed for reproducibility.

    Returns:
        {
          "baseline_paths": (n_paths, n_days+1) ndarray,
          "strategy_paths": (n_paths, n_days+1) ndarray,
          "baseline_stats": {"p5", "p50", "p95", "final_median"},
          "strategy_stats": {"p5", "p50", "p95", "final_median"},
          "strategy_beats_baseline_pct": float,
        }
    """
    np.random.seed(random_seed)

    fear_pnl = df[df["sentiment_class"].isin(["Fear", "Extreme Fear"])]["daily_pnl"].dropna().values
    greed_pnl = df[df["sentiment_class"].isin(["Greed", "Extreme Greed"])]["daily_pnl"].dropna().values
    neutral_pnl = df[df["sentiment_class"] == "Neutral"]["daily_pnl"].dropna().values
    all_pnl = df["daily_pnl"].dropna().values

    # Use recent sentiment sequence as regime schedule
    sentiment_series = df.groupby("date")["sentiment_class"].first().reset_index()
    recent = list(sentiment_series["sentiment_class"].tail(n_days))
    while len(recent) < n_days:
        recent.append("Neutral")
    regime_schedule = recent[:n_days]

    def sample(sentiment_class):
        if sentiment_class in ["Fear", "Extreme Fear"] and len(fear_pnl) > 0:
            return np.random.choice(fear_pnl)
        elif sentiment_class in ["Greed", "Extreme Greed"] and len(greed_pnl) > 0:
            return np.random.choice(greed_pnl)
        elif len(neutral_pnl) > 0:
            return np.random.choice(neutral_pnl)
        return np.random.choice(all_pnl)

    baseline = np.zeros((n_paths, n_days + 1))
    strategy = np.zeros((n_paths, n_days + 1))

    for p in range(n_paths):
        for d, sc in enumerate(regime_schedule):
            pnl = sample(sc)
            baseline[p, d + 1] = baseline[p, d] + pnl
            if sc in ["Fear", "Extreme Fear"]:
                strategy[p, d + 1] = strategy[p, d] + pnl * fear_multiplier
            elif sc in ["Greed", "Extreme Greed"]:
                strategy[p, d + 1] = strategy[p, d] + pnl * greed_multiplier
            else:
                strategy[p, d + 1] = strategy[p, d] + pnl

    def path_stats(paths):
        p5, p50, p95 = np.percentile(paths, [5, 50, 95], axis=0)
        return {"p5": p5, "p50": p50, "p95": p95, "final_median": float(p50[-1])}

    strategy_beats = float(np.mean(strategy[:, -1] > baseline[:, -1]) * 100)

    return {
        "baseline_paths": baseline,
        "strategy_paths": strategy,
        "baseline_stats": path_stats(baseline),
        "strategy_stats": path_stats(strategy),
        "strategy_beats_baseline_pct": strategy_beats,
        "n_paths": n_paths,
        "n_days": n_days,
        "regime_schedule": regime_schedule,
    }
