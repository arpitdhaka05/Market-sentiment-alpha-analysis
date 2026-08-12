"""
save_images.py
Standalone chart export script.
Regenerates all 4 chart PNGs from the merged dataset.
Run: python save_images.py
"""

import os
import warnings
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set_theme(
    style="whitegrid",
    palette="mako",
    rc={"axes.spines.right": False, "axes.spines.top": False, "figure.figsize": (10, 6)}
)

os.makedirs("images", exist_ok=True)

# ── Load Data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("Output/merged_trader_sentiment.csv", parse_dates=["date"])
print(f"✅ Loaded {len(df)} rows from merged dataset")

# ── Chart 1: PnL Distribution by Sentiment Regime ─────────────────────────────
print("📊 Generating PnL distribution chart...")
fig, ax = plt.subplots(figsize=(12, 6))
regime_order = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
palette = {"Extreme Fear": "#d32f2f", "Fear": "#f44336",
           "Neutral": "#78909c", "Greed": "#388e3c", "Extreme Greed": "#1b5e20"}
sns.boxplot(
    data=df,
    x="sentiment_class",
    y="daily_pnl",
    order=[r for r in regime_order if r in df["sentiment_class"].unique()],
    palette=palette,
    showfliers=False,
    ax=ax,
)
ax.set_title("Daily PnL Distribution by Market Sentiment Regime", fontsize=14, fontweight="bold")
ax.set_xlabel("Sentiment Regime", fontsize=12)
ax.set_ylabel("Daily PnL (USD)", fontsize=12)
ax.axhline(0, color="black", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("images/pnl_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Saved images/pnl_distribution.png")

# ── Chart 2: Long/Short Bias by Regime ────────────────────────────────────────
print("📊 Generating long/short bias chart...")
regime_summary = df.groupby("sentiment_class").agg(
    avg_trades=("num_trades", "mean"),
    avg_long_pct=("long_pct", "mean"),
    avg_trade_size=("avg_trade_size_usd", "mean"),
).reset_index()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, col, label in zip(
    axes,
    ["avg_trades", "avg_long_pct", "avg_trade_size"],
    ["Avg Trade Count", "Avg Long Position %", "Avg Trade Size (USD)"]
):
    regime_order_present = [r for r in regime_order if r in regime_summary["sentiment_class"].values]
    data_ordered = regime_summary.set_index("sentiment_class").reindex(regime_order_present)[col]
    colors = [palette.get(r, "#78909c") for r in regime_order_present]
    ax.bar(regime_order_present, data_ordered, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xlabel("Sentiment Regime")
    ax.tick_params(axis='x', rotation=30)
    if col == "avg_long_pct":
        ax.axhline(0.5, color="black", linestyle="--", alpha=0.5, label="50% neutral")
        ax.legend()

plt.suptitle("Behavioral Shifts by Sentiment Regime", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("images/long_short_bias.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Saved images/long_short_bias.png")

# ── Chart 3: Trader Archetypes (K-Means) ─────────────────────────────────────
print("📊 Generating trader archetypes chart...")
# Compute lifetime metrics per trader
lifetime = df.groupby("Account").agg(
    avg_win_rate=("win_rate", "mean"),
    total_trades=("num_trades", "sum"),
    avg_trade_size=("avg_trade_size_usd", "mean"),
    total_pnl=("daily_pnl", "sum"),
).dropna()

scaler = StandardScaler()
X = scaler.fit_transform(lifetime[["avg_win_rate", "total_trades", "avg_trade_size"]])
km = KMeans(n_clusters=3, random_state=42, n_init=10)
lifetime["cluster"] = km.fit_predict(X)

archetype_names = {0: "Snipers", 1: "Algorithms", 2: "Gamblers"}
lifetime["archetype"] = lifetime["cluster"].map(archetype_names)

fig, ax = plt.subplots(figsize=(10, 6))
cluster_palette = {"Snipers": "#1565C0", "Algorithms": "#2E7D32", "Gamblers": "#C62828"}
for archetype, group in lifetime.groupby("archetype"):
    ax.scatter(
        group["avg_win_rate"] * 100,
        group["total_pnl"],
        label=f"Cluster — {archetype} (n={len(group)})",
        color=cluster_palette.get(archetype, "#78909c"),
        s=80, alpha=0.8, edgecolors="white", linewidths=0.5
    )
ax.set_title("Trader Archetypes: Risk vs Reward by Cluster", fontsize=14, fontweight="bold")
ax.set_xlabel("Average Win Rate (%)")
ax.set_ylabel("Total PnL (USD)")
ax.axhline(0, color="black", linestyle="--", alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig("images/trader_archetypes.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Saved images/trader_archetypes.png")

# ── Chart 4: Monte Carlo Strategy Simulation ──────────────────────────────────
print("📊 Generating Monte Carlo strategy chart...")

# Simulate 1,000 paths × 30 days for baseline vs strategy
np.random.seed(42)
N_PATHS = 1000
N_DAYS = 30

fear_days = df[df["sentiment_class"].isin(["Fear", "Extreme Fear"])]["daily_pnl"].dropna()
greed_days = df[df["sentiment_class"].isin(["Greed", "Extreme Greed"])]["daily_pnl"].dropna()
neutral_days = df[df["sentiment_class"] == "Neutral"]["daily_pnl"].dropna()
sentiment_series = df.groupby("date")["sentiment_class"].first().reset_index()
recent_sentiment = list(sentiment_series["sentiment_class"].tail(N_DAYS))
while len(recent_sentiment) < N_DAYS:
    recent_sentiment.append("Neutral")
recent_sentiment = recent_sentiment[:N_DAYS]

def sample_daily_pnl(sentiment_class):
    if sentiment_class in ["Fear", "Extreme Fear"] and len(fear_days) > 0:
        return np.random.choice(fear_days.values)
    elif sentiment_class in ["Greed", "Extreme Greed"] and len(greed_days) > 0:
        return np.random.choice(greed_days.values)
    elif len(neutral_days) > 0:
        return np.random.choice(neutral_days.values)
    else:
        return np.random.choice(df["daily_pnl"].dropna().values)

baseline_paths = np.zeros((N_PATHS, N_DAYS + 1))
strategy_paths = np.zeros((N_PATHS, N_DAYS + 1))

for path in range(N_PATHS):
    for day in range(N_DAYS):
        sc = recent_sentiment[day]
        pnl = sample_daily_pnl(sc)
        baseline_paths[path, day + 1] = baseline_paths[path, day] + pnl
        # Strategy: ×2 in Fear, ×0.5 in Greed
        if sc in ["Fear", "Extreme Fear"]:
            strategy_paths[path, day + 1] = strategy_paths[path, day] + pnl * 2.0
        elif sc in ["Greed", "Extreme Greed"]:
            strategy_paths[path, day + 1] = strategy_paths[path, day] + pnl * 0.5
        else:
            strategy_paths[path, day + 1] = strategy_paths[path, day] + pnl

fig, ax = plt.subplots(figsize=(12, 6))
days = range(N_DAYS + 1)

# Plot fan (5th-95th band)
b5, b50, b95 = np.percentile(baseline_paths, [5, 50, 95], axis=0)
s5, s50, s95 = np.percentile(strategy_paths, [5, 50, 95], axis=0)

ax.fill_between(days, b5, b95, alpha=0.15, color="#f44336", label="Baseline 5th–95th band")
ax.fill_between(days, s5, s95, alpha=0.15, color="#4caf50", label="Strategy 5th–95th band")
ax.plot(days, b50, color="#f44336", linewidth=2, label=f"Baseline Median (final: ${b50[-1]:,.0f})")
ax.plot(days, s50, color="#4caf50", linewidth=2, label=f"Strategy Median (final: ${s50[-1]:,.0f})")
ax.axhline(0, color="black", linestyle="--", alpha=0.5)
ax.set_title(f"Monte Carlo Strategy Validation — {N_PATHS:,} Paths × {N_DAYS} Days", fontsize=14, fontweight="bold")
ax.set_xlabel("Trading Days")
ax.set_ylabel("Cumulative PnL (USD)")
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig("images/monte_carlo.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Saved images/monte_carlo.png")

print("\n✅ All 4 charts regenerated successfully in images/")
