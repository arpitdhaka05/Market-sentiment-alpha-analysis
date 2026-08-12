# Data Sources

## Files in This Directory

### `sentiment.csv` ✅ (Included — Public Data)
**Source:** Alternative.me Bitcoin Fear & Greed Index API  
**URL:** `https://api.alternative.me/fng/?limit=0&format=csv`  
**Schema:**
| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | int (unix epoch) | UTC timestamp |
| `value` | int (0–100) | Fear/Greed score |
| `classification` | str | Extreme Fear / Fear / Neutral / Greed / Extreme Greed |
| `date` | date | YYYY-MM-DD |

**Coverage:** 2018-02-01 to 2025-04-30 (2,644 rows)

---

### `trader_data.csv` ❌ (Not Included — Proprietary)
**Source:** Hyperliquid DEX trade-level order flow export  
**Schema:**
| Column | Type | Description |
|--------|------|-------------|
| `Account` | str | Wallet address (hex) |
| `Timestamp` | int (epoch ms) | Trade execution time |
| `Side` | str | BUY or SELL |
| `Closed PnL` | float | Realized PnL in USD |
| `Size USD` | float | Trade notional in USD |
| ... | ... | + 11 more columns |

**Note:** Raw trader data is not redistributed (211,224 rows of proprietary DEX order flow).  
The pre-merged analysis dataset is available in `Output/merged_trader_sentiment.csv`.

To replicate with your own data: export trade history from Hyperliquid and place as `data/trader_data.csv` with the schema above.

---

## Pre-Computed Output

### `Output/merged_trader_sentiment.csv`
The Phase 1 pipeline output — traders merged with sentiment, with lag features engineered.  
77 rows | 32 unique traders | Date range: 2023-03-28 to 2025-02-19
