# PEAD Earnings Prediction — Quantitative Event Study

> Predicting post-earnings market outperformance using SUE, technical signals, and macro regime filtering.

## Project Overview

This project builds a supervised classification model to predict whether a stock will **outperform the S&P 500** in the 5 trading days following an earnings announcement.

The core hypothesis is **Post-Earnings Announcement Drift (PEAD)**: stocks with strong positive earnings surprises continue to drift upward, and vice versa. We test whether this effect holds across different **macro regimes** (low-rate 2019–2021 vs high-rate 2022+) and whether ML can refine the simple SUE ranking into a tradeable signal.

### Target Variable
- **Primary**: Binary — did the stock's 5-day return exceed the S&P 500? (1 = outperform, 0 = underperform)
- **Secondary**: 3-class — Significant Beat / Neutral / Significant Miss (±5% excess return threshold)

---

## Repository Structure

```
earnings-prediction/
├── notebooks/
│   ├── 01_data_collection.ipynb       # Build the panel dataset
│   ├── 02_eda_car_baseline.ipynb      # EDA, PEAD validation, CAR plots
│   ├── 03_logistic_regression.ipynb   # MVP model + calibration
│   ├── 04_random_forest_regime.ipynb  # LightGBM + SHAP + regime analysis
│   └── 05_backtest.ipynb              # Strategy simulation + PnL
├── utils/
│   └── helpers.py                     # Shared functions (data, features, targets)
├── data/                              # Generated data (not committed to git)
│   ├── panel_dataset.csv
│   ├── logit_predictions.csv
│   └── lgbm_predictions.csv
├── outputs/                           # Charts and plots
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/earnings-prediction.git
cd earnings-prediction
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Mac/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Open in VS Code
```bash
code .
```
Select the `venv` kernel in Jupyter notebooks (top-right corner).

---

## Running the Project

Run notebooks **in order**. Each notebook saves outputs that the next one depends on.

| Notebook | What it does | Output |
|---|---|---|
| `01_data_collection` | Downloads price/earnings data, builds panel | `data/panel_dataset.csv` |
| `02_eda_car_baseline` | Validates PEAD signal, CAR plots | `outputs/02_*.png` |
| `03_logistic_regression` | MVP binary classifier | `data/logit_predictions.csv` |
| `04_random_forest_regime` | LightGBM + SHAP + regime split | `data/lgbm_predictions.csv` |
| `05_backtest` | Long/short strategy simulation | `outputs/05_*.png` |

> **Quick test**: Set `MAX_TICKERS = 50` in notebook 01 for a fast run (~5 min). Set to `None` for full S&P 500 (~45 min).

---

## Feature Categories

### Earnings Surprise (from yfinance / Capital IQ)
| Feature | Description |
|---|---|
| `sue` | Standardized Unexpected Earnings: `(Actual - Estimate) / StdDev(Estimates)` |
| `eps_surprise_pct` | Raw surprise as % of estimate |
| `hist_beat_rate` | Fraction of last 4 quarters where ticker beat estimates |
| `beat_streak` | Consecutive quarters of beating estimates |

### Technical / Price Signals (from yfinance)
| Feature | Description |
|---|---|
| `ret_5d`, `ret_14d`, `ret_20d` | Pre-earnings momentum |
| `rsi_14` | 14-day RSI |
| `rvol_pctile` | Realized vol percentile (IV proxy) |
| `vol_ratio_1d` | Day-of volume vs 30-day average (>200% = high volume) |
| `vol_term_ratio` | Short-term vs long-term vol (elevated = anticipation) |

### Macro Regime (^TNX, ^VIX via yfinance)
| Feature | Description |
|---|---|
| `rate_regime` | 1 if 10Y yield ≥ 2%, 0 otherwise |
| `vix_regime` | 1 if VIX ≥ 20, 0 otherwise |
| `tnx` | Raw 10Y yield (continuous) |
| `vix_5d_chg` | Change in VIX over 5 days (fear rising/falling) |

---

## Key Results (to be filled in after running)

| Metric | Logistic Regression | LightGBM |
|---|---|---|
| CV Accuracy | — | — |
| CV AUC | — | — |
| Top-decile precision | — | — |
| Backtest win rate | — | — |
| Approx Sharpe | — | — |

---

## Methodology Notes

### Why excess return vs S&P 500?
Using **alpha** (stock return minus market return) as the target correctly handles market-wide moves. If the market falls 3% and the stock falls 1%, that's a winning trade — the stock outperformed by 2%.

### Why TimeSeriesSplit, not K-Fold?
Standard K-Fold randomly shuffles data, which can put 2024 events in the training set while predicting 2020 events. This is **lookahead bias** — a critical error in financial ML. TimeSeriesSplit always trains on the past and tests on the future.

### Why is SUE better than raw EPS surprise?
SUE normalises the surprise by the **dispersion of analyst estimates**. A $0.05 beat where all analysts agreed (low dispersion) is far more meaningful than a $0.05 beat where estimates ranged by $1.00. *(Full SUE requires Capital IQ — yfinance approximation uses rolling forecast error std.)*

### Capital IQ Integration
If your group has Capital IQ access, replace the `get_earnings_history()` function in `utils/helpers.py` with true consensus data including:
- Analyst estimate standard deviation (for proper SUE)
- Revenue estimate vs actual (revenue surprise)
- Number of analysts covering the stock

---

## References

- Ball & Brown (1968) — Original PEAD paper
- Foster, Olsen & Shevlin (1984) — SUE methodology  
- Bernard & Thomas (1989) — PEAD persistence evidence
- Hou, Xue & Zhang (2020) — Replication of PEAD in modern data

---

## Team

*Add your names here*

---

*This project is for academic research purposes. Not financial advice.*
