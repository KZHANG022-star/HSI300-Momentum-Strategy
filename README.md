# HSI300-Momentum-Strategy
Inspired by the strong performance of SPMO, this strategy is designed to test the effectiveness of the momentum factor within the China large-cap index, namely the HSI300. The Invesco S&amp;P 500 Momentum ETF tracks the S&amp;P 500 Momentum Index, which selects S&amp;P 500 stocks with high momentum scores and weights them by market cap and momentum.

## Overview
This repository contains a complete backtesting system for momentum strategies on the CSI 300 Index (HSI300), inspired by the S&P 500 Momentum ETF (SPMO) methodology. The system consists of two main components: data collection (`getdata_exact.py`) and strategy backtesting (`mo_hsi300_exact_drift.py`).

## System Architecture

### 1. Data Collection Module (`getdata_exact.py`)
**Purpose**: Automated historical data acquisition from BaoStock API

**Key Features:**
- **Dynamic Universe Construction**: Builds historical CSI 300 constituent lists (2013-present) by querying monthly snapshots
- **Survivorship-Bias-Free Dataset**: Downloads prices for ALL stocks that ever appeared in the index
- **Industry Classification**: Retrieves current industry data (approximation for historical periods)
- **Benchmark Data**: Fetches CSI 300 Index prices for performance comparison
- **Robust Data Handling**: Implements forward-filling for non-trading days and missing data

**Data Flow:**
1. `history_cons.csv` - Monthly constituent snapshots
2. `all_stocks_close.csv` - Adjusted closing prices for all historical components
3. `stock_industry.csv` - Industry classification mapping
4. `benchmark.csv` - CSI 300 Index daily prices

### 2. Strategy Backtesting Module (`mo_hsi300_exact_drift.py`)
**Purpose**: Implements and evaluates a momentum-based trading strategy with realistic constraints

**Core Strategy Logic:**
- **Momentum Factor**: Calculates 12-month momentum adjusted by 21-day volatility (risk-adjusted score)
- **Dynamic Universe**: Only trades stocks that were actually in CSI 300 at rebalance time
- **Industry Neutrality**: Limits exposure to any single industry (max 5 stocks per industry by default)
- **Realistic Portfolio Drift**: Simulates natural holding changes between rebalances (no daily rebalancing)
- **Transaction Costs**: Incorporates 15bps trading costs

**Key Innovations:**
1. **Avoids Look-ahead Bias**: Factors are lagged by one day (T-1 used for T decisions)
2. **Survivorship-Bias-Free**: Uses only stocks available at each historical moment
3. **Practical Implementation**: Quarterly rebalancing with natural holding drift
4. **Comprehensive Logging**: Detailed trade records with industry attribution

**Strategy Parameters:**
- Top N stocks: 20 (configurable)
- Max per industry: 5 (configurable)
- Rebalancing frequency: Quarterly (3 months)
- Momentum lookback: 12 months (252 days)
- Volatility window: 12 months (252 days)

## Performance Metrics
The backtesting engine calculates:
- Cumulative returns
- Annualized returns
- Annualized volatility
- Sharpe ratio
- Maximum drawdown
- Year-by-year performance comparison vs benchmark

## Usage Instructions

### Step 1: Data Collection
```bash
python getdata_exact.py
```
*Requires BaoStock API credentials*

### Step 2: Run Backtest
```bash
python mo_hsi300_exact_drift.py
```

### Customization
- Modify `START_DATE` and `END_DATE` in `getdata_exact.py` for different time periods
- Adjust `top_n`, `max_per_ind`, `cost_rate` in `mo_hsi300_exact_drift.py` for strategy variants
- Change rebalancing frequency by modifying the `resample('3ME')` parameter

## Output Files
1. `trade_logs_dynamic.csv` - Detailed transaction records
2. Console output with annual performance metrics
3. Interactive plots showing:
   - Strategy vs benchmark equity curves
   - Strategy drawdown chart

## Dependencies
- Python 3.7+
- pandas, numpy, matplotlib
- baostock (for data collection)
- tqdm (for progress bars)

## Limitations & Assumptions
1. **Industry Classification**: Uses current industry data as proxy for historical periods
2. **Liquidity**: Assumes all stocks are tradeable without slippage
3. **Corporate Actions**: Adjusted closing prices account for splits and dividends
4. **Trading Costs**: Fixed 15bps per transaction (bid-ask spread not modeled)
5. **Survivorship Bias**: Mitigated but not completely eliminated for very early periods

## Applications
- Academic research on momentum factor effectiveness in Chinese markets
- Quantitative strategy development and validation
- Portfolio construction and risk management
- Factor timing and allocation decisions

This framework provides a robust foundation for momentum strategy research in Chinese equities, combining academic rigor with practical implementation considerations.
