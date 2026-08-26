"""
backtest_lookback.py

Walk-forward backtest to compare HERC performance across different
lookback windows (e.g. 504 vs 756 trading days).

At each monthly rebalance date, weights are computed using ONLY data
available up to that date (no lookahead), then held until the next
rebalance. Portfolio daily returns are stitched together into an equity
curve for each lookback setting, and standard performance/turnover
metrics are compared.

Usage:
    export HF_TOKEN="your_token_here"
    python backtest_lookback.py --universe COMBINED --lookbacks 504 756 --rebalance monthly

Results are saved to ./backtest_results/ as CSV + PNG equity curve chart.
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
import data_manager
from herc_model import HERCAllocator


def get_rebalance_dates(dates: pd.DatetimeIndex, freq: str) -> list:
    """Return the last available trading date in each period (month/quarter)."""
    s = pd.Series(dates, index=dates)
    if freq == "monthly":
        grouped = s.groupby([dates.year, dates.month])
    elif freq == "quarterly":
        grouped = s.groupby([dates.year, dates.quarter])
    else:
        raise ValueError("freq must be 'monthly' or 'quarterly'")
    return sorted([grp.max() for _, grp in grouped])


def turnover(prev_weights: dict, new_weights: dict) -> float:
    tickers = set(prev_weights) | set(new_weights)
    return sum(abs(new_weights.get(t, 0.0) - prev_weights.get(t, 0.0)) for t in tickers) / 2.0


def max_drawdown(cum_returns: pd.Series) -> float:
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    return drawdown.min()


def run_backtest(returns_matrix: pd.DataFrame, lookback: int, rebalance_freq: str,
                  backtest_start: str, allocator_kwargs: dict) -> dict:
    """
    returns_matrix: wide DataFrame of log returns, full history, Date-indexed.
    lookback: trading days used to estimate HERC weights at each rebalance.
    """
    all_dates = returns_matrix.index
    bt_start_date = pd.Timestamp(backtest_start)

    # Only rebalance on dates where we have >= lookback prior observations
    # AND we are past the requested backtest start.
    eligible_dates = all_dates[(all_dates >= bt_start_date)]
    if len(eligible_dates) < 21:
        raise ValueError("Not enough post-start history to backtest.")

    rebalance_dates = get_rebalance_dates(pd.DatetimeIndex(eligible_dates), rebalance_freq)

    daily_portfolio_returns = []
    weights_history = {}
    optimal_k_history = []
    turnovers = []
    prev_weights = {}

    for i, reb_date in enumerate(rebalance_dates):
        hist = returns_matrix.loc[:reb_date]
        if len(hist) < lookback:
            continue  # not enough history yet at this rebalance
        window = hist.iloc[-lookback:]

        allocator = HERCAllocator(**allocator_kwargs)
        weights = allocator.allocate(window)
        weights_history[reb_date.strftime("%Y-%m-%d")] = weights
        if allocator.get_optimal_k():
            optimal_k_history.append(allocator.get_optimal_k())

        if prev_weights:
            turnovers.append(turnover(prev_weights, weights))
        prev_weights = weights

        # Determine holding period: from day after reb_date to next reb_date (inclusive)
        period_end = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else all_dates[-1]
        holding_mask = (returns_matrix.index > reb_date) & (returns_matrix.index <= period_end)
        holding_returns = returns_matrix.loc[holding_mask]

        if holding_returns.empty:
            continue

        w_vec = pd.Series(weights).reindex(holding_returns.columns).fillna(0.0)
        port_ret = holding_returns.dot(w_vec)  # log returns, additive along assets is an approximation
        daily_portfolio_returns.append(port_ret)

    if not daily_portfolio_returns:
        raise ValueError("Backtest produced no return periods — check backtest_start / lookback.")

    port_returns = pd.concat(daily_portfolio_returns).sort_index()
    port_returns = port_returns[~port_returns.index.duplicated(keep='first')]

    # Convert log returns to cumulative growth of $1
    cum_returns = np.exp(port_returns.cumsum())

    n_years = (cum_returns.index[-1] - cum_returns.index[0]).days / 365.25
    total_return = cum_returns.iloc[-1] - 1.0
    cagr = cum_returns.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else np.nan
    ann_vol = port_returns.std() * np.sqrt(252)
    sharpe = (port_returns.mean() * 252 - config.RISK_FREE_RATE) / ann_vol if ann_vol > 0 else np.nan
    mdd = max_drawdown(cum_returns)
    avg_turnover = float(np.mean(turnovers)) if turnovers else np.nan
    avg_k = float(np.mean(optimal_k_history)) if optimal_k_history else np.nan

    return {
        "lookback": lookback,
        "cum_returns": cum_returns,
        "port_returns": port_returns,
        "weights_history": weights_history,
        "metrics": {
            "total_return": total_return,
            "cagr": cagr,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": mdd,
            "avg_turnover_per_rebalance": avg_turnover,
            "avg_optimal_k": avg_k,
            "n_rebalances": len(weights_history),
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Backtest HERC across lookback windows")
    parser.add_argument("--universe", default="COMBINED", choices=list(config.UNIVERSES.keys()))
    parser.add_argument("--lookbacks", nargs="+", type=int, default=[504, 756])
    parser.add_argument("--rebalance", default="monthly", choices=["monthly", "quarterly"])
    parser.add_argument("--backtest-start", default="2012-01-01",
                         help="First rebalance date considered (needs lookback+ days of prior history)")
    parser.add_argument("--linkage-method", default=config.LINKAGE_METHOD)
    parser.add_argument("--return-metric", default=config.RETURN_METRIC)
    parser.add_argument("--n-clusters", type=int, default=None)
    parser.add_argument("--max-clusters", type=int, default=config.MAX_CLUSTERS)
    parser.add_argument("--out-dir", default="./backtest_results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading master data...")
    df_master = data_manager.load_master_data()
    tickers = config.UNIVERSES[args.universe]
    returns_matrix = data_manager.prepare_returns_matrix(df_master, tickers)
    print(f"Universe '{args.universe}': {returns_matrix.shape[1]} tickers, "
          f"{len(returns_matrix)} observations from {returns_matrix.index.min().date()} "
          f"to {returns_matrix.index.max().date()}")

    allocator_kwargs = dict(
        linkage_method=args.linkage_method,
        return_metric=args.return_metric,
        risk_free_rate=config.RISK_FREE_RATE,
        n_clusters=args.n_clusters,
        max_clusters=args.max_clusters,
        gap_reference_samples=config.GAP_STAT_REFERENCE_SAMPLES,
    )

    results = {}
    for lb in args.lookbacks:
        print(f"\n--- Backtesting lookback={lb} days ({args.rebalance} rebalance) ---")
        results[lb] = run_backtest(returns_matrix, lb, args.rebalance, args.backtest_start, allocator_kwargs)
        m = results[lb]["metrics"]
        print(f"  CAGR: {m['cagr']:.2%} | Vol: {m['ann_vol']:.2%} | Sharpe: {m['sharpe']:.2f} | "
              f"MaxDD: {m['max_drawdown']:.2%} | Avg Turnover: {m['avg_turnover_per_rebalance']:.2%} | "
              f"Avg k: {m['avg_optimal_k']:.1f} | Rebalances: {m['n_rebalances']}")

    # --- Comparison table ---
    rows = []
    for lb, res in results.items():
        row = {"lookback": lb, **res["metrics"]}
        rows.append(row)
    df_compare = pd.DataFrame(rows).set_index("lookback")
    csv_path = os.path.join(args.out_dir, f"comparison_{args.universe}_{args.rebalance}.csv")
    df_compare.to_csv(csv_path)
    print(f"\nSaved comparison table: {csv_path}")
    print(df_compare.to_string())

    # --- Equity curve chart ---
    fig, ax = plt.subplots(figsize=(11, 6))
    for lb, res in results.items():
        ax.plot(res["cum_returns"].index, res["cum_returns"].values, label=f"Lookback {lb}d")
    ax.set_title(f"HERC Equity Curves by Lookback Window — {args.universe} ({args.rebalance} rebalance)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png_path = os.path.join(args.out_dir, f"equity_curves_{args.universe}_{args.rebalance}.png")
    fig.savefig(png_path, dpi=150)
    print(f"Saved equity curve chart: {png_path}")

    # --- Raw results dump ---
    json_path = os.path.join(args.out_dir, f"raw_results_{args.universe}_{args.rebalance}.json")
    with open(json_path, "w") as f:
        json.dump(
            {str(lb): {"metrics": res["metrics"], "weights_history": res["weights_history"]}
             for lb, res in results.items()},
            f, indent=2, default=str
        )
    print(f"Saved raw results: {json_path}")


if __name__ == "__main__":
    main()
