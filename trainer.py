"""
Main training script for HERC Allocator engine.
Computes daily top-3 HERC weights and shrinking-window allocations.
"""

import json
import pandas as pd
import numpy as np

import config
import data_manager
from herc_model import HERCAllocator
import push_results


def get_top_n_weights(weights_dict: dict, n: int = 5) -> dict:
    """Keep only top N weights and renormalize to 1."""
    sorted_items = sorted(weights_dict.items(), key=lambda x: x[1], reverse=True)[:n]
    top_weights = dict(sorted_items)
    total = sum(top_weights.values())
    return {k: v / total for k, v in top_weights.items()}


def run_herc_allocation():
    print(f"=== P2-ETF-HERC-ALLOCATOR Run: {config.TODAY} ===")

    df_master = data_manager.load_master_data()

    allocator = HERCAllocator(
        linkage_method=config.LINKAGE_METHOD,
        return_metric=config.RETURN_METRIC,
        risk_free_rate=config.RISK_FREE_RATE,
        n_clusters=config.N_CLUSTERS,
        max_clusters=config.MAX_CLUSTERS,
        gap_reference_samples=config.GAP_STAT_REFERENCE_SAMPLES,
    )

    # ---------------------------
    # 1. Daily Trading (504d, top 3)
    # ---------------------------
    daily_full = {}
    daily_top5 = {}
    daily_cluster_info = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- Daily Trading: {universe_name} ---")
        returns_matrix = data_manager.prepare_returns_matrix(df_master, tickers)

        if len(returns_matrix) < config.MIN_OBSERVATIONS:
            continue

        recent_returns = returns_matrix.iloc[-config.LOOKBACK_WINDOW:]
        print(f"  Using {len(recent_returns)} observations")

        weights = allocator.allocate(recent_returns)
        daily_full[universe_name] = weights
        daily_top5[universe_name] = get_top_n_weights(weights, config.TOP_N_DAILY)

        linkage, original_tickers = allocator.get_linkage_and_labels()
        if linkage is not None and original_tickers is not None:
            daily_cluster_info[universe_name] = {
                "linkage": linkage.tolist(),
                "original_tickers": original_tickers,
                "cluster_labels": allocator.get_cluster_labels(),
                "optimal_k": allocator.get_optimal_k()
            }

        top_items = list(daily_top5[universe_name].items())
        print(f"  Optimal clusters (k): {allocator.get_optimal_k()}")
        print(f"  Top 3: {', '.join([f'{t}: {w:.2%}' for t, w in top_items])}")

    # ---------------------------
    # 2. Shrinking Windows
    # ---------------------------
    shrinking_results = {}

    for start_year in config.SHRINKING_WINDOW_START_YEARS:
        start_date = pd.Timestamp(f"{start_year}-01-01")
        window_label = f"{start_year}-{config.TODAY[:4]}"
        print(f"\n--- Shrinking Window: {window_label} ---")

        mask = df_master['Date'] >= start_date
        df_window = df_master[mask].copy()

        if len(df_window) < 252:
            print(f"  Skipping (less than 1 year of data)")
            continue

        window_weights = {}
        for universe_name, tickers in config.UNIVERSES.items():
            returns_matrix = data_manager.prepare_returns_matrix(df_window, tickers)
            if len(returns_matrix) < 252:
                continue
            weights = allocator.allocate(returns_matrix)
            window_weights[universe_name] = weights

        shrinking_results[window_label] = {
            "start_year": start_year,
            "start_date": start_date.isoformat(),
            "weights": window_weights
        }

    # ---------------------------
    # Build payload and push
    # ---------------------------
    output_payload = {
        "run_date": config.TODAY,
        "config": {
            "lookback_window": config.LOOKBACK_WINDOW,
            "linkage_method": config.LINKAGE_METHOD,
            "return_metric": config.RETURN_METRIC,
            "top_n_daily": config.TOP_N_DAILY,
            "n_clusters": config.N_CLUSTERS,
            "max_clusters": config.MAX_CLUSTERS
        },
        "daily_trading": {
            "full_weights": daily_full,
            "top5_weights": daily_top5,
            "cluster_info": daily_cluster_info
        },
        "shrinking_windows": shrinking_results
    }

    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")


if __name__ == "__main__":
    run_herc_allocation()
