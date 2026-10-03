"""
Configuration for P2-ETF-HERC-ALLOCATOR engine.
"""

import os
from datetime import datetime

# --- Hugging Face Repositories ---
HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-herc-allocator-results"

# --- Universe Definitions (mirroring master data exactly) ---
FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]

EQUITY_SECTORS_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "URA", "VUG", "VTV", "SPYG", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME", "SMH", "SOXX", "XLB", "IWD", "IWO",
    "IWF", "XSD", "XBI", "IWM", "CPER", "COPX",
]

ALL_TICKERS = list(set(FI_COMMODITIES_TICKERS + EQUITY_SECTORS_TICKERS))

UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES_TICKERS,
    "EQUITY_SECTORS": EQUITY_SECTORS_TICKERS,
    "COMBINED": ALL_TICKERS
}

# --- HERC Parameters ---
LOOKBACK_WINDOW = 756  # 4-year lookback for daily trading
SHRINKING_WINDOW_START_YEARS = list(range(2008, 2025))
LINKAGE_METHOD = "ward"
MIN_OBSERVATIONS = 100
TOP_N_DAILY = 3  # Restrict daily allocation to top 3 ETFs

# ETFs that are effectively duplicate/overlapping exposure (e.g. multiple
# semiconductor-sector funds). When building the top-N daily picks, only
# the highest-weighted member of each group is eligible to be selected —
# the rest are skipped in favor of the next-best, genuinely different
# ticker. Add more groups as you identify other overlapping exposures.
SIMILARITY_GROUPS = [
    ["SMH", "SOXX", "XSD"],  # semiconductor ETFs
]

# Return metric used for the inter-cluster equal-risk-contribution split.
# 'inverse_variance' (pure risk parity), 'sharpe', 'mean_return', 'return_over_var'
RETURN_METRIC = "mean_return"  # Use raw annualized returns
RISK_FREE_RATE = 0.0

# Number of hierarchical clusters to cut the dendrogram into before running
# equal-risk-contribution across clusters. Set to None to auto-select via the
# gap statistic (Tibshirani et al., bounded by MAX_CLUSTERS).
N_CLUSTERS = None
MAX_CLUSTERS = 10
GAP_STAT_REFERENCE_SAMPLES = 20  # number of random reference draws for gap statistic

# --- Date Handling ---
TODAY = datetime.now().strftime("%Y-%m-%d")

# --- Optional: Hugging Face Token ---
HF_TOKEN = os.environ.get("HF_TOKEN", None)
