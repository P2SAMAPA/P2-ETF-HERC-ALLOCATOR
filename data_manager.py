"""
Data loading and preprocessing for HERC Allocator engine.
Handles master_data.parquet with DatetimeIndex and wide-format price data.
"""

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download

import config


def load_master_data() -> pd.DataFrame:
    """
    Downloads master_data.parquet from Hugging Face and loads into DataFrame.
    Returns a DataFrame with columns: Date, ticker, log_return.
    """
    print(f"Downloading {config.HF_DATA_FILE} from {config.HF_DATA_REPO}...")
    file_path = hf_hub_download(
        repo_id=config.HF_DATA_REPO,
        filename=config.HF_DATA_FILE,
        repo_type="dataset",
        token=config.HF_TOKEN,
        cache_dir="./hf_cache"
    )
    df_wide = pd.read_parquet(file_path)
    print(f"Loaded {len(df_wide)} rows and {len(df_wide.columns)} columns.")

    # Reset DatetimeIndex to Date column
    if isinstance(df_wide.index, pd.DatetimeIndex):
        df_wide = df_wide.reset_index().rename(columns={'index': 'Date'})
    df_wide['Date'] = pd.to_datetime(df_wide['Date'])

    return df_wide


def compute_log_returns(df_wide: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """
    Compute log returns for given tickers from price columns.
    Returns a long-format DataFrame with Date, ticker, log_return.
    """
    available_tickers = [t for t in tickers if t in df_wide.columns]
    print(f"Found {len(available_tickers)} ticker columns out of {len(tickers)} expected.")

    # Melt to long format
    df_long = pd.melt(
        df_wide,
        id_vars=['Date'],
        value_vars=available_tickers,
        var_name='ticker',
        value_name='price'
    )

    # Compute log returns per ticker
    df_long = df_long.sort_values(['ticker', 'Date'])
    df_long['log_return'] = df_long.groupby('ticker')['price'].transform(
        lambda x: np.log(x / x.shift(1))
    )
    df_long = df_long.dropna(subset=['log_return'])

    return df_long[['Date', 'ticker', 'log_return']]


def prepare_returns_matrix(df_wide: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """
    Prepare a wide-format DataFrame of log returns with Date index.
    """
    df_long = compute_log_returns(df_wide, tickers)
    pivot_returns = df_long.pivot(index='Date', columns='ticker', values='log_return')
    return pivot_returns[tickers].dropna()
