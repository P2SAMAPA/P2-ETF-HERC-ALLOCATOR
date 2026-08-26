# P2-ETF-HERC-ALLOCATOR

**Hierarchical Equal Risk Contribution (HERC) Allocation Engine for ETF Portfolios**

## Overview

`P2-ETF-HERC-ALLOCATOR` generates robust, risk-based portfolio weights for ETFs using **Hierarchical Equal Risk Contribution (HERC)**, introduced by Raffinot (2018) as an extension of Hierarchical Risk Parity (HRP). Like HRP, it requires no expected-return estimates and is robust to estimation error — but it structurally differs from HRP in two ways:

1. **Clustering is finite, not exhaustive.** The correlation dendrogram is cut into a data-driven number of clusters (chosen via the gap statistic) instead of being recursed all the way down to individual assets.
2. **Two-stage risk allocation.** Risk is equalized *between* clusters via top-down recursive bisection (same mechanic as HRP), while risk *within* each cluster is set with a single, non-recursive naive risk-parity (or return-tilted) pass.

This two-stage structure is what gives HERC its name — risk is contributed equally across the clusters identified in the correlation structure, rather than being bisected uniformly down to single names.

The engine uses a four-step process:

1. **Hierarchical Clustering** – Groups ETFs based on their correlation structure (identical distance metric and linkage to HRP, so results are comparable).
2. **Optimal Cluster Selection** – Cuts the dendrogram into *k* clusters, chosen automatically via the gap statistic (Tibshirani et al., 2001).
3. **Quasi-Diagonalization** – Reorders assets so cluster members are contiguous, exactly as in HRP.
4. **Two-Stage Risk Allocation** – Recursively splits risk between clusters (equal risk contribution), then allocates naive risk-parity weights within each cluster in a single pass.

Results are pushed daily to a dedicated Hugging Face dataset and visualized via a Streamlit dashboard.

## Universe Coverage

| Universe             | Tickers                                                                        |
| --------------------- | ------------------------------------------------------------------------------ |
| **FI / Commodities** | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV                                             |
| **Equity Sectors**   | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM (plus extended sector/factor set — see `config.py`) |
| **Combined**         | All tickers above                                                              |

### Diversified top-N picks

The daily top-3 selection avoids picking multiple overlapping/duplicate
ETFs together — e.g. SMH, SOXX, and XSD are all semiconductor-sector
funds, so if the top-weighted pick is SMH, SOXX and XSD are skipped in
favor of the next-best, genuinely different ticker. Groups are defined in
`config.SIMILARITY_GROUPS` and can be extended with other overlapping
exposures (e.g. multiple gold miners, multiple biotech funds) as needed.

Data is sourced from: [`P2SAMAPA/fi-etf-macro-signal-master-data`](https://huggingface.co/datasets/P2SAMAPA/fi-etf-macro-signal-master-data) — the **same master data source** used by the sibling HRP repo.

## Methodology

### Hierarchical Equal Risk Contribution (HERC)

1. **Covariance Estimation:** Compute the covariance matrix from the lookback window of log returns.
2. **Distance Matrix:** Convert correlation to distance: `d = sqrt((1 - ρ) / 2)`.
3. **Hierarchical Clustering:** Apply Ward's linkage to form a cluster tree.
4. **Optimal k:** Select the number of clusters via the gap statistic, capped at `MAX_CLUSTERS`.
5. **Quasi-Diagonalization:** Reorder assets based on cluster leaves.
6. **Inter-Cluster Split:** Recursively bisect the quasi-diagonal order, allocating weight inversely to relative cluster risk — stopping once a contiguous block belongs entirely to one identified cluster.
7. **Intra-Cluster Weights:** Within each final cluster, assign weights via a single naive risk-parity (or return-tilted) pass — no further recursion.

## File Structure

```
P2-ETF-HERC-ALLOCATOR/
├── config.py            # Paths, universes, HERC parameters
├── data_manager.py       # Data loading and preprocessing
├── herc_model.py          # Core HERC allocation logic
├── trainer.py             # Main orchestration script
├── push_results.py         # Upload results to Hugging Face
├── streamlit_app.py         # Interactive dashboard
├── requirements.txt          # Python dependencies
├── .github/workflows/          # Scheduled GitHub Action
└── .streamlit/                  # Streamlit theme
```

## Running Locally

```
git clone <your-repo-url>
cd P2-ETF-HERC-ALLOCATOR
pip install -r requirements.txt
export HF_TOKEN="your_token_here"
python trainer.py
streamlit run streamlit_app.py
```

## Dashboard Features

- **Pie & Bar Charts:** Visualize portfolio weights across ETFs.
- **Weight Table:** Detailed allocation percentages, with cluster membership shown per ticker.
- **Cluster Dendrogram:** Understand how assets are grouped by correlation, with the gap-statistic cluster cut highlighted.
- **Three Tabs:** Separate views for Combined, Equity, and FI/Commodities universes.

## Backtesting Lookback Windows

`backtest_lookback.py` runs a walk-forward comparison of HERC performance
across three lookback windows — **504, 630, and 756 trading days** — using
monthly or quarterly rebalancing with no lookahead. This is a one-off
research tool, not something meant to run daily.

**Locally:**

```
export HF_TOKEN="your_token_here"
python backtest_lookback.py --universe COMBINED --rebalance monthly --backtest-start 2008-01-01
```

(All three lookbacks — 504/630/756 — run automatically; pass `--lookbacks`
yourself only if you want to override that set. `--backtest-start` defaults
to 2008-01-01 since the underlying master data goes back that far — early
rebalances are automatically skipped until enough trailing history exists
for a given lookback, so the 2008–2009 crisis period naturally enters the
504/630/756-day windows once available rather than causing errors.)

Results (comparison CSV, equity-curve PNG, raw weights JSON) are saved to
`./backtest_results/` and, if `HF_TOKEN` is set, also pushed to the same
results dataset (`HF_OUTPUT_REPO`) under a timestamped `backtests/`
subfolder — alongside the daily `herc_weights_*.json` files, without
overwriting them. Pass `--no-push` to save locally only.

**From GitHub Actions (no local setup needed):**

Go to **Actions → Backtest Lookback Comparison (504 vs 630 vs 756, Manual)
→ Run workflow**, choose the universe, rebalance frequency, and start
date, then run it — all three lookback windows are always compared in a
single run. Make sure `HF_TOKEN` is set as a repo secret first (Settings →
Secrets → Actions). Results are pushed to the HF dataset the same way as a
local run, and are also attached to the workflow run as a downloadable
artifact under the run summary.

## Before you run this

The `push_results.py` / `streamlit_app.py` files point at:

```
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-herc-allocator-results"
```

Create that Hugging Face **dataset** repo (mirroring `p2-etf-hrp-allocator-results`) before your first run, and set the `HF_TOKEN` secret in your GitHub repo (Settings → Secrets → Actions) for the scheduled workflow to be able to push results.

## License

MIT License
