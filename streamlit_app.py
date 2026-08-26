"""
Streamlit Dashboard for HERC Allocator.
Displays daily top-3 allocations and shrinking windows.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from huggingface_hub import HfApi, hf_hub_download
import json
import numpy as np
import scipy.cluster.hierarchy as sch

import config

st.set_page_config(
    page_title="P2Quant HERC Allocator",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: 600;
    color: #1f77b4;
    margin-bottom: 0.5rem;
}
.sub-header {
    font-size: 1.2rem;
    color: #555;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_latest_results():
    try:
        api = HfApi(token=config.HF_TOKEN)
        files = api.list_repo_files(repo_id=config.HF_OUTPUT_REPO, repo_type="dataset")
        json_files = sorted([f for f in files if f.endswith('.json')], reverse=True)
        if not json_files:
            return None

        local_path = hf_hub_download(
            repo_id=config.HF_OUTPUT_REPO,
            filename=json_files[0],
            repo_type="dataset",
            token=config.HF_TOKEN,
            cache_dir="./hf_cache"
        )
        with open(local_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None


def create_dendrogram(linkage: list, labels: list, cluster_labels: dict = None, optimal_k: int = None):
    if linkage is None or len(linkage) == 0:
        return None
    if labels is None or len(labels) == 0:
        return None

    expected_rows = len(labels) - 1
    if len(linkage) != expected_rows:
        if len(linkage) > expected_rows:
            linkage = linkage[:expected_rows]
        else:
            return None

    Z = np.array(linkage, dtype=np.float64)
    if Z.shape[1] != 4:
        return None

    fig, ax = plt.subplots(figsize=(12, 5))
    color_threshold = None
    if optimal_k and optimal_k > 1:
        # Approximate a color threshold that yields ~optimal_k clusters
        heights = np.sort(Z[:, 2])
        idx = max(len(heights) - optimal_k, 0)
        if idx < len(heights):
            color_threshold = heights[idx]

    sch.dendrogram(
        Z, labels=labels, orientation='top', leaf_rotation=45, leaf_font_size=10,
        ax=ax, color_threshold=color_threshold
    )
    title = "Hierarchical Clustering Dendrogram"
    if optimal_k:
        title += f"  (cut at k = {optimal_k} clusters)"
    ax.set_title(title)
    ax.set_xlabel("ETF Ticker")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    return fig


def display_allocation_tab(weights_dict: dict, cluster_info: dict, universe_key: str):
    if not weights_dict:
        st.info("No weights available.")
        return

    df = pd.DataFrame(weights_dict.items(), columns=['Ticker', 'Weight']).sort_values('Weight', ascending=False)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Portfolio Allocation")
        fig_pie = go.Figure(go.Pie(
            labels=df['Ticker'], values=df['Weight'], hole=0.4,
            textinfo='label+percent',
            marker=dict(colors=['#1f77b4' if i == 0 else '#a0aec0' for i in range(len(df))])
        ))
        fig_pie.update_layout(height=450)
        st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_{universe_key}")

    with col2:
        st.markdown("### Weight Distribution")
        fig_bar = go.Figure(go.Bar(
            x=df['Ticker'], y=df['Weight'],
            marker_color=['#1f77b4' if i == 0 else '#a0aec0' for i in range(len(df))],
            text=df['Weight'].apply(lambda x: f'{x:.2%}'), textposition='outside'
        ))
        fig_bar.update_layout(xaxis_title="ETF Ticker", yaxis_title="Weight", height=450)
        st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{universe_key}")

    st.markdown("### Detailed Weights")
    df_display = df.copy()
    df_display['Weight'] = df_display['Weight'].apply(lambda x: f'{x:.2%}')
    if cluster_info and cluster_info.get('cluster_labels'):
        df_display['Cluster'] = df_display['Ticker'].map(cluster_info['cluster_labels'])
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    st.markdown("### Hierarchical Clustering")
    if cluster_info:
        linkage = cluster_info.get('linkage')
        original_tickers = cluster_info.get('original_tickers')
        optimal_k = cluster_info.get('optimal_k')
        if optimal_k:
            st.caption(f"Gap-statistic selected **k = {optimal_k}** clusters for equal risk contribution.")
        if linkage and original_tickers and len(original_tickers) > 2:
            fig_dendro = create_dendrogram(linkage, original_tickers, cluster_info.get('cluster_labels'), optimal_k)
            if fig_dendro:
                st.pyplot(fig_dendro, use_container_width=True)
            else:
                st.warning("Dendrogram could not be created.")
        else:
            st.info("Insufficient cluster information.")
    else:
        st.info("No cluster information available.")


# --- Sidebar ---
st.sidebar.markdown("## ⚙️ Configuration")
st.sidebar.markdown(f"**Data Source:** `{config.HF_DATA_REPO}`")
st.sidebar.markdown(f"**Results Repo:** `{config.HF_OUTPUT_REPO}`")
st.sidebar.divider()
st.sidebar.markdown("### 🧩 HERC Parameters")
st.sidebar.markdown(f"- Daily Lookback: **{config.LOOKBACK_WINDOW} days**")
st.sidebar.markdown(f"- Top N Daily: **{config.TOP_N_DAILY}**")
st.sidebar.markdown(f"- Linkage Method: **{config.LINKAGE_METHOD}**")
st.sidebar.markdown(f"- Return Metric: **{config.RETURN_METRIC}**")
n_clusters_display = config.N_CLUSTERS if config.N_CLUSTERS is not None else f"auto (gap statistic, max {config.MAX_CLUSTERS})"
st.sidebar.markdown(f"- Clusters: **{n_clusters_display}**")
st.sidebar.divider()

data = load_latest_results()
if data:
    st.sidebar.markdown(f"**Run Date:** {data.get('run_date', 'Unknown')}")
else:
    st.sidebar.markdown("*No data available*")

st.sidebar.divider()
st.sidebar.markdown("### 📖 About")
st.sidebar.markdown("""
**HERC Allocator** generates robust portfolio weights using Hierarchical
Equal Risk Contribution.

- The dendrogram is cut into a data-driven number of clusters.
- Risk is equalized **between clusters** via recursive bisection.
- Weights **within** each cluster are set with a single, non-recursive
  naive risk-parity (or return-tilted) pass.
- Daily trading shows top-3 focused allocation.
- Shrinking windows reveal allocation stability across different
  historical periods.
""")

# --- Main Content ---
st.markdown('<div class="main-header">🧩 P2Quant HERC Allocator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Hierarchical Equal Risk Contribution – Daily Rebalanced & Historical Windows</div>', unsafe_allow_html=True)

if data is None:
    st.warning("No data available. Please run the daily pipeline first.")
    st.stop()

is_new_format = 'daily_trading' in data
if not is_new_format:
    st.warning("Legacy data format detected. Please run the latest trainer.")
    st.stop()

daily_data = data['daily_trading']
shrinking_data = data.get('shrinking_windows', {})

tab_daily, tab_shrink = st.tabs(["📋 Daily Trading (Top 3)", "📆 Shrinking Windows"])

with tab_daily:
    st.markdown("### Daily Allocation – Restricted to Top 3 ETFs per Universe")
    daily_subtabs = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])
    universe_keys = ["COMBINED", "EQUITY_SECTORS", "FI_COMMODITIES"]

    for subtab, ukey in zip(daily_subtabs, universe_keys):
        with subtab:
            weights = daily_data['top5_weights'].get(ukey, {})
            cluster_info = daily_data.get('cluster_info', {}).get(ukey, {})
            display_allocation_tab(weights, cluster_info, f"daily_{ukey}")

with tab_shrink:
    if not shrinking_data:
        st.warning("No shrinking windows data available yet.")
        st.stop()

    st.markdown("### Allocation Evolution Across Historical Windows")
    shrink_subtabs = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])

    for subtab, ukey in zip(shrink_subtabs, universe_keys):
        with subtab:
            windows = []
            for label, winfo in sorted(shrinking_data.items(), key=lambda x: x[1]['start_year'], reverse=True):
                w = winfo['weights'].get(ukey, {})
                if w:
                    top_items = sorted(w.items(), key=lambda x: x[1], reverse=True)[:5]
                    windows.append({
                        'Window': label,
                        'Top Allocation': ', '.join([f"{t} ({v:.1%})" for t, v in top_items])
                    })

            if windows:
                df_windows = pd.DataFrame(windows)
                st.dataframe(df_windows, use_container_width=True, hide_index=True)

                selected_window = st.selectbox("Select window to view full allocation:",
                                                [w['Window'] for w in windows], key=f"select_{ukey}")
                if selected_window:
                    full_weights = shrinking_data[selected_window]['weights'].get(ukey, {})
                    st.markdown(f"#### Full Allocation for {selected_window}")
                    df_full = pd.DataFrame(full_weights.items(), columns=['Ticker', 'Weight'])
                    df_full = df_full.sort_values('Weight', ascending=False)
                    df_full['Weight'] = df_full['Weight'].apply(lambda x: f'{x:.2%}')
                    st.dataframe(df_full, use_container_width=True, hide_index=True)
            else:
                st.info(f"No shrinking window data for {ukey}.")
