"""
Hierarchical Equal Risk Contribution (HERC) allocation model.

HERC (Raffinot, 2018) differs from HRP in two structural ways that this
implementation preserves:

1. The dendrogram is CUT into a finite number of clusters (chosen
   automatically via the gap statistic, or fixed via config.N_CLUSTERS)
   instead of being recursed all the way down to individual assets.
2. Once recursion reaches a contiguous block that belongs to a single
   cluster, weights inside that cluster are assigned with a single,
   non-recursive naive risk-parity (or return-tilted) pass — there is no
   further bisection inside a cluster.

Between clusters, risk is equalized via the same top-down recursive
bisection mechanic used in HRP (splitting the quasi-diagonalized order in
half at each step and allocating inversely to relative cluster risk),
which is what gives HERC its "equal risk contribution" property at the
cluster level.
"""

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from typing import Dict, List, Tuple


class HERCAllocator:
    """
    Hierarchical Equal Risk Contribution portfolio allocator.

    Supports inverse-variance, Sharpe, mean-return, and return-over-variance
    weighting metrics (same options as the HRP engine) for both the
    inter-cluster risk-parity split and the intra-cluster weighting pass.
    """

    def __init__(
        self,
        linkage_method: str = 'ward',
        return_metric: str = 'mean_return',
        risk_free_rate: float = 0.0,
        n_clusters: int = None,
        max_clusters: int = 10,
        gap_reference_samples: int = 20,
    ):
        self.linkage_method = linkage_method
        self.return_metric = return_metric
        self.risk_free_rate = risk_free_rate
        self.n_clusters = n_clusters
        self.max_clusters = max_clusters
        self.gap_reference_samples = gap_reference_samples

        self.linkage = None
        self.original_tickers = None
        self.returns_used = None
        self.cluster_labels = None   # ticker -> cluster id
        self.optimal_k = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def allocate(self, returns: pd.DataFrame) -> Dict[str, float]:
        if returns.shape[1] < 2:
            return {returns.columns[0]: 1.0}

        self.original_tickers = returns.columns.tolist()
        self.returns_used = returns.copy()
        n = len(self.original_tickers)

        cov = returns.cov()
        corr = returns.corr()
        dist_condensed = ssd.squareform(((1 - corr) / 2) ** 0.5)
        self.linkage = sch.linkage(dist_condensed, method=self.linkage_method)

        # --- Determine number of clusters to cut the tree into ---
        if self.n_clusters is not None:
            k = int(min(max(self.n_clusters, 1), max(n - 1, 1)))
        else:
            k = self._optimal_k_gap_statistic(dist_condensed, n)
        self.optimal_k = k

        # --- Quasi-diagonalize (same leaf ordering approach as HRP) ---
        ordered_indices = sch.leaves_list(self.linkage)
        ordered_tickers = [self.original_tickers[i] for i in ordered_indices]

        # --- Cut dendrogram into k clusters, map onto quasi-diagonal order ---
        cluster_ids_original_order = sch.fcluster(self.linkage, t=k, criterion='maxclust')
        ticker_to_cluster = dict(zip(self.original_tickers, cluster_ids_original_order))
        self.cluster_labels = ticker_to_cluster
        ordered_cluster_ids = [ticker_to_cluster[t] for t in ordered_tickers]

        cov_ordered = cov.loc[ordered_tickers, ordered_tickers]
        returns_ordered = returns[ordered_tickers]

        weights = self._recursive_bisection(cov_ordered, returns_ordered, ordered_cluster_ids)
        return dict(zip(ordered_tickers, weights))

    def get_linkage_and_labels(self) -> Tuple[np.ndarray, List[str]]:
        return self.linkage, self.original_tickers

    def get_cluster_labels(self) -> Dict[str, int]:
        return self.cluster_labels

    def get_optimal_k(self) -> int:
        return self.optimal_k

    # ------------------------------------------------------------------
    # Inter-cluster equal risk contribution (recursive bisection)
    # ------------------------------------------------------------------
    def _recursive_bisection(self, cov: pd.DataFrame, returns: pd.DataFrame, cluster_ids: List[int]) -> np.ndarray:
        n = cov.shape[0]
        if n == 1:
            return np.array([1.0])

        # Stop recursing once the contiguous block is a single cluster:
        # allocate within it via one non-recursive naive risk-parity pass.
        if len(set(cluster_ids)) == 1:
            return self._intra_cluster_weights(cov, returns)

        mid = n // 2
        left_indices = list(range(mid))
        right_indices = list(range(mid, n))

        cov_left = cov.iloc[left_indices, left_indices]
        cov_right = cov.iloc[right_indices, right_indices]
        ret_left = returns.iloc[:, left_indices]
        ret_right = returns.iloc[:, right_indices]
        cl_left = cluster_ids[:mid]
        cl_right = cluster_ids[mid:]

        w_left_naive = self._naive_risk_parity_weights(cov_left)
        w_right_naive = self._naive_risk_parity_weights(cov_right)

        score_left = self._cluster_risk_score(cov_left, ret_left, w_left_naive)
        score_right = self._cluster_risk_score(cov_right, ret_right, w_right_naive)

        alpha = score_right / (score_left + score_right)

        weights = np.zeros(n)
        weights[left_indices] = alpha * self._recursive_bisection(cov_left, ret_left, cl_left)
        weights[right_indices] = (1 - alpha) * self._recursive_bisection(cov_right, ret_right, cl_right)
        return weights

    # ------------------------------------------------------------------
    # Intra-cluster weighting (single pass, no further recursion)
    # ------------------------------------------------------------------
    def _intra_cluster_weights(self, cov: pd.DataFrame, returns: pd.DataFrame) -> np.ndarray:
        """Naive risk-parity (optionally return-tilted) weights across all
        members of a single identified cluster."""
        return self._metric_weights(cov, returns)

    def _naive_risk_parity_weights(self, cov: pd.DataFrame) -> np.ndarray:
        """Pure inverse-variance weights, used to size relative cluster risk
        during the inter-cluster split (kept metric-agnostic so the
        equal-risk-contribution property between clusters is well defined)."""
        diag = np.diag(cov)
        diag = np.where(diag < 1e-10, 1e-10, diag)
        w = 1.0 / diag
        return w / w.sum()

    def _metric_weights(self, cov: pd.DataFrame, returns: pd.DataFrame) -> np.ndarray:
        n = cov.shape[0]
        if self.return_metric == 'inverse_variance':
            return self._naive_risk_parity_weights(cov)

        metrics = np.zeros(n)
        for i in range(n):
            asset_ret = returns.iloc[:, i].values
            vol = np.sqrt(cov.iloc[i, i])
            if vol < 1e-10:
                vol = 1e-10
            mean_ret = np.mean(asset_ret) * 252  # annualized

            if self.return_metric == 'sharpe':
                metrics[i] = (mean_ret - self.risk_free_rate) / (vol * np.sqrt(252))
            elif self.return_metric == 'mean_return':
                metrics[i] = mean_ret
            elif self.return_metric == 'return_over_var':
                metrics[i] = mean_ret / (cov.iloc[i, i] * 252)
            else:
                raise ValueError(f"Unknown return metric: {self.return_metric}")

        min_val = np.min(metrics)
        if min_val < 0:
            metrics = metrics - min_val + 1e-6
        w = metrics
        return w / w.sum()

    def _cluster_risk_score(self, cov: pd.DataFrame, returns: pd.DataFrame, weights: np.ndarray) -> float:
        if self.return_metric == 'inverse_variance':
            return np.linalg.multi_dot((weights, cov.values, weights))

        metrics = []
        for i in range(cov.shape[0]):
            asset_ret = returns.iloc[:, i].values
            vol = np.sqrt(cov.iloc[i, i])
            if vol < 1e-10:
                vol = 1e-10
            mean_ret = np.mean(asset_ret) * 252
            if self.return_metric == 'sharpe':
                m = (mean_ret - self.risk_free_rate) / (vol * np.sqrt(252))
            elif self.return_metric == 'mean_return':
                m = mean_ret
            elif self.return_metric == 'return_over_var':
                m = mean_ret / (cov.iloc[i, i] * 252)
            else:
                m = 0.0
            metrics.append(m)

        metrics = np.array(metrics)
        min_val = np.min(metrics)
        if min_val < 0:
            metrics = metrics - min_val + 1e-6
        weighted_metric = np.dot(weights, metrics)
        return 1.0 / (weighted_metric + 1e-6)

    # ------------------------------------------------------------------
    # Optimal number of clusters via the Gap Statistic
    # (Tibshirani, Walther & Hastie, 2001), applied to the correlation
    # distance matrix / linkage used for the tree.
    # ------------------------------------------------------------------
    def _optimal_k_gap_statistic(self, dist_condensed: np.ndarray, n: int) -> int:
        if n <= 2:
            return 1

        max_k = int(min(self.max_clusters, n - 1))
        if max_k < 2:
            return 1

        dist_full = ssd.squareform(dist_condensed)

        def within_cluster_dispersion(k: int, D: np.ndarray) -> float:
            Z = sch.linkage(ssd.squareform(D, checks=False), method=self.linkage_method)
            labels = sch.fcluster(Z, t=k, criterion='maxclust')
            Wk = 0.0
            for c in set(labels):
                idx = np.where(labels == c)[0]
                if len(idx) <= 1:
                    continue
                sub = D[np.ix_(idx, idx)]
                Wk += sub.sum() / (2.0 * len(idx))
            return Wk

        off_diag = dist_full[np.triu_indices(n, k=1)]
        lo, hi = float(off_diag.min()), float(off_diag.max())
        if hi <= lo:
            hi = lo + 1e-6

        gaps = []
        rng = np.random.default_rng(42)
        for k in range(1, max_k + 1):
            Wk = within_cluster_dispersion(k, dist_full)
            ref_Wks = []
            for _ in range(self.gap_reference_samples):
                rand_vals = rng.uniform(lo, hi, size=off_diag.shape)
                D_rand = np.zeros((n, n))
                D_rand[np.triu_indices(n, k=1)] = rand_vals
                D_rand = D_rand + D_rand.T
                ref_Wks.append(within_cluster_dispersion(k, D_rand))
            ref_Wks = np.array(ref_Wks)
            ref_Wks = np.where(ref_Wks <= 0, 1e-10, ref_Wks)
            Wk_safe = Wk if Wk > 0 else 1e-10

            log_ref = np.log(ref_Wks)
            gap_k = np.mean(log_ref) - np.log(Wk_safe)
            sk = np.std(log_ref) * np.sqrt(1 + 1.0 / self.gap_reference_samples)
            gaps.append((k, gap_k, sk))

        for i in range(len(gaps) - 1):
            k, gap_k, _ = gaps[i]
            _, gap_k1, sk1 = gaps[i + 1]
            if gap_k >= gap_k1 - sk1:
                return max(k, 2)

        return max(gaps[-1][0], 2)
