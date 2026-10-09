"""Case-clustered bootstrap and shared paired-test helpers.

Each sampled DDXPlus case contributes four matched evidence states (25%, 50%,
75%, 100%). These states are not independent, so confidence intervals resample
*cases* with replacement and keep every state of a sampled case together,
including repeats when a case is drawn more than once.

The module also holds the small paired-test helpers shared by the analysis
scripts: exact McNemar, Holm correction and the exact sign test.

Note on inference: the McNemar and Stuart-Maxwell p-values treat matched
*states* as independent pairs and are not adjusted for clustering within case.
The clustered bootstrap intervals are reported alongside them and do account
for repeated cases. The two can disagree near conventional thresholds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

DEFAULT_N_BOOT = 10_000
DEFAULT_CI_LEVEL = 0.95
CASE_ORDERS = ("sorted", "first_appearance")

Statistic = Callable[[pd.DataFrame], float]


# ---------------------------------------------------------------------------
# Case-clustered bootstrap
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapResult:
    """Point estimate with a percentile case-clustered bootstrap interval."""

    estimate: float
    ci_low: float
    ci_high: float
    n_boot: int
    n_valid: int  # replicates with a finite statistic (e.g. non-zero denominator)
    n_clusters: int
    seed: int
    case_order: str

    def as_dict(self) -> dict:
        return asdict(self)


class ClusteredBootstrap:
    """Reusable case-clustered bootstrap over one state-level table.

    The draws are an ``(n_boot, n_clusters)`` matrix of cluster indices from
    ``numpy.random.default_rng(seed).integers(0, n_clusters, ...)`` (PCG64).
    They are generated once and shared by every statistic evaluated with the
    same object, so related metrics are computed on identical resamples.
    Intervals are percentile intervals with NumPy's ``linear`` quantile method.

    Which case a drawn index refers to depends on ``case_order``. The same seed
    therefore gives different intervals under different orders:

    - ``"sorted"``: cases in lexicographic order of their IDs. Used for the
      paper's cross-model disagreement intervals (seed 20260831).
    - ``"first_appearance"``: cases in the order they first appear in ``df``.
      Used for the paper's Step 5 metric intervals (seed 20260830), where rows
      follow the evaluation-set sampling order. Pass rows in that order.

    ``seed`` and ``case_order`` have no defaults so every analysis states them.

    Parameters
    ----------
    df:
        State-level table. One row per matched evidence state. If several
        conditions are compared, put them in columns of the same row rather
        than in separate rows, so they are resampled together.
    seed:
        NumPy ``default_rng`` seed for the bootstrap draws.
    case_order:
        ``"sorted"`` or ``"first_appearance"`` (see above).
    cluster_col:
        Column that identifies the resampling unit (default ``case_id``).
    n_boot, ci_level:
        Number of replicates and two-sided interval level.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        seed: int,
        case_order: str,
        cluster_col: str = "case_id",
        n_boot: int = DEFAULT_N_BOOT,
        ci_level: float = DEFAULT_CI_LEVEL,
    ) -> None:
        if df.empty:
            raise ValueError("Cannot bootstrap an empty table.")
        if cluster_col not in df.columns:
            raise KeyError(f"Missing cluster column {cluster_col!r}.")
        if case_order not in CASE_ORDERS:
            raise ValueError(f"case_order must be one of {CASE_ORDERS}, got {case_order!r}.")

        self.df = df.reset_index(drop=True)
        self.cluster_col = cluster_col
        self.n_boot = int(n_boot)
        self.ci_level = float(ci_level)
        self.seed = int(seed)
        self.case_order = case_order

        ids = self.df[cluster_col]
        clusters = sorted(ids.unique()) if case_order == "sorted" else list(dict.fromkeys(ids))
        positions = {c: i for i, c in enumerate(clusters)}
        codes = ids.map(positions).to_numpy()
        self.clusters = clusters
        self._rows_by_cluster = [np.flatnonzero(codes == i) for i in range(len(clusters))]
        self.n_clusters = len(clusters)

        rng = np.random.default_rng(self.seed)
        self.draws = rng.integers(0, self.n_clusters, size=(self.n_boot, self.n_clusters))

    def resample_indices(self, b: int) -> np.ndarray:
        """Row positions of bootstrap replicate ``b`` (with cluster repeats)."""
        return np.concatenate([self._rows_by_cluster[c] for c in self.draws[b]])

    def ci(self, statistic: Statistic) -> BootstrapResult:
        """Percentile interval for ``statistic(df)``.

        Replicates where the statistic is undefined (NaN, e.g. a zero
        denominator) are dropped. ``n_valid`` reports how many remain.
        """
        estimate = float(statistic(self.df))
        values = np.empty(self.n_boot)
        for b in range(self.n_boot):
            values[b] = statistic(self.df.iloc[self.resample_indices(b)])
        valid = values[np.isfinite(values)]
        alpha = 1.0 - self.ci_level
        if valid.size:
            lo, hi = np.quantile(valid, [alpha / 2, 1 - alpha / 2], method="linear")
        else:
            lo = hi = float("nan")
        return BootstrapResult(
            estimate=estimate,
            ci_low=float(lo),
            ci_high=float(hi),
            n_boot=self.n_boot,
            n_valid=int(valid.size),
            n_clusters=self.n_clusters,
            seed=self.seed,
            case_order=self.case_order,
        )


def case_clustered_bootstrap(df: pd.DataFrame, statistic: Statistic, **kwargs) -> BootstrapResult:
    """One-off convenience wrapper around :class:`ClusteredBootstrap`."""
    return ClusteredBootstrap(df, **kwargs).ci(statistic)


# ---------------------------------------------------------------------------
# Statistic builders
# ---------------------------------------------------------------------------


def safe_ratio(numerator: float, denominator: float) -> float:
    """Ratio that returns NaN (never 0) for a zero denominator."""
    return float(numerator) / float(denominator) if denominator else float("nan")


def mean_of(col: str) -> Statistic:
    """Statistic: mean of a boolean/0-1 column."""
    return lambda d: safe_ratio(d[col].sum(), len(d))


def mean_difference(col_a: str, col_b: str) -> Statistic:
    """Statistic: mean(col_a) - mean(col_b) over the same matched rows."""
    return lambda d: safe_ratio(d[col_a].sum(), len(d)) - safe_ratio(d[col_b].sum(), len(d))


# ---------------------------------------------------------------------------
# Paired tests and multiplicity correction
# ---------------------------------------------------------------------------


def exact_mcnemar(a: Sequence[bool], b: Sequence[bool]) -> dict:
    """Exact (binomial) two-sided McNemar test for paired binary outcomes.

    ``a`` and ``b`` hold the outcome for the same matched states under two
    conditions. Only the discordant counts enter the test.
    """
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("Paired outcome arrays must have the same length.")
    n11 = int(np.sum(a & b))
    n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b))
    n00 = int(np.sum(~a & ~b))
    if n10 + n01 == 0:
        pvalue = 1.0
    else:
        pvalue = float(mcnemar([[n11, n10], [n01, n00]], exact=True).pvalue)
    return {
        "n_pairs": int(a.size),
        "a1_b1": n11,
        "a1_b0": n10,
        "a0_b1": n01,
        "a0_b0": n00,
        "discordant": n10 + n01,
        "p_exact_mcnemar": pvalue,
    }


def holm_correction(pvalues: Sequence[float], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Holm step-down adjustment. Returns ``(reject, p_adjusted)``.

    Apply it to one family of tests at a time. In the paper the three pairwise
    Stuart-Maxwell tests form one family and the nine action-specific McNemar
    tests form a separate family.
    """
    reject, p_adj, _, _ = multipletests(np.asarray(pvalues, dtype=float), alpha=alpha, method="holm")
    return reject, p_adj


def exact_sign_test(wins: int, losses: int) -> float:
    """Two-sided exact sign test of P(win) = 0.5 over non-tied pairs."""
    n = int(wins) + int(losses)
    if n == 0:
        return 1.0
    return float(binomtest(int(wins), n, 0.5, alternative="two-sided").pvalue)
