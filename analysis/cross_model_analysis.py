"""Cross-model comparison of Stage 1 action selection under identical evidence.

Every model sees exactly the same revealed evidence for every state. The
analysis describes how Stage 1 DECIDE / ASK / DEFER choices differ between
models. It does not establish that any model is clinically safer or better
calibrated, because there is no ground truth for which action is appropriate.

Expected input: one Stage 1 state table per model (CSV or JSONL)
----------------------------------------------------------------
Pass them as ``--states NAME=PATH`` (e.g. ``qwen=...``). Each table needs
``case_id``, ``evidence_level`` and ``stage1_action`` in {DECIDE, ASK, DEFER}.
If ``hidden_evidence_count`` is present, unfulfillable ASK is reported
separately in the action-rate table and plot. For the 3x3 tests below,
unfulfillable ASK is still counted as ASK, because it is the observed Stage 1
action. All tables must cover the same set of states.

Statistics
----------
- Agreement structure: all models agree / exactly two agree / all differ.
- Most frequent joint action patterns, among disagreement states only by
  default (see ``common_patterns``).
- Pairwise disagreement with case-clustered bootstrap CIs (shared draws).
  Bootstrap settings come from ``bootstrap.cross_model`` (seed 20260831,
  sorted case order), which reproduces the paper's intervals exactly.
- Pairwise Stuart-Maxwell test of marginal homogeneity on the 3x3 action
  table (closed form, asymptotic chi-square, df = 2), Holm-corrected over the
  model pairs as one family.
- Action-specific exact McNemar tests (each action vs. the other two, per pair),
  Holm-corrected over all pair x action tests as a separate family.

The p-values treat matched states as independent and are not adjusted for
clustering within case. The clustered bootstrap CIs are reported alongside them.

Usage
-----
    python analysis/cross_model_analysis.py --states qwen=q.csv gemini=g.csv gpt=o.csv \
        --config configs/example_config.yaml --plot figures/stage1_action_rates.pdf
"""

from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import chi2

from bootstrap_utils import ClusteredBootstrap, exact_mcnemar, holm_correction, mean_of
from compute_metrics import (
    ASK,
    EVIDENCE_LEVELS,
    STATE_KEY,
    THREE_ACTIONS,
    UNFULFILLABLE_ASK,
    add_bootstrap_cli_args,
    bootstrap_kwargs,
    label_unfulfillable_ask,
    load_config,
    load_table,
    normalize_evidence_level,
)


# ---------------------------------------------------------------------------
# Building the matched wide table
# ---------------------------------------------------------------------------


def build_wide(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per state, one ``<model>`` action column per model.

    Also adds ``<model>_unfulfillable`` when ``hidden_evidence_count`` exists.
    Raises if the models do not cover identical states.
    """
    wide, reference_keys = None, None
    for name, df in tables.items():
        d = df.copy()
        d["case_id"] = d["case_id"].astype(str)
        d["evidence_level"] = normalize_evidence_level(d["evidence_level"])
        if d.duplicated(STATE_KEY).any():
            raise ValueError(f"{name}: duplicate (case_id, evidence_level) rows.")
        bad = set(d["stage1_action"].unique()) - set(THREE_ACTIONS)
        if bad:
            raise ValueError(f"{name}: invalid Stage 1 actions {sorted(bad)}")
        keys = set(map(tuple, d[STATE_KEY].to_numpy()))
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise ValueError(f"{name}: state set differs from the first model.")
        cols = {"stage1_action": name}
        if "hidden_evidence_count" in d.columns:
            d[f"{name}_unfulfillable"] = label_unfulfillable_ask(d)
            cols[f"{name}_unfulfillable"] = f"{name}_unfulfillable"
        part = d[STATE_KEY + list(cols)].rename(columns=cols)
        wide = part if wide is None else wide.merge(part, on=STATE_KEY, validate="one_to_one")
    return wide.sort_values(STATE_KEY).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Descriptive agreement
# ---------------------------------------------------------------------------


def agreement_structure(wide: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Counts of states by number of distinct actions chosen across models."""
    n_distinct = wide[models].nunique(axis=1)
    rows = [("all_agree", int((n_distinct == 1).sum()))]
    if len(models) == 3:
        rows += [("exactly_two_agree", int((n_distinct == 2).sum())),
                 ("all_differ", int((n_distinct == 3).sum()))]
    rows.append(("any_disagreement", int((n_distinct > 1).sum())))
    out = pd.DataFrame(rows, columns=["pattern", "count"])
    out["rate"] = out["count"] / len(wide)
    return out


def disagreement_by_evidence(wide: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    any_dis = wide[models].nunique(axis=1) > 1
    return (wide.assign(any_disagreement=any_dis)
            .groupby("evidence_level")["any_disagreement"].agg(["sum", "count", "mean"])
            .rename(columns={"sum": "n_disagree", "count": "n_states", "mean": "rate"})
            .reset_index())


def common_patterns(
    wide: pd.DataFrame, models: list[str], top: int = 10, disagreement_only: bool = True
) -> pd.DataFrame:
    """Most frequent joint Stage 1 action patterns.

    By default (``disagreement_only=True``) only states where at least two
    models chose different actions are counted. This is the scope of the
    paper's common-pattern table: its three reported patterns (45, 21 and 19
    states) are the top three among the 137 disagreement states, **not** among
    all 200 states. Over all states, unanimous patterns such as ASK/ASK/ASK
    would also appear in the top three. Set ``disagreement_only=False`` for the
    all-state ranking.

    The returned table has ``scope`` and ``n_states_in_scope`` columns so the
    denominator is explicit.
    """
    in_scope = wide[models].nunique(axis=1) > 1 if disagreement_only else pd.Series(True, index=wide.index)
    data = wide.loc[in_scope]
    out = data.groupby(models).size().rename("count").sort_values(ascending=False).head(top).reset_index()
    out["scope"] = "disagreement_states_only" if disagreement_only else "all_states"
    out["n_states_in_scope"] = int(in_scope.sum())
    out["share_of_scope"] = out["count"] / out["n_states_in_scope"]
    return out


def action_rates(wide: pd.DataFrame, models: list[str], by_evidence: bool = False) -> pd.DataFrame:
    """Stage 1 action rates per model, splitting out unfulfillable ASK when known."""
    rows = []
    groups = wide.groupby("evidence_level") if by_evidence else [("all", wide)]
    for level, grp in groups:
        for m in models:
            act = grp[m].astype(str)
            if f"{m}_unfulfillable" in grp.columns:
                act = act.where(~grp[f"{m}_unfulfillable"].astype(bool), UNFULFILLABLE_ASK)
            counts = act.value_counts()
            for a in (*THREE_ACTIONS, UNFULFILLABLE_ASK):
                rows.append({"evidence_level": level, "model": m, "action": a,
                             "count": int(counts.get(a, 0)), "rate": counts.get(a, 0) / len(grp)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pairwise inference
# ---------------------------------------------------------------------------


def contingency(a: pd.Series, b: pd.Series, categories=THREE_ACTIONS) -> np.ndarray:
    """k x k table: rows = model A action, columns = model B action."""
    table = pd.crosstab(pd.Categorical(a, categories), pd.Categorical(b, categories), dropna=False)
    return table.reindex(index=categories, columns=categories, fill_value=0).to_numpy()


def stuart_maxwell(table: np.ndarray) -> dict:
    """Stuart-Maxwell test of marginal homogeneity for a k x k paired table.

    Uses the closed form with the last category dropped:
    d_i = n_i. - n_.i;  V_ii = n_i. + n_.i - 2 n_ii;  V_ij = -(n_ij + n_ji).
    The statistic d' V^{-1} d is compared to chi-square with k-1 df.
    Returns NaN if V is singular (e.g. a category that is never used).
    """
    t = np.asarray(table, dtype=float)
    k = t.shape[0]
    rows, cols = t.sum(axis=1), t.sum(axis=0)
    d = (rows - cols)[: k - 1]
    v = -(t + t.T)
    np.fill_diagonal(v, rows + cols - 2 * np.diag(t))
    v = v[: k - 1, : k - 1]
    try:
        stat = float(d @ np.linalg.solve(v, d))
    except np.linalg.LinAlgError:
        return {"chi2": float("nan"), "df": k - 1, "p_raw": float("nan")}
    return {"chi2": stat, "df": k - 1, "p_raw": float(chi2.sf(stat, k - 1))}


def pairwise_tests(wide: pd.DataFrame, models: list[str], alpha: float = 0.05, **boot_kwargs) -> pd.DataFrame:
    """Pairwise disagreement (clustered CI) and Holm-corrected Stuart-Maxwell tests."""
    pairs = list(combinations(models, 2))
    flags = wide.copy()
    for a, b in pairs:
        flags[f"dis_{a}_{b}"] = flags[a] != flags[b]
    boot = ClusteredBootstrap(flags, **boot_kwargs)

    rows = []
    for a, b in pairs:
        ci = boot.ci(mean_of(f"dis_{a}_{b}"))
        sm = stuart_maxwell(contingency(wide[a], wide[b]))
        rows.append({"model_a": a, "model_b": b, "disagreement": ci.estimate,
                     "ci_low": ci.ci_low, "ci_high": ci.ci_high, **sm})
    out = pd.DataFrame(rows)
    reject, p_adj = holm_correction(out["p_raw"].fillna(1.0), alpha=alpha)
    out["p_holm"], out["reject_holm"] = p_adj, reject
    return out


def action_specific_mcnemar(wide: pd.DataFrame, models: list[str], alpha: float = 0.05) -> pd.DataFrame:
    """Each action vs. the other two, per model pair. Holm over all tests jointly."""
    rows = []
    for a, b in combinations(models, 2):
        for action in THREE_ACTIONS:
            res = exact_mcnemar(wide[a] == action, wide[b] == action)
            rows.append({
                "model_a": a, "model_b": b, "action": action,
                "rate_a": float((wide[a] == action).mean()),
                "rate_b": float((wide[b] == action).mean()),
                "diff_pp": 100 * float((wide[a] == action).mean() - (wide[b] == action).mean()),
                **res,
            })
    out = pd.DataFrame(rows)
    reject, p_adj = holm_correction(out["p_exact_mcnemar"], alpha=alpha)
    out["p_holm"], out["reject_holm"] = p_adj, reject
    return out


# ---------------------------------------------------------------------------
# Plot: Stage 1 action rates by initial evidence level
# ---------------------------------------------------------------------------

# Fixed categorical order: each action keeps its color and marker in every panel.
ACTION_STYLE = {
    "DECIDE": ("#2a78d6", "o"),
    "ASK": ("#eb6834", "s"),
    "DEFER": ("#1baf7a", "^"),
    UNFULFILLABLE_ASK: ("#eda100", "D"),
}


def plot_action_rates(rates_by_evidence: pd.DataFrame, models: list[str], path: str) -> None:
    """Small multiples (one panel per model) of Stage 1 action rate vs. evidence."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(models), figsize=(3.4 * len(models), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        sub = rates_by_evidence[rates_by_evidence["model"] == m]
        for action, (color, marker) in ACTION_STYLE.items():
            s = sub[sub["action"] == action].sort_values("evidence_level")
            if s.empty or (action == UNFULFILLABLE_ASK and s["count"].sum() == 0):
                continue
            ax.plot(s["evidence_level"], 100 * s["rate"], color=color, marker=marker,
                    linewidth=2, markersize=6, label=action.replace("_", " ").title())
        ax.set_title(m, fontsize=10)
        ax.set_xticks(list(EVIDENCE_LEVELS))
        ax.set_xlabel("Initial evidence (%)")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Stage 1 action rate (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_states(items: list[str]) -> dict[str, pd.DataFrame]:
    tables = {}
    for item in items:
        name, _, path = item.partition("=")
        if not path:
            raise ValueError(f"Expected NAME=PATH, got {item!r}")
        tables[name] = load_table(path)
    return tables


def main() -> None:
    p = argparse.ArgumentParser(description="Cross-model Stage 1 action comparison.")
    p.add_argument("--states", nargs="+", required=True, help="NAME=PATH per model, e.g. qwen=qwen_states.csv")
    p.add_argument("--config", default=None)
    p.add_argument("--plot", default=None, help="Optional output path for the action-rate figure.")
    add_bootstrap_cli_args(p)
    args = p.parse_args()

    config = load_config(args.config)
    alpha = float(config.get("alpha", 0.05))
    tables = parse_states(args.states)
    models = list(tables)
    wide = build_wide(tables)

    print(f"{len(wide)} matched states, {wide['case_id'].nunique()} cases, models: {models}\n")
    print(agreement_structure(wide, models).to_string(index=False))
    print("\nAny disagreement by evidence level:")
    print(disagreement_by_evidence(wide, models).to_string(index=False))
    patterns = common_patterns(wide, models, disagreement_only=True)
    print(f"\nMost frequent Stage 1 action patterns - scope: DISAGREEMENT STATES ONLY "
          f"(n = {patterns['n_states_in_scope'].iloc[0]} of {len(wide)} states):")
    print(patterns.drop(columns=["scope", "n_states_in_scope"]).to_string(index=False))
    print("\nOverall Stage 1 action rates (ASK = executable ASK; UNFULFILLABLE_ASK = ASK with an empty hidden pool):")
    print(action_rates(wide, models).pivot(index="model", columns="action", values="rate").to_string())
    print("\nPairwise disagreement and Stuart-Maxwell tests:")
    boot = bootstrap_kwargs(config, "cross_model", args.seed, args.case_order)
    print(pairwise_tests(wide, models, alpha=alpha, **boot).to_string(index=False))
    print("\nAction-specific exact McNemar tests:")
    print(action_specific_mcnemar(wide, models, alpha=alpha).to_string(index=False))

    if args.plot:
        plot_action_rates(action_rates(wide, models, by_evidence=True), models, args.plot)
        print(f"\nSaved figure to {args.plot}")


if __name__ == "__main__":
    main()
