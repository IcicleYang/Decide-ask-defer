"""Targeted vs. matched random ASK comparison (Qwen only in the paper).

For each executable Stage 1 ASK (non-empty hidden pool H_t), the *targeted*
condition reveals the evidence item the model selected. The *random* condition
reveals one item drawn uniformly from the same H_t. Both answers come from the
same case record, and the same post-ASK DECIDE / DEFER step follows. One
random comparator is drawn per executable ASK. The analysis therefore does not
average over repeated random draws.

The primary analysis keeps pairs where the random draw equals the targeted
item, since that is a valid randomized outcome. The sensitivity analysis drops
those pairs (``--distinct-only``).

Expected input: one pair table (CSV or JSONL), one row per executable ASK
------------------------------------------------------------------------
==========================  ===============================================
column                      meaning
==========================  ===============================================
case_id                     DDXPlus case identifier (resampling unit)
evidence_level              Initial evidence level of the state
gold_diagnosis              Canonical DDXPlus label
targeted_evidence_id        Item selected by the model at Stage 2
random_evidence_id          Item drawn uniformly from the same H_t
targeted_action             Post-ASK action under targeted info: DECIDE|DEFER
targeted_diagnosis          Diagnosis under targeted info (if DECIDE)
random_action               Post-ASK action under random info: DECIDE|DEFER
random_diagnosis            Diagnosis under random info (if DECIDE)
==========================  ===============================================

Reported quantities
-------------------
- Post-ASK DECIDE rate and correct DECIDE rate: targeted, random, difference
  (pp), case-clustered bootstrap CI of the difference, exact McNemar p.
- Conditional accuracy among final DECIDE and DEFER rate: descriptive only.
  DEFER is the complement of DECIDE in the one-step protocol, and conditional
  accuracy has different conditioning sets per arm, so no paired test is run.
- Pair-level ordering correct DECIDE > DEFER > incorrect DECIDE, with an exact
  sign test on non-tied pairs. The ordering only classifies transitions. It is
  not a clinical utility function and does not imply that DEFER is clinically
  appropriate.

Bootstrap provenance (unverified)
--------------------------------
The paper's targeted-vs-random intervals were computed by a separate script
that is not part of this repository. Its bootstrap seed and case order have
not been verified, so ``bootstrap.ask_targeted_vs_random`` in the example
config is left empty and this script requires ``--seed`` and
``--case-order`` explicitly. Intervals from this script are valid
case-clustered intervals but are **not** guaranteed to equal the paper's.
Point estimates, McNemar p-values, the sign test and win/tie/loss counts do
not depend on the bootstrap.

Usage
-----
    python analysis/ask_targeted_vs_random.py --pairs path/to/qwen_pairs.csv \
        --config configs/example_config.yaml --seed <SEED> --case-order sorted [--distinct-only]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from bootstrap_utils import ClusteredBootstrap, exact_mcnemar, exact_sign_test, mean_difference, safe_ratio
from compute_metrics import (
    DECIDE,
    DEFER,
    POST_ASK_ACTIONS,
    STATE_KEY,
    add_bootstrap_cli_args,
    bootstrap_kwargs,
    exact_label_match,
    load_config,
    load_table,
    normalize_evidence_level,
)

PAIR_COLUMNS = [
    "case_id",
    "evidence_level",
    "gold_diagnosis",
    "targeted_evidence_id",
    "random_evidence_id",
    "targeted_action",
    "targeted_diagnosis",
    "random_action",
    "random_diagnosis",
]

ARMS = ("targeted", "random")


def validate_pairs(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in PAIR_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Pair table is missing columns: {missing}")
    out = df.copy()
    out["case_id"] = out["case_id"].astype(str)
    out["evidence_level"] = normalize_evidence_level(out["evidence_level"])
    if out.duplicated(STATE_KEY).any():
        raise ValueError("Duplicate (case_id, evidence_level) pairs.")
    if (out["evidence_level"] == 100).any():
        raise ValueError("Executable ASK requires a non-empty hidden pool; 100% states should not appear.")
    for arm in ARMS:
        bad = set(out[f"{arm}_action"].unique()) - set(POST_ASK_ACTIONS)
        if bad:
            raise ValueError(f"{arm}_action must be DECIDE or DEFER; found {sorted(bad)}")
    return out


def add_pair_flags(pairs: pd.DataFrame) -> pd.DataFrame:
    """Add per-arm DECIDE / correct / DEFER flags and the same-item indicator."""
    out = pairs.copy()
    for arm in ARMS:
        decide = out[f"{arm}_action"] == DECIDE
        match = exact_label_match(out[f"{arm}_diagnosis"], out["gold_diagnosis"])
        out[f"{arm}_decide"] = decide
        out[f"{arm}_correct"] = decide & match
        out[f"{arm}_defer"] = out[f"{arm}_action"] == DEFER
        # Ordinal outcome: correct DECIDE (2) > DEFER (1) > incorrect DECIDE (0)
        out[f"{arm}_rank"] = np.select([out[f"{arm}_correct"], out[f"{arm}_defer"]], [2, 1], default=0)
    out["same_item"] = out["targeted_evidence_id"].astype(str) == out["random_evidence_id"].astype(str)
    return out


def compare_targeted_random(flagged: pd.DataFrame, **boot_kwargs) -> pd.DataFrame:
    """Targeted vs. random summary table (rates as fractions, differences in pp)."""
    boot = ClusteredBootstrap(flagged, **boot_kwargs)
    rows = []

    for label, suffix in [("post_ask_decide", "decide"), ("correct_decide", "correct")]:
        t, r = f"targeted_{suffix}", f"random_{suffix}"
        diff = boot.ci(mean_difference(t, r))
        test = exact_mcnemar(flagged[t], flagged[r])
        rows.append({
            "metric": label,
            "targeted": flagged[t].mean(),
            "random": flagged[r].mean(),
            "diff_pp": 100 * diff.estimate,
            "ci_low_pp": 100 * diff.ci_low,
            "ci_high_pp": 100 * diff.ci_high,
            "p_exact_mcnemar": test["p_exact_mcnemar"],
            "discordant": test["discordant"],
        })

    t_acc = safe_ratio(flagged["targeted_correct"].sum(), flagged["targeted_decide"].sum())
    r_acc = safe_ratio(flagged["random_correct"].sum(), flagged["random_decide"].sum())
    rows.append({"metric": "conditional_accuracy", "targeted": t_acc, "random": r_acc,
                 "diff_pp": 100 * (t_acc - r_acc)})
    t_def, r_def = flagged["targeted_defer"].mean(), flagged["random_defer"].mean()
    rows.append({"metric": "defer", "targeted": t_def, "random": r_def, "diff_pp": 100 * (t_def - r_def)})

    out = pd.DataFrame(rows)
    out["n_pairs"] = len(flagged)
    out["n_same_item"] = int(flagged["same_item"].sum())
    return out


def win_tie_loss(flagged: pd.DataFrame) -> dict:
    """Pair-level ordering from the targeted perspective plus exact sign test."""
    wins = int((flagged["targeted_rank"] > flagged["random_rank"]).sum())
    losses = int((flagged["targeted_rank"] < flagged["random_rank"]).sum())
    ties = len(flagged) - wins - losses
    return {
        "n_pairs": len(flagged),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "p_sign_test": exact_sign_test(wins, losses),
    }


def run(pairs: pd.DataFrame, distinct_only: bool = False, **boot_kwargs) -> tuple[pd.DataFrame, dict]:
    flagged = add_pair_flags(validate_pairs(pairs))
    if distinct_only:
        flagged = flagged.loc[~flagged["same_item"]].reset_index(drop=True)
    return compare_targeted_random(flagged, **boot_kwargs), win_tie_loss(flagged)


def main() -> None:
    p = argparse.ArgumentParser(description="Targeted vs. random ASK comparison.")
    p.add_argument("--pairs", required=True, help="Pair table (CSV/JSONL), one row per executable ASK.")
    p.add_argument("--config", default=None)
    p.add_argument("--distinct-only", action="store_true",
                   help="Sensitivity analysis: drop pairs whose random draw equals the targeted item.")
    p.add_argument("--output", default=None)
    add_bootstrap_cli_args(p)
    args = p.parse_args()

    config = load_config(args.config)
    boot = bootstrap_kwargs(config, "ask_targeted_vs_random", args.seed, args.case_order)
    summary, wtl = run(load_table(args.pairs), args.distinct_only, **boot)
    print(summary.to_string(index=False))
    print("\nPair-level ordering (correct DECIDE > DEFER > incorrect DECIDE):", wtl)
    if args.output:
        summary.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
