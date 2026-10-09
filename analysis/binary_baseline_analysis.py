"""Matched DECIDE / ABSTAIN baseline vs. the three-action final policy (Qwen only in the paper).

The binary baseline uses the same model, provider, cases, evidence states and
revealed evidence as the three-action condition. Only the action space differs.
ABSTAIN ends the interaction and no information is acquired. Differences are
reported as *three-action final targeted policy minus binary baseline*.

Expected inputs (CSV or JSONL)
------------------------------
``--three-action``: state table in the format of ``compute_metrics.py``, with
``stage1_action`` in {DECIDE, ASK, DEFER} plus ``post_ask_action`` /
``post_ask_diagnosis`` for executable ASK states (targeted condition).

``--binary``: state table with the same ``case_id``, ``evidence_level`` and
``gold_diagnosis`` values, ``stage1_action`` in {DECIDE, ABSTAIN} and
``stage1_diagnosis``. ``hidden_evidence_count`` is accepted but not used.

The two tables must cover exactly the same set of states. To reproduce the
paper's intervals, pass the three-action rows in evaluation-set sampling
order: bootstrap settings come from ``bootstrap.step5_metrics`` (seed
20260830, first-appearance case order). The matched table keeps the
three-action row order.

Transition definitions (operational, relative to the binary baseline)
---------------------------------------------------------------------
- Wrong decision rescue: binary incorrect DECIDE, and the three-action policy
  ends in a correct DECIDE or in DEFER. Denominator: binary incorrect DECIDE.
- Lost correct autonomy: binary correct DECIDE, and the three-action policy
  does not end in a correct DECIDE. Denominator: binary correct DECIDE.
- Abstention recovery: binary ABSTAIN, and the three-action policy ends in a
  *correct* DECIDE. Denominator: binary ABSTAIN. This is the frozen
  definition behind the paper's reported 15 / 63. Counting any DECIDE,
  correct or not, would give a larger number.

These categories describe changes relative to the baseline. A "rescue" does
not mean DEFER was clinically correct. DDXPlus has no ground-truth label for
whether deferral was appropriate.

Usage
-----
    python analysis/binary_baseline_analysis.py --three-action qwen_states.csv \
        --binary qwen_binary_states.csv --config configs/example_config.yaml
"""

from __future__ import annotations

import argparse

import pandas as pd

from bootstrap_utils import ClusteredBootstrap, exact_mcnemar, mean_difference, safe_ratio
from compute_metrics import (
    ABSTAIN,
    BINARY_ACTIONS,
    DEFER,
    STATE_COLUMNS,
    STATE_KEY,
    add_bootstrap_cli_args,
    bootstrap_kwargs,
    compose_final_policy,
    load_config,
    load_table,
    outcome_flags,
    summarize,
    validate_state_table,
)

FLAG_COLS = ["action", "decide", "correct_decide", "wrong_decide"]


def build_matched_table(three_action: pd.DataFrame, binary: pd.DataFrame) -> pd.DataFrame:
    """Join final three-action and binary outcomes on the state key.

    Output columns: ``case_id``, ``evidence_level``, ``incomplete`` and
    ``tri_*`` / ``bin_*`` copies of the outcome flags.
    """
    tri = validate_state_table(three_action)
    tri = outcome_flags(compose_final_policy(tri), "final_action", "final_diagnosis")

    bin_required = [c for c in STATE_COLUMNS if c != "hidden_evidence_count"]
    bin_ = validate_state_table(binary, required=bin_required, allowed_stage1=BINARY_ACTIONS)
    bin_ = outcome_flags(bin_, "stage1_action", "stage1_diagnosis")

    tri_keys = set(map(tuple, tri[STATE_KEY].to_numpy()))
    bin_keys = set(map(tuple, bin_[STATE_KEY].to_numpy()))
    if tri_keys != bin_keys:
        raise ValueError(
            f"State sets differ: {len(tri_keys - bin_keys)} only in three-action, "
            f"{len(bin_keys - tri_keys)} only in binary."
        )

    left = tri[STATE_KEY + ["gold_diagnosis", "incomplete"] + FLAG_COLS].rename(
        columns={c: f"tri_{c}" for c in FLAG_COLS})
    right = bin_[STATE_KEY + ["gold_diagnosis"] + FLAG_COLS].rename(
        columns={c: f"bin_{c}" for c in FLAG_COLS} | {"gold_diagnosis": "bin_gold_diagnosis"})
    matched = left.merge(right, on=STATE_KEY, how="inner", validate="one_to_one")
    if (matched["gold_diagnosis"] != matched["bin_gold_diagnosis"]).any():
        raise ValueError("Gold diagnosis differs between matched conditions.")
    return matched.drop(columns="bin_gold_diagnosis")


def binary_baseline_summary(binary: pd.DataFrame, **boot_kwargs) -> pd.DataFrame:
    """Primary metrics for the binary baseline alone (includes ABSTAIN rate)."""
    bin_required = [c for c in STATE_COLUMNS if c != "hidden_evidence_count"]
    bin_ = validate_state_table(binary, required=bin_required, allowed_stage1=BINARY_ACTIONS)
    flagged = outcome_flags(bin_, "stage1_action", "stage1_diagnosis")
    return summarize(flagged, BINARY_ACTIONS, **boot_kwargs)


def matched_differences(matched: pd.DataFrame, **boot_kwargs) -> pd.DataFrame:
    """Three-action minus binary differences with clustered CIs and McNemar p."""
    rows = []
    full = matched.loc[~matched["incomplete"]].reset_index(drop=True)
    specs = [
        ("autonomous_coverage", "decide", matched, False),
        ("correct_autonomous_coverage", "correct_decide", matched, True),
        ("wrong_decision_rate", "wrong_decide", matched, True),
        ("full_evidence_wrong_decision_rate", "wrong_decide", full, False),
    ]
    boots = {id(matched): ClusteredBootstrap(matched, **boot_kwargs)}
    if not full.empty:
        boots[id(full)] = ClusteredBootstrap(full, **boot_kwargs)
    for name, flag, data, test in specs:
        if data.empty:
            continue
        res = boots[id(data)].ci(mean_difference(f"tri_{flag}", f"bin_{flag}"))
        row = {
            "metric": name,
            "binary": data[f"bin_{flag}"].mean(),
            "three_action": data[f"tri_{flag}"].mean(),
            "diff_pp": 100 * res.estimate,
            "ci_low_pp": 100 * res.ci_low,
            "ci_high_pp": 100 * res.ci_high,
            "n_states": len(data),
        }
        if test:
            row["p_exact_mcnemar"] = exact_mcnemar(data[f"tri_{flag}"], data[f"bin_{flag}"])["p_exact_mcnemar"]
        rows.append(row)
    return pd.DataFrame(rows)


def transitions(matched: pd.DataFrame) -> pd.DataFrame:
    """Wrong decision rescue, lost correct autonomy and abstention recovery."""
    bin_wrong = matched["bin_wrong_decide"]
    bin_correct = matched["bin_correct_decide"]
    bin_abstain = matched["bin_action"] == ABSTAIN
    tri_correct = matched["tri_correct_decide"]
    tri_defer = matched["tri_action"] == DEFER

    rows = [
        ("wrong_decision_rescue", (bin_wrong & (tri_correct | tri_defer)).sum(), bin_wrong.sum()),
        ("lost_correct_autonomy", (bin_correct & ~tri_correct).sum(), bin_correct.sum()),
        ("abstention_recovery", (bin_abstain & tri_correct).sum(), bin_abstain.sum()),
    ]
    return pd.DataFrame(
        [{"transition": n, "count": int(k), "denominator": int(d), "rate": safe_ratio(k, d)} for n, k, d in rows]
    )


def transition_matrix(matched: pd.DataFrame) -> pd.DataFrame:
    """Cross-tab of binary outcome vs. three-action final outcome."""
    def label(prefix: str) -> pd.Series:
        lab = matched[f"{prefix}_action"].astype(str)
        lab = lab.where(~matched[f"{prefix}_correct_decide"], "DECIDE_correct")
        return lab.where(~matched[f"{prefix}_wrong_decide"], "DECIDE_wrong")
    return pd.crosstab(label("bin").rename("binary"), label("tri").rename("three_action"))


def main() -> None:
    p = argparse.ArgumentParser(description="Matched binary baseline comparison.")
    p.add_argument("--three-action", required=True)
    p.add_argument("--binary", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--output-prefix", default=None, help="Write <prefix>_differences.csv and <prefix>_transitions.csv")
    add_bootstrap_cli_args(p)
    args = p.parse_args()

    kw = bootstrap_kwargs(load_config(args.config), "step5_metrics", args.seed, args.case_order)
    three, binary = load_table(args.three_action), load_table(args.binary)
    matched = build_matched_table(three, binary)

    print("Binary baseline metrics:")
    print(binary_baseline_summary(binary, **kw).to_string(index=False))
    diffs = matched_differences(matched, **kw)
    print("\nThree-action final policy minus binary baseline:")
    print(diffs.to_string(index=False))
    trans = transitions(matched)
    print("\nMatched transitions:")
    print(trans.to_string(index=False))
    print("\nTransition matrix:")
    print(transition_matrix(matched).to_string())

    if args.output_prefix:
        diffs.to_csv(f"{args.output_prefix}_differences.csv", index=False)
        trans.to_csv(f"{args.output_prefix}_transitions.csv", index=False)


if __name__ == "__main__":
    main()
