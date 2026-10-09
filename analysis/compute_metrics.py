"""Primary decision metrics for the DECIDE / ASK / DEFER workshop protocol.

All metrics are computed offline from frozen, state-level model outputs.
Nothing here calls a model API. This repository ships no raw data. The
functions document and check the input schema they expect.

Expected input: one state table per model and condition (CSV or JSONL)
----------------------------------------------------------------------
One row per matched evidence state; ``(case_id, evidence_level)`` is unique.

==========================  =====================================================
column                      meaning
==========================  =====================================================
case_id                     DDXPlus case identifier (string). Resampling unit.
evidence_level              Initial evidence level: 25/50/75/100 or 0.25-1.0.
gold_diagnosis              Canonical DDXPlus pathology label.
hidden_evidence_count       Size of the case-specific hidden evidence pool H_t.
stage1_action               DECIDE | ASK | DEFER (three-action condition), or
                            DECIDE | ABSTAIN (binary baseline).
stage1_diagnosis            Diagnosis returned with the Stage 1 response.
post_ask_action             (optional) DECIDE | DEFER after an executable ASK.
post_ask_diagnosis          (optional) Diagnosis returned after the ASK answer.
==========================  =====================================================

Diagnosis correctness is exact, case-sensitive string equality with the
canonical label: no whitespace or case normalization and no fuzzy matching.
A diagnosis is only scored when the action is DECIDE. Rates with a zero
denominator are NaN, never 0.

Row order matters for the bootstrap: the paper's Step 5 intervals use
``case_order="first_appearance"`` with seed 20260830. Pass rows in
evaluation-set sampling order (cases in sampling order, then evidence level)
to reproduce them.

Metric definitions (N = all evaluated states)
---------------------------------------------
- Conditional decision accuracy: correct DECIDE / all DECIDE
- Autonomous coverage:           all DECIDE / N
- Correct autonomous coverage:   correct DECIDE / N
- Wrong decision rate:           incorrect DECIDE / N
- Unsafe premature decision rate: incorrect DECIDE with initial evidence < 100%
                                  / all states with initial evidence < 100%

By construction, autonomous coverage = correct autonomous coverage + wrong
decision rate. For the final targeted policy, "premature" is stratified by the
*initial* evidence level. One acquired item can complete the record, so it
does not mean the evidence was still incomplete at decision time.

Usage
-----
    python analysis/compute_metrics.py --states path/to/qwen_states.csv \
        --stage final --config configs/example_config.yaml --by-evidence

Bootstrap settings come from ``bootstrap.step5_metrics`` in the config
(seed 20260830, first-appearance case order) unless overridden with
``--seed`` / ``--case-order``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import pandas as pd
import yaml

from bootstrap_utils import ClusteredBootstrap, safe_ratio

DECIDE, ASK, DEFER, ABSTAIN = "DECIDE", "ASK", "DEFER", "ABSTAIN"
UNFULFILLABLE_ASK = "UNFULFILLABLE_ASK"
THREE_ACTIONS = (DECIDE, ASK, DEFER)
BINARY_ACTIONS = (DECIDE, ABSTAIN)
POST_ASK_ACTIONS = (DECIDE, DEFER)
FINAL_ACTIONS = (DECIDE, DEFER, UNFULFILLABLE_ASK)
EVIDENCE_LEVELS = (25, 50, 75, 100)
STATE_KEY = ["case_id", "evidence_level"]

STATE_COLUMNS = [
    "case_id",
    "evidence_level",
    "gold_diagnosis",
    "hidden_evidence_count",
    "stage1_action",
    "stage1_diagnosis",
]


# ---------------------------------------------------------------------------
# I/O and validation
# ---------------------------------------------------------------------------


def load_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV or JSONL state table."""
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, dtype={"case_id": str})
    if path.suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True, dtype={"case_id": str})
    raise ValueError(f"Unsupported file type: {path.suffix} (expected .csv or .jsonl)")


def load_config(path: str | Path | None) -> dict:
    """Read the public YAML config. Returns an empty dict if no path is given."""
    if path is None:
        return {}
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def bootstrap_kwargs(
    config: dict,
    analysis: str,
    seed: int | None = None,
    case_order: str | None = None,
) -> dict:
    """:class:`ClusteredBootstrap` keyword arguments for one analysis.

    Reads the shared settings under ``bootstrap`` and the per-analysis
    ``seed`` / ``case_order`` under ``bootstrap.<analysis>`` in the config.
    The ``seed`` and ``case_order`` arguments (e.g. from the command line)
    take precedence. Raises if either is still missing: there is no default
    bootstrap seed.
    """
    boot = config.get("bootstrap", {}) or {}
    section = boot.get(analysis, {}) or {}
    kwargs = {
        "seed": seed if seed is not None else section.get("seed"),
        "case_order": case_order if case_order is not None else section.get("case_order"),
    }
    missing = [k for k, v in kwargs.items() if v is None]
    if missing:
        raise ValueError(
            f"No bootstrap {' / '.join(missing)} for analysis {analysis!r}. Set bootstrap.{analysis} "
            "in the config or pass --seed / --case-order explicitly."
        )
    kwargs["seed"] = int(kwargs["seed"])
    if "n_boot" in boot:
        kwargs["n_boot"] = int(boot["n_boot"])
    if "ci_level" in boot:
        kwargs["ci_level"] = float(boot["ci_level"])
    if "cluster_col" in boot:
        kwargs["cluster_col"] = str(boot["cluster_col"])
    return kwargs


def add_bootstrap_cli_args(parser: argparse.ArgumentParser) -> None:
    """Shared ``--seed`` / ``--case-order`` overrides for the analysis CLIs."""
    parser.add_argument("--seed", type=int, default=None, help="Bootstrap seed (overrides the config).")
    parser.add_argument("--case-order", choices=["sorted", "first_appearance"], default=None,
                        help="Case order for bootstrap draws (overrides the config).")


def normalize_evidence_level(series: pd.Series) -> pd.Series:
    """Return evidence levels as integer percentages (25, 50, 75, 100)."""
    values = pd.to_numeric(series)
    if values.max() <= 1.0:
        values = values * 100
    out = values.round().astype(int)
    unexpected = set(out.unique()) - set(EVIDENCE_LEVELS)
    if unexpected:
        raise ValueError(f"Unexpected evidence levels: {sorted(unexpected)}")
    return out


def validate_state_table(
    df: pd.DataFrame,
    required: list[str] = STATE_COLUMNS,
    allowed_stage1: tuple[str, ...] = THREE_ACTIONS,
) -> pd.DataFrame:
    """Check the schema and return a copy with normalized evidence levels."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"State table is missing columns: {missing}")
    out = df.copy()
    out["case_id"] = out["case_id"].astype(str)
    out["evidence_level"] = normalize_evidence_level(out["evidence_level"])
    if out.duplicated(STATE_KEY).any():
        raise ValueError("Duplicate (case_id, evidence_level) rows.")
    bad = set(out["stage1_action"].unique()) - set(allowed_stage1)
    if bad:
        raise ValueError(f"Invalid Stage 1 actions {sorted(bad)}; allowed {allowed_stage1}")
    return out


# ---------------------------------------------------------------------------
# Deriving outcomes
# ---------------------------------------------------------------------------


def label_unfulfillable_ask(df: pd.DataFrame) -> pd.Series:
    """True where Stage 1 chose ASK but the hidden evidence pool was empty.

    The observed action is kept as is. It is never converted post hoc to
    DECIDE, DEFER or ABSTAIN.
    """
    return (df["stage1_action"] == ASK) & (df["hidden_evidence_count"] == 0)


def compose_final_policy(df: pd.DataFrame) -> pd.DataFrame:
    """Final outcome of the one-step targeted ASK policy.

    - Stage 1 DECIDE / DEFER   -> unchanged
    - Executable Stage 1 ASK   -> post-ASK DECIDE / DEFER
    - Unfulfillable Stage 1 ASK -> UNFULFILLABLE_ASK (unresolved)

    Adds ``final_action`` and ``final_diagnosis`` columns.
    """
    out = df.copy()
    unfulfillable = label_unfulfillable_ask(out)
    executable = (out["stage1_action"] == ASK) & ~unfulfillable

    if executable.any():
        for col in ("post_ask_action", "post_ask_diagnosis"):
            if col not in out.columns:
                raise KeyError(f"Executable ASK states require column {col!r}.")
        post = out.loc[executable, "post_ask_action"]
        bad = set(post.dropna().unique()) - set(POST_ASK_ACTIONS)
        if bad or post.isna().any():
            raise ValueError(f"Post-ASK actions must be DECIDE or DEFER; found {sorted(bad)} / missing values.")

    out["final_action"] = out["stage1_action"]
    out["final_diagnosis"] = out["stage1_diagnosis"]
    out.loc[executable, "final_action"] = out.loc[executable, "post_ask_action"]
    out.loc[executable, "final_diagnosis"] = out.loc[executable, "post_ask_diagnosis"]
    out.loc[unfulfillable, "final_action"] = UNFULFILLABLE_ASK
    out.loc[unfulfillable, "final_diagnosis"] = None
    return out


def exact_label_match(diagnosis: pd.Series, gold: pd.Series) -> pd.Series:
    """Exact, case-sensitive string equality with no normalization.

    Missing diagnoses never match.
    """
    return diagnosis.notna() & gold.notna() & (diagnosis.astype(object) == gold.astype(object))


def outcome_flags(
    df: pd.DataFrame,
    action_col: str,
    diagnosis_col: str,
    gold_col: str = "gold_diagnosis",
) -> pd.DataFrame:
    """Add boolean outcome flags used by every metric.

    Adds ``action``, ``decide``, ``correct_decide``, ``wrong_decide`` and
    ``incomplete`` (initial evidence < 100%).
    """
    out = df.copy()
    out["action"] = out[action_col]
    out["decide"] = out[action_col] == DECIDE
    out["correct_decide"] = out["decide"] & exact_label_match(out[diagnosis_col], out[gold_col])
    out["wrong_decide"] = out["decide"] & ~out["correct_decide"]
    out["incomplete"] = out["evidence_level"] < 100
    return out


# ---------------------------------------------------------------------------
# Metrics (each takes a flagged table and returns a fraction or NaN)
# ---------------------------------------------------------------------------


def conditional_decision_accuracy(d: pd.DataFrame) -> float:
    return safe_ratio(d["correct_decide"].sum(), d["decide"].sum())


def autonomous_coverage(d: pd.DataFrame) -> float:
    return safe_ratio(d["decide"].sum(), len(d))


def correct_autonomous_coverage(d: pd.DataFrame) -> float:
    return safe_ratio(d["correct_decide"].sum(), len(d))


def wrong_decision_rate(d: pd.DataFrame) -> float:
    return safe_ratio(d["wrong_decide"].sum(), len(d))


def unsafe_premature_decision_rate(d: pd.DataFrame) -> float:
    inc = d["incomplete"]
    return safe_ratio((d["wrong_decide"] & inc).sum(), inc.sum())


def premature_error_given_decision(d: pd.DataFrame) -> float:
    """Wrong DECIDE among DECIDE at initial evidence < 100%."""
    inc = d["incomplete"]
    return safe_ratio((d["wrong_decide"] & inc).sum(), (d["decide"] & inc).sum())


def full_evidence_wrong_decision_rate(d: pd.DataFrame) -> float:
    full = ~d["incomplete"]
    return safe_ratio((d["wrong_decide"] & full).sum(), full.sum())


def action_rate(action: str) -> Callable[[pd.DataFrame], float]:
    return lambda d: safe_ratio((d["action"] == action).sum(), len(d))


PRIMARY_METRICS: dict[str, Callable[[pd.DataFrame], float]] = {
    "conditional_decision_accuracy": conditional_decision_accuracy,
    "autonomous_coverage": autonomous_coverage,
    "correct_autonomous_coverage": correct_autonomous_coverage,
    "wrong_decision_rate": wrong_decision_rate,
    "unsafe_premature_decision_rate": unsafe_premature_decision_rate,
    "premature_error_given_decision": premature_error_given_decision,
    "full_evidence_wrong_decision_rate": full_evidence_wrong_decision_rate,
}


def metric_suite(actions: tuple[str, ...]) -> dict[str, Callable[[pd.DataFrame], float]]:
    """Primary metrics plus one rate per action in ``actions``."""
    suite = dict(PRIMARY_METRICS)
    for a in actions:
        suite[f"{a.lower()}_rate"] = action_rate(a)
    return suite


def summarize(
    flagged: pd.DataFrame,
    actions: tuple[str, ...],
    with_ci: bool = True,
    **boot_kwargs,
) -> pd.DataFrame:
    """Point estimates (and optional case-clustered CIs) for one stratum."""
    suite = metric_suite(actions)
    rows = []
    boot = ClusteredBootstrap(flagged, **boot_kwargs) if with_ci else None
    for name, fn in suite.items():
        if boot is not None:
            res = boot.ci(fn)
            rows.append({"metric": name, "estimate": res.estimate, "ci_low": res.ci_low,
                         "ci_high": res.ci_high, "n_valid_boot": res.n_valid})
        else:
            rows.append({"metric": name, "estimate": fn(flagged)})
    out = pd.DataFrame(rows)
    out["n_states"] = len(flagged)
    out["n_cases"] = flagged["case_id"].nunique()
    return out


def summarize_by_evidence(flagged: pd.DataFrame, actions: tuple[str, ...]) -> pd.DataFrame:
    """Point estimates per initial evidence level (no CIs)."""
    parts = []
    for level, grp in flagged.groupby("evidence_level"):
        s = summarize(grp, actions, with_ci=False)
        s.insert(0, "evidence_level", level)
        parts.append(s)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--states", required=True, help="State table (CSV/JSONL) for one model and condition.")
    p.add_argument("--stage", choices=["stage1", "final", "binary"], default="stage1",
                   help="stage1: Stage 1 actions; final: targeted one-step ASK policy; binary: DECIDE/ABSTAIN baseline.")
    p.add_argument("--config", default=None)
    p.add_argument("--by-evidence", action="store_true", help="Also report point estimates per evidence level.")
    p.add_argument("--no-ci", action="store_true", help="Skip bootstrap intervals.")
    add_bootstrap_cli_args(p)
    p.add_argument("--output", default=None, help="Optional CSV path for the summary.")
    args = p.parse_args()

    config = load_config(args.config)
    allowed = BINARY_ACTIONS if args.stage == "binary" else THREE_ACTIONS
    states = validate_state_table(load_table(args.states), allowed_stage1=allowed)

    if args.stage == "final":
        states = compose_final_policy(states)
        flagged, actions = outcome_flags(states, "final_action", "final_diagnosis"), FINAL_ACTIONS
    else:
        if args.stage == "stage1":
            states = states.assign(stage1_unfulfillable_ask=label_unfulfillable_ask(states))
        flagged, actions = outcome_flags(states, "stage1_action", "stage1_diagnosis"), allowed

    boot = {} if args.no_ci else bootstrap_kwargs(config, "step5_metrics", args.seed, args.case_order)
    summary = summarize(flagged, actions, with_ci=not args.no_ci, **boot)
    if args.stage == "stage1":
        n_unf = int(flagged["stage1_unfulfillable_ask"].sum())
        print(f"Unfulfillable ASK at Stage 1: {n_unf} / {len(flagged)} states")
    print(summary.to_string(index=False))

    if args.by_evidence:
        print()
        print(summarize_by_evidence(flagged, actions).to_string(index=False))
    if args.output:
        summary.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
