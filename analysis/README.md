# Analysis

Offline analysis code for the workshop paper. Every function works on frozen, state-level model outputs and makes **no model API calls**. **No raw data or model outputs are distributed here.** Each script documents the input schema it expects at the top of the file and checks it at load time.

| File | Purpose | Scope in the paper |
|---|---|---|
| `bootstrap_utils.py` | Case-clustered bootstrap (shared draws), exact McNemar, Holm correction, exact sign test | All analyses |
| `compute_metrics.py` | Conditional decision accuracy, autonomous coverage, correct autonomous coverage, wrong decision rate, unsafe premature decision rate, action rates, final one-step policy composition | All models |
| `cross_model_analysis.py` | Agreement structure, common action patterns (disagreement states only by default), pairwise disagreement, Stuart–Maxwell tests, action-specific McNemar tests, Holm correction, Stage 1 action-rate plot | Qwen, Gemini, GPT |
| `ask_targeted_vs_random.py` | Targeted vs. matched random ASK, distinct-item sensitivity, pair-level win/tie/loss | **Qwen only** |
| `binary_baseline_analysis.py` | Matched DECIDE/ABSTAIN baseline vs. three-action final policy, plus matched transitions | **Qwen only** |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the scripts from the repository root, e.g. `python analysis/compute_metrics.py --help`. They import each other as sibling modules.

## Input tables

**State table** (`compute_metrics.py`, `binary_baseline_analysis.py`, `cross_model_analysis.py`). One row per `(case_id, evidence_level)`, one file per model and condition:

| Column | Description |
|---|---|
| `case_id` | DDXPlus case identifier, also the bootstrap resampling unit |
| `evidence_level` | Initial evidence level: 25 / 50 / 75 / 100 (or 0.25–1.0) |
| `gold_diagnosis` | Canonical DDXPlus pathology label |
| `hidden_evidence_count` | Size of the case-specific hidden pool \|H_t\|. It is 0 at 100% |
| `stage1_action` | `DECIDE` / `ASK` / `DEFER`, or `DECIDE` / `ABSTAIN` for the binary baseline |
| `stage1_diagnosis` | Diagnosis returned at Stage 1 |
| `post_ask_action` | Optional: `DECIDE` / `DEFER` after an executable ASK (targeted condition) |
| `post_ask_diagnosis` | Optional: diagnosis after the ASK answer |

**Pair table** (`ask_targeted_vs_random.py`). One row per executable ASK, with `targeted_*` and `random_*` columns for the evidence ID, post-ASK action and diagnosis. See the module docstring.

## Conventions

- Diagnosis correctness is exact, case-sensitive string equality with the canonical label: no whitespace or case normalization and no fuzzy matching. This is the frozen rule used in the paper.
- A zero denominator gives `NaN`, never 0.
- An ASK chosen when the hidden pool is empty is labeled `UNFULFILLABLE_ASK`. It is never converted to another action.
- `DEFER` is treated as a non-decision outcome, not as a correct one. No ground truth for action appropriateness is available.
- Confidence intervals are percentile intervals (NumPy `linear` quantiles) from a case-clustered bootstrap: 10,000 replicates, resampling cases with replacement and keeping all of each case's states together. `ClusteredBootstrap` has no default seed. Every analysis must give a `seed` and a `case_order`:
  - `sorted`: cases in lexicographic ID order.
  - `first_appearance`: cases in the order they first appear in the input rows.

  The same seed with a different case order gives different intervals.
- Common action patterns (`common_patterns`) are counted over **disagreement states only** by default, which is the scope of the paper's table. Its three reported patterns (45, 21, 19) are the top three among the 137 disagreement states, not among all 200 states. The output is labeled with its scope and denominator. Pass `disagreement_only=False` for the all-state ranking.
- McNemar and Stuart–Maxwell p-values treat matched states as independent and are **not** adjusted for clustering within case. Holm correction is applied per family: the three Stuart–Maxwell model pairs form one family and the nine action-specific McNemar tests another.

## Reproduction status

Checked by running this code on the frozen state-level outputs behind the paper (not distributed here):

| Analysis | Script | Bootstrap settings | Status against the paper |
|---|---|---|---|
| Cross-model agreement counts, Stuart–Maxwell χ² and Holm p, action-specific McNemar and Holm p, disagreement-only pattern counts | `cross_model_analysis.py` | n/a | **Exact** |
| Pairwise disagreement CIs | `cross_model_analysis.py` | seed 20260831, `sorted` | **Exact** |
| Stage 1 and final targeted-policy metrics (all three models), Qwen binary-baseline metrics, three-action minus binary differences (estimates, CIs, McNemar p) | `compute_metrics.py`, `binary_baseline_analysis.py` | seed 20260830, `first_appearance` (rows in sampling order) | **Exact** for all 335 frozen intervals these metrics cover. The CLI prints CIs for the overall scope; per-evidence-level CIs come from the same `ClusteredBootstrap` with a level-filtered statistic |
| Matched transitions (wrong decision rescue 14/32, lost correct autonomy 43/105, abstention recovery 15/63) | `binary_baseline_analysis.py` | n/a | **Exact.** Abstention recovery uses the frozen definition (binary ABSTAIN → three-action *correct* DECIDE) |
| Qwen targeted vs. random ASK (rates, differences, McNemar p, sign test, CIs) | `ask_targeted_vs_random.py` | **unverified**, so `--seed` and `--case-order` are required | **Not verified.** The pair-level outputs and the original script were not available for this check. Point estimates and tests follow the paper's definitions; CIs may differ from the paper |

Not implemented here: three auxiliary Step 5 rates that the paper does not report as primary metrics (incomplete-evidence DECIDE rate, unfulfillable ASK rate among 100% states, unfulfillable ASK rate among all ASK).

## Examples

```bash
# Stage 1 or final-policy metrics for one model
python analysis/compute_metrics.py --states <qwen_states.csv> --stage final --config configs/example_config.yaml --by-evidence

# Cross-model comparison and Stage 1 action-rate figure
python analysis/cross_model_analysis.py --states qwen=<q.csv> gemini=<g.csv> gpt=<o.csv> \
    --config configs/example_config.yaml --plot figures/stage1_action_rates.pdf

# Qwen targeted vs. random ASK (add --distinct-only for the sensitivity analysis).
# Bootstrap provenance is unverified, so --seed and --case-order are required.
python analysis/ask_targeted_vs_random.py --pairs <qwen_pairs.csv> --config configs/example_config.yaml \
    --seed <SEED> --case-order sorted

# Qwen matched binary baseline
python analysis/binary_baseline_analysis.py --three-action <qwen_states.csv> \
    --binary <qwen_binary_states.csv> --config configs/example_config.yaml
```

Paths in `<…>` are local files you build from your own DDXPlus copy and model runs. They are not part of this repository.
