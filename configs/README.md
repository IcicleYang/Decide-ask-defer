# Configs

`example_config.yaml` holds the public settings of the workshop protocol: the evaluation-case sampling seed, evidence levels, generation settings, model identifiers, and per-analysis bootstrap settings.

## No secrets

This file must never contain API keys, tokens, cloud project IDs, service-account paths or endpoints. `.gitignore` excludes `.env` and common credential files.

## Seeds

| Field | Value | Used for |
|---|---|---|
| `case_sampling_seed` | 20260824 | Selecting the 50 DDXPlus test cases (pathology-aware sampling). Not a generation seed and not a bootstrap seed |
| `bootstrap.step5_metrics.seed` | 20260830 | Case-clustered CIs for Stage 1 / final-policy metrics and the Qwen binary-baseline comparison |
| `bootstrap.cross_model.seed` | 20260831 | Case-clustered CIs for pairwise cross-model disagreement |
| `bootstrap.ask_targeted_vs_random.seed` | *unset* | Provenance of the paper's targeted-vs-random CIs is **unverified** (see below) |

No generation seed was set for any provider.

## Bootstrap case order

Every analysis draws a `10000 × 50` matrix of case indices with `numpy.random.default_rng(seed).integers(0, 50, …)`. Which case each index points to depends on the case order. The same seed with a different order gives different intervals.

- `sorted`: cases in lexicographic order of their IDs. Used for the cross-model analysis.
- `first_appearance`: cases in the order they first appear in the input table. Used for the Step 5 metrics. To reproduce the paper, input rows must follow the evaluation-set sampling order.

## Reproduction status of the paper's intervals

| Analysis | Settings | Status |
|---|---|---|
| Cross-model pairwise disagreement | seed 20260831, `sorted` | Reproduces the paper exactly |
| Stage 1, final-policy and binary-baseline metrics, three-action minus binary differences | seed 20260830, `first_appearance` | Reproduces the frozen Step 5 intervals exactly when rows are in sampling order |
| Qwen targeted vs. random ASK | unset | **Not verified.** The paper's intervals came from a separate script not included here. `ask_targeted_vs_random.py` requires `--seed` and `--case-order` explicitly, and its intervals may differ from the paper's |

## Generation settings

Generation controls were not fully standardized across providers. GPT was called with `reasoning_effort="none"` and `max_completion_tokens=2048`. No matching parameter was set for Qwen or Gemini, so cross-model differences may reflect invocation settings as well as model behavior.

The GPT and Gemini model IDs come from the frozen run configurations. The Qwen ID (`qwen/qwen3.8-27b`) comes from the frozen dataset-assembly manifest, which records the Qwen and Qwen-binary runs. Hosted model versions can change or be retired, so check the IDs against each provider's current catalogue before re-running.
