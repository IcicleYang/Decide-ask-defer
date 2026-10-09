# Decide, Ask, or Defer: Clinical LLMs under Incomplete Evidence

**Mingzhan Yang, Weili Wu**
NeurIPS 2026 GenAI4Health Workshop

This repository holds the reproducibility materials for the workshop paper: the prompt templates, protocol documentation, a public configuration, offline analysis code and synthetic format examples. Links to the paper are in [`paper/`](paper/README.md).

## Overview

Clinical LLMs have to decide what diagnosis to produce, and also whether the available evidence is enough to support an autonomous decision. Binary DECIDE/ABSTAIN formulations merge different non-decision states into one action, and they do not test information acquisition explicitly. We study a three-action formulation together with a blinded protocol that keeps evidence-completeness metadata away from the model. Qwen, Gemini and GPT are evaluated on 200 matched evidence states built from 50 synthetic DDXPlus cases.

This is an exploratory behavioral evaluation. It is **not** a validated clinical decision policy and makes no claim of clinical safety or readiness for deployment.

## Actions

| Action | Definition |
|---|---|
| **DECIDE** | The evidence currently available is sufficient for an autonomous diagnosis. The model returns a diagnosis from a closed candidate list. |
| **ASK** | The evidence currently available appears insufficient, and more patient-specific information could reasonably help. The model then picks **one** item from the case-specific hidden evidence pool. Its answer comes from the case record, and the model must then choose DECIDE or DEFER. |
| **DEFER** | The case should be transferred to a clinician rather than resolved autonomously. |

An ASK chosen when no hidden evidence remains is recorded as **Unfulfillable ASK** and is never re-labeled. The full protocol is in [`docs/protocol.md`](docs/protocol.md).

## Main findings

As reported in the paper:

1. **Action selection differs substantially across models under identical evidence.** At least two models disagree in 137 of the 200 states. Pairwise Stuart–Maxwell tests reject marginal homogeneity for every model pair after Holm correction. Disagreement remains high at full evidence (37 of 50 states). Among the 137 disagreement states, the most frequent pattern is Qwen ASK with Gemini and GPT both DECIDE (45 states), followed by Qwen DEFER with Gemini and GPT both DECIDE (21) and Qwen ASK, Gemini ASK, GPT DEFER (19). These are the top three **among disagreement states only**; over all 200 states, unanimous ASK (33) and unanimous DECIDE (30) would rank second and third.
2. **For Qwen, targeted information changes decision propensity more clearly than correctness.** Across 105 executable ASKs, the post-ASK DECIDE rate is 67.62% with targeted information and 57.14% with a matched random item. That is +10.48 pp: the case-clustered 95% CI excludes zero, but the exact McNemar p = .052, so the paper treats the effect as inconclusive. Correct DECIDE rises by 5.71 pp with a CI that includes zero (McNemar p = .263).
3. **For Qwen, the matched DECIDE/ABSTAIN baseline shows a trade-off between safety and autonomy.** The three-action final policy lowers correct autonomous coverage by 13.5 pp (McNemar p < .001), while the overall wrong decision rate changes by −1.0 pp (not significant). It rescues 14 of 32 binary wrong decisions, but 43 of 105 binary correct decisions lose correct autonomous status.

Separating information acquisition from clinician deferral exposes behavior that binary abstention hides. It does **not** yield a policy that is uniformly safer or more accurate.

**Scope of the evidence.** There is no ground truth for whether an action was appropriate. DDXPlus provides diagnosis labels but no labels for whether deferral was appropriate, so DEFER is treated as a non-decision outcome and never as a clinically correct action. The targeted-vs-random and matched binary analyses were run **only for Qwen**. Generation settings were not fully standardized across providers.

## Repository structure

```
decide-ask-defer/
├── README.md
├── LICENSE                 MIT
├── requirements.txt        Python dependencies of analysis/
├── paper/                  Links to the paper (arXiv / OpenReview / workshop)
├── prompts/                Frozen system, Stage 1, Stage 2 and post-ASK prompts, plus output schema
├── configs/                Public example configuration (no credentials)
├── analysis/               Offline metric and statistics code with documented input schemas
├── figures/                Description of paper figures and how to regenerate them
├── examples/               Small fictional JSON examples of the input/output format
└── docs/                   Protocol documentation
```

## Data

This repository **does not redistribute** DDXPlus records, the sampled case list, model outputs or API logs.

- DDXPlus (Tchango et al., 2022) is publicly available from its original authors. Obtain it from the official source and follow its license terms.
- The evaluation set is 50 cases from the DDXPlus test split, sampled pathology-aware with seed `20260824` (see `docs/protocol.md`).
- Files in `examples/` are fully fictional and only illustrate formats.

## Reproduction overview

1. **Environment.** `pip install -r requirements.txt` (Python ≥ 3.10).
2. **Evidence states.** From your local DDXPlus copy, sample 50 test cases as described in `docs/protocol.md` and build the 25/50/75/100% evidence states and hidden pools.
3. **Model runs.** Use the templates in `prompts/` and the settings in `configs/example_config.yaml`. The exact Qwen DECIDE/ABSTAIN baseline instruction is not included (see `prompts/README.md`), so the binary baseline cannot be re-run with identical wording. Provider credentials must come from your own environment and are never stored in the repository. Hosted model versions may change over time, so exact outputs may not be reproducible.
4. **State tables.** Convert the parsed outputs to the state-table and pair-table schemas documented in `analysis/README.md`.
5. **Analysis.** Run the scripts in `analysis/`. They make no model calls and work only on your local frozen outputs.

### What reproduces the paper exactly

The analysis code was run on the frozen state-level outputs behind the paper (not distributed here), using the settings in `configs/example_config.yaml`:

| Analysis | Status |
|---|---|
| Cross-model agreement counts, common patterns (disagreement states), Stuart–Maxwell tests, action-specific McNemar tests, Holm correction | **Exact** |
| Cross-model disagreement CIs (bootstrap seed 20260831, sorted case order) | **Exact** |
| Stage 1 and final-policy metrics, Qwen binary-baseline metrics, three-action minus binary differences and their CIs (bootstrap seed 20260830, first-appearance case order) | **Exact** |
| Matched transitions (14/32, 43/105, 15/63) | **Exact** |
| Qwen targeted vs. random ASK | **Not verified.** The original bootstrap settings and pair-level outputs were not available for this release, so its intervals may differ from the paper |

Exact interval reproduction needs the documented seed and case order for each analysis, and, for first-appearance order, input rows in evaluation-set sampling order. See `configs/README.md` and `analysis/README.md`.

## Scope

This repository covers **only** the accepted workshop paper. Materials for later unpublished extensions are intentionally excluded. Private run logs, intermediate artifacts and credentials are also excluded.

## Citation

The arXiv and final workshop bibliographic metadata are not available yet. The entry below contains only verified fields. Once the preprint is posted, add the arXiv identifier (`eprint`, `archivePrefix`) and URL.

```bibtex
@misc{yang2026decide,
  title        = {Decide, Ask, or Defer: Clinical {LLMs} under Incomplete Evidence},
  author       = {Yang, Mingzhan and Wu, Weili},
  year         = {2026},
  howpublished = {NeurIPS 2026 GenAI4Health Workshop}
}
```

## License

The code in this repository is released under the [MIT License](LICENSE). DDXPlus is not included and remains subject to its own license.
