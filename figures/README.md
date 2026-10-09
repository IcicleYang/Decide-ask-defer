# Figures

This folder is reserved for the paper figures. **No figure files are included yet.**

| Paper figure | Content | How to regenerate |
|---|---|---|
| Figure 1 | Binary abstention vs. the DECIDE / ASK / DEFER formulation (schematic) | Conceptual diagram, not generated from data |
| Figure 2 | Stage 1 action rates by initial evidence level for Qwen, Gemini and GPT, with unfulfillable ASK shown separately | `analysis/cross_model_analysis.py --plot` produces a comparable panel from local state tables. Styling differs from the paper version |
| Figure 3 | Qwen three-action final policy vs. matched DECIDE/ABSTAIN baseline: (a) metric differences with case-clustered 95% CIs, (b) matched transitions | Numbers come from `analysis/binary_baseline_analysis.py`. No plotting script is included |

In Figure 3, DEFER is treated as a non-decision outcome, and no ground-truth label for whether an action was appropriate is assumed.
