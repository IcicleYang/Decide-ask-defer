# Evaluation Protocol (`v2_blinded_action_selection`)

This document describes the protocol used in *Decide, Ask, or Defer: Clinical LLMs under Incomplete Evidence* (NeurIPS 2026 GenAI4Health Workshop). It is an exploratory behavioral evaluation on synthetic cases. It is not a clinical decision policy.

## 1. Evidence states

- **Cases.** 50 cases from the DDXPlus test split, chosen by pathology-aware sampling with seed 20260824. One case is drawn per pathology group, then the remaining cases are drawn from the rest of the pool. No filtering on pathology, age or evidence count. The set covers 49 pathologies: 48 with one case each, and anemia with two.
- **States.** For each case, the revealed evidence `E_t` is 25%, 50%, 75% or 100% of the case evidence `E`, giving 200 matched states. Every evaluated model and condition sees exactly the same revealed evidence for a given state.

## 2. Blinded Stage 1

The model gets the system instruction (a synthetic differential-diagnosis benchmark, not patient care), the revealed patient record `E_t`, the closed list of candidate pathologies, and the action definitions:

- **DECIDE**: the evidence currently available is sufficient for an autonomous diagnosis.
- **ASK**: the evidence currently available appears insufficient, and more patient-specific information could reasonably help.
- **DEFER**: the case should be transferred to a clinician rather than resolved autonomously.

It returns exactly one action. A DECIDE response includes a diagnosis from the candidate list, which is scored against the canonical DDXPlus label. A DEFER ends the autonomous path. Stage 1 does **not** show candidate hidden questions, so the availability of questions cannot serve as a shortcut for deciding whether evidence remains.

### Metadata hidden from the model

At every stage, including the binary baseline, the text sent to the model never contains:

- the case identifier
- the evidence percentage / evidence level
- the number of hidden evidence items
- the identities of hidden evidence items (at Stage 1)
- the completeness status of the case
- the ground-truth diagnosis

## 3. Case-specific hidden evidence pool

For each state, the hidden pool is `H_t = E \ E_t`, the evidence items of *this* case that have not yet been revealed. At 100% evidence, `H_t` is empty.

## 4. One-step ASK

If Stage 1 returns ASK and `H_t` is not empty:

1. **Stage 2.** The model sees the IDs, question text and data type of every item in `H_t`, but not the answers. It selects exactly one item `q_t ∈ H_t`. A selection outside `H_t` fails validation.
2. **Answer retrieval.** The answer to `q_t` comes directly from the underlying DDXPlus case record. No model generates it. The revealed evidence becomes `E_{t+1} = E_t ∪ {q_t}`.
3. **Post-ASK final action.** The model reassesses after this single answer and must choose **DECIDE or DEFER**. A second ASK is not allowed. This rule comes from the protocol and is not measured as model behavior.

ASK is therefore a simplified form of information seeking: one choice from a finite, case-specific pool, not open-ended questioning.

## 5. Unfulfillable ASK

If Stage 1 returns ASK when `H_t = ∅` (which happens only at 100% evidence), the state is recorded as **Unfulfillable ASK**. The observed action is kept and is never converted post hoc to DECIDE, DEFER or ABSTAIN. In final-policy metrics it is reported as an unresolved outcome.

## 6. Matched random comparator (Qwen only)

For each executable Qwen ASK, the *targeted* condition reveals the item the model selected. The *random* condition reveals one item drawn uniformly from the same `H_t`, takes its answer from the same case record, and applies the same post-ASK DECIDE/DEFER step. The two conditions differ only in how the item is chosen.

- One random draw is made per executable ASK, so variability across repeated draws is not averaged out.
- Pairs where the random draw happens to equal the targeted item are valid randomized outcomes and stay in the primary analysis. A sensitivity analysis excludes them.
- A pair-level ordering (correct DECIDE > DEFER > incorrect DECIDE) is used only to classify transitions. It is not a clinical utility function.

## 7. Matched binary baseline (Qwen only)

The baseline uses the same model, provider, cases, evidence states and revealed evidence as the three-action condition. Only the action space changes, to **DECIDE / ABSTAIN**. ABSTAIN ends the interaction and no information is acquired. Completeness metadata stays hidden. This isolates the effect of the action space while holding the model and the evidence fixed. The exact wording of the binary instruction is not included in this repository (see `prompts/README.md`).

## 8. Output validation

- Stage 1 allows only DECIDE, ASK or DEFER. A DECIDE must name a diagnosis from the candidate list.
- Stage 2 runs only after an executable ASK. The selected ID must belong to the current `H_t`.
- Post-ASK allows only DECIDE or DEFER.
- The binary baseline allows only DECIDE or ABSTAIN.

## 9. Metrics and statistics

See `analysis/README.md`. In brief: conditional decision accuracy, autonomous coverage, correct autonomous coverage, wrong decision rate and unsafe premature decision rate (wrong DECIDE at initial evidence < 100%, divided by all states with initial evidence < 100%). Confidence intervals come from a case-clustered percentile bootstrap with 10,000 replicates. Each analysis has its own seed and case order, documented in `configs/README.md`. Paired binary outcomes use exact McNemar tests. Cross-model action distributions use pairwise Stuart–Maxwell tests. Holm correction is applied within each family of tests.

## 10. Interpretation constraints

- **No ground truth for action appropriateness.** DDXPlus has diagnosis labels but no labels for whether DECIDE, ASK or DEFER was the appropriate action. DEFER is treated as a non-decision outcome, **not** as a clinically correct action.
- **Model-specific deeper analyses.** The targeted-vs-random and matched binary analyses were run only for Qwen and should not be generalized to the other models.
- **No uniform improvement.** The three-action formulation exposes behavior that binary abstention hides. It is not shown to be uniformly safer or more accurate. For Qwen it shows a trade-off between safety and autonomy.
- **Non-standardized generation controls.** GPT was called with `reasoning_effort="none"`. No matching setting was fixed for Qwen or Gemini. Cross-model differences may partly reflect invocation settings.
- **Synthetic cases.** 50 DDXPlus cases do not capture the variability of real clinical workflows.
