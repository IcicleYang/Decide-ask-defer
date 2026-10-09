# Prompts

These are the prompt templates for the workshop protocol (`v2_blinded_action_selection`). All model calls are framed as a **synthetic differential-diagnosis benchmark, not patient care**.

| File | Used when | Status |
|---|---|---|
| `system_prompt.txt` | Every call (system message) | Frozen text |
| `stage1_action_selection.txt` | Stage 1: initial DECIDE / ASK / DEFER choice | Frozen text |
| `stage2_ask_selection.txt` | Stage 2: only after an *executable* Stage 1 ASK | Frozen text |
| `post_ask_final_action.txt` | After the selected item's answer is revealed | Frozen text |

"Frozen text" means the wording matches the formal-run requests exactly. The four files were checked against every frozen GPT request (319 requests: 200 Stage 1, 41 Stage 2, 78 post-ASK) and against the Gemini request cache. Template fields in braces (`{age}`, `{observed_history}`, …) are filled per evidence state.

**Binary baseline prompt: not included.** The exact frozen wording of the Qwen DECIDE/ABSTAIN instruction could not be recovered for this release. Rather than publish a reconstruction, the file is omitted. The baseline's design is described below and in `docs/protocol.md`, but its exact instruction text is not part of this repository.

## Template fields

- `{observed_history}`: one line per revealed evidence item, written as
  `- {question} [{evidence_id}]: {display_value}`
- `{candidate_pathologies}`: one line per pathology in the closed candidate list, written as `- {pathology}`
- `{hidden_evidence_list}`: one line per item in the case-specific hidden evidence pool, written as
  `- {hidden_evidence_id}: {question} (type={data_type})`. Answers are **not** shown.

## How messages are assembled

1. **Stage 1.** System message `system_prompt.txt`. User message `stage1_action_selection.txt`, filled with the currently revealed evidence only. Hidden questions are not shown at this stage.
2. **Stage 2.** Runs only if Stage 1 returned ASK **and** the hidden evidence pool is non-empty. User message `stage2_ask_selection.txt`. The model must pick one ID from the listed pool.
3. **Post-ASK.** The answer to the selected item comes straight from the case record. No model generates it. The user message is the Stage 1 message followed by `post_ask_final_action.txt`. Only DECIDE or DEFER is allowed, and a second ASK is excluded by protocol.
4. **Unfulfillable ASK.** If Stage 1 returns ASK when the hidden pool is empty, Stages 2 and 3 do not run. The state is recorded as `UNFULFILLABLE_ASK`.
5. **Binary baseline (prompt text not included).** Same system message, patient record and candidate list as Stage 1, with a DECIDE / ABSTAIN action space and completeness metadata still hidden. ABSTAIN ends the interaction. No information is acquired.

## Structured output

Formal runs used structured JSON output. The three-action response object has these fields, all required. This was checked against the GPT request schema and the parsed Gemini responses:

| Field | Type | Notes |
|---|---|---|
| `top_diagnosis` | string | Must come from the candidate list when the action is DECIDE |
| `confidence` | number in [0, 1] | Recorded, not used by the workshop metrics |
| `information_sufficient` | boolean | Recorded, not used by the workshop metrics |
| `recommended_action` | `"DECIDE"` \| `"ASK"` \| `"DEFER"` | Stage 1 action. Stage 2 returns `"ASK"`. Post-ASK allows only `"DECIDE"`/`"DEFER"` |
| `ask_evidence_id` | string \| null | Null at Stage 1. At Stage 2, an ID from the hidden pool |
| `ask_question` | string \| null | Null at Stage 1. At Stage 2, the selected question text |
| `ask_may_change_diagnosis` | boolean | Recorded, not used by the workshop metrics |
| `boundary_reason` | `"high_uncertainty"` \| `"conflicting_evidence"` \| `"high_risk"` \| null | Recorded, not used by the workshop metrics |
| `brief_rationale` | string | Under 40 words |

See `examples/` for synthetic instances of this format.
