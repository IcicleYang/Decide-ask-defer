# Examples

Small, **fully fictional** files that show the input and output formats of the protocol. They contain no DDXPlus records, no real model responses and no experimental results. Evidence IDs (`EX_…`), questions and pathology names (`Example pathology A/B/C`) are invented placeholders.

| File | Shows |
|---|---|
| `example_state.json` | One evidence state, split into what the model sees and the evaluation metadata that stays hidden from it |
| `example_stage1_output.json` | A Stage 1 ASK response and the Stage 2 item selection that follows |
| `example_post_ask_output.json` | The post-ASK message, the final DECIDE/DEFER response, and the resulting row of the analysis state table |

The response objects use the structured-output fields documented in `prompts/README.md`. `derived_state_row` in `example_post_ask_output.json` uses the state-table schema expected by `analysis/`.
