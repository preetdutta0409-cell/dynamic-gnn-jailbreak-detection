# Recommended-target synthetic dataset statistics

This dataset contains deterministic synthetic interactions from 50 authored scenario families with 10 sessions each. Action labels are draft rubric annotations and have not received independent human review. SAFE/MALICIOUS labels remain UNREVIEWED; ambiguous examples from the initial pilot remain NEEDS_ADJUDICATION. Do not train or report model performance until teammates review the labels.

## Counts

| Measure | Count |
|---|---:|
| Total events | 5530 |
| Total sessions | 500 |
| Distinct scenarios | 50 |
| Proposed actions | 1000 |
| SAFE actions | 524 |
| MALICIOUS actions | 456 |
| AMBIGUOUS actions | 20 |
| Reviewed SAFE/MALICIOUS actions | 0 |
| Unreviewed SAFE/MALICIOUS actions | 980 |
| Actions awaiting adjudication | 20 |
| Sessions containing tool calls | 80 |
| Sessions containing retrieved context | 290 |

Binary draft balance: 524 SAFE / 456 MALICIOUS (53.5% / 46.5%). No binary action is yet eligible for training.

## Event-type distribution

| Event type | Count |
|---|---:|
| assistant_message | 1580 |
| policy_decision | 1000 |
| proposed_action | 1000 |
| retrieved_context | 290 |
| tool_call | 80 |
| tool_result | 80 |
| user_message | 1500 |

## Trust-level distribution

| Trust level | Count |
|---|---:|
| trusted | 2580 |
| untrusted | 2950 |

## Attack-category distribution

| Attack category | Count |
|---|---:|
| ambiguous_influence | 10 |
| ambiguous_scope | 10 |
| credential_exfiltration | 10 |
| cross_tenant_access | 10 |
| data_exfiltration | 30 |
| direct_jailbreak | 10 |
| external_content_injection | 50 |
| indirect_prompt_injection | 40 |
| instruction_conflict | 10 |
| instruction_hierarchy | 30 |
| memory_poisoning | 10 |
| metadata_injection | 10 |
| multi_stage_attack | 10 |
| multi_turn_jailbreak | 10 |
| none | 70 |
| policy_bypass | 10 |
| privilege_escalation | 20 |
| restriction_override | 10 |
| roleplay_jailbreak | 20 |
| sensitive_data_access | 10 |
| system_prompt_extraction | 10 |
| tool_misuse | 20 |
| tool_output_injection | 30 |
| unauthorized_disclosure | 10 |
| unauthorized_file_access | 10 |
| unauthorized_modification | 10 |
| unauthorized_tool_execution | 20 |

## Scenario-grouped split plan

| Split | Sessions | Distinct scenarios |
|---|---:|---:|
| train | 350 | 35 |
| validation | 80 | 8 |
| test | 70 | 7 |

Assignments are stored in scenario_splits.json. Sessions and events have not been manually divided into separate files; every scenario family remains in exactly one split.

## Generation and labeling method

The dataset is synthetic and deterministic. It uses 50 distinct scenario families with ten sessions per family, varied tasks and contexts, source channels, mock resources, and proposed actions. It includes benign tasks, direct and multi-turn jailbreaks, instruction-hierarchy manipulation, prompt extraction, role-play attacks, indirect document/web/tool-output injections, sensitive-data and credential exfiltration attempts, unauthorized tool use, privilege escalation, and ambiguous authorization cases. The attack families include counterfactual safe handling and compromised outcomes. All tools and actions are simulated; addresses use example.test; no real credentials, private data, or external side effects are used.

Labels follow docs/dataset_labeling_guidelines.md and remain in the separate manifest joined by proposed-action event_id. The original event schema's ten top-level fields are preserved; scenario_id is repeated in metadata.scenario_id and is authoritative in the manifest. A teammate must independently review binary labels and adjudicate ambiguous labels before training or evaluation.
