# Graph Schema

| Field | Value |
| --- | --- |
| Schema version | `1.0.0` |
| Status | Phase 1 contract, frozen for implementation |
| Owner | Member 3 (Graph Construction) |
| Consumers | Graph construction, feature extraction, GNN models, runtime policy engine, evaluation |
| Upstream contract | Canonical Event schema (Member 2, `src/ingestion/event_schema.py`) |
| Reference fixture | `tests/graph/test_fixtures.py` (`TINY_SESSION_FIXTURE`) |

This document is the normative data contract for the dynamic graph. It defines
what a node is, what an edge is, and which graph structures are legal. It does
not describe graph algorithms, tensor layouts, model features, or runtime
decisions. Those belong to other modules.

The key words **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are used as
described in RFC 2119.

---

## 1. Scope and module boundary

This schema governs the output of the graph construction module only:

- It consumes a sequence of Canonical Events.
- It produces a session-scoped graph of event nodes and typed edges.
- It preserves the Canonical Event fields without semantic reinterpretation.
- It does not classify, score, embed, or label events.
- It does not decide whether an action is allowed or blocked.

Ingestion validation is owned by Member 2. Node features are owned by Member 4.
Model and runtime behaviour are out of scope. Graph construction MUST NOT
silently repair malformed events; it MUST reject them under the validation
rules in section 11.

---

## 2. Non-negotiable invariants

These invariants apply to every graph in Phase 1.

| ID | Invariant |
| --- | --- |
| `INV-1` | **Event-level granularity only.** Every graph node is exactly one Canonical Event. |
| `INV-2` | **One event, one node.** The node identifier is the event's original `event_id`, unchanged. |
| `INV-3` | Only the six node types in section 6 are allowed. No entity, document, turn, chunk, or content-derived node types are allowed. |
| `INV-4` | Only the four edge types in section 7 are allowed. `INFLUENCES` and `CONFLICTS_WITH` are restricted and MUST NOT be created in Phase 1. |
| `INV-5` | An edge MUST have an explicit field-level basis. Temporal adjacency, shared session, shared source, shared document, or textual similarity is not sufficient evidence. |
| `INV-6` | No edge may cross session boundaries. |
| `INV-7` | Graph updates are deterministic and idempotent. Replaying the same events produces the same nodes and edges. |
| `INV-8` | Canonical Event fields are preserved. `event_id`, `timestamp`, `source_type`, `source_id`, and `trust_level` MUST NOT be rewritten, normalized away, or replaced by graph-level proxies. |
| `INV-9` | Graph construction MUST NOT compute model features, embeddings, labels, or risk scores. |
| `INV-10` | A restricted edge type MUST cause a non-retryable schema error, not a silent drop. |

---

## 3. Node granularity: event-level

### 3.1 Decision

The graph uses **event-level granularity**.

Each event becomes **one unique node**. The node preserves the event's original
`event_id`, `timestamp`, `source`, and `trust`. Multiple events from the same
user, assistant, tool, document, or policy source remain separate nodes. Events
are never merged, collapsed into turns, or replaced by entity nodes.

This decision is deliberate:

- It preserves the full provenance required for jailbreak and prompt-injection
  analysis.
- It keeps temporal ordering lossless.
- It prevents an entity node from hiding which particular event introduced
  untrusted content.
- It allows the model to reason over sequences of events rather than over
  aggregated identities.

**Do not mix granularities.** A graph MUST NOT contain event nodes and
entity-level nodes in the same structure. A future document-graph or
entity-graph design would be a separate graph and a schema version change, not
an extension of this contract.

### 3.2 Prohibited node patterns

The following are explicitly invalid in Phase 1:

| Prohibited pattern | Why it is invalid |
| --- | --- |
| One `user` node per person, with many events attached | Collapses event-level provenance into entity-level aggregation. |
| One `assistant` node per model instance | Hides which response contained which content. |
| A `document` node for each retrieved document | Introduces a seventh node type and a second granularity. Document identity belongs in `retrieved_context.metadata`. |
| A `turn` or `conversation` node | Coarsens the graph and obscures individual event order. |
| A `chunk` node | A retrieved chunk is represented by the event that retrieved it, not as a separate node. |
| A node for an extracted person, URL, email address, or claim | Content-derived nodes are outside this schema and are not provenance-preserving. |
| Splitting one event into multiple nodes | Violates one-event-one-node identity and duplicates provenance. |
| Merging duplicate events with the same source | Hides repeated or adversarial events. Duplicate `event_id` values are an error. |

### 3.3 Node identity

- The node identifier is exactly `event_id`.
- The identifier is opaque, case-sensitive, and MUST NOT be hashed, truncated,
  normalized, case-folded, or prefixed.
- `event_id` MUST be globally unique within the runtime. Because the node ID is
  the bare `event_id`, two sessions MUST NOT reuse the same `event_id`.
- A duplicate `event_id` is a hard validation error. The builder MUST NOT
  overwrite the existing node.

---

## 4. Canonical Event to node mapping

### 4.1 Canonical Event fields

The input record is the Canonical Event defined by Member 2. Every event MUST
contain all ten keys. Optionality is represented by `None`, never by omitting a
key.

| Canonical field | Type | Graph treatment |
| --- | --- | --- |
| `event_id` | `str` | Node identifier. Preserved exactly. |
| `session_id` | `str` | Session partition key and node attribute. Preserved exactly. |
| `timestamp` | `str` | Timezone-aware UTC ISO 8601 timestamp. Preserved exactly as supplied. |
| `event_type` | `str` | Used only to assign `node_type` through the closed mapping in section 4.2. |
| `source_type` | `str` | Preserved as an informational source attribute. Not used to infer `node_type`. |
| `source_id` | `str \| None` | Preserved exactly. Identifies the emitting principal or resource. |
| `content` | `str` | Preserved verbatim. Graph construction MUST NOT truncate, rewrite, or summarise it. |
| `parent_event_id` | `str \| None` | Preserved and used only for the explicit `RESPONDS_TO` rule. |
| `trust_level` | `str` | Preserved exactly. Phase 1 allowed values are `trusted` and `untrusted`. |
| `metadata` | `dict` | Preserved as a JSON-compatible mapping. The builder MUST NOT mutate the caller's object. |

`timestamp` MUST be a timezone-aware UTC datetime string, for example
`2026-03-02T10:00:00+00:00`. Naive timestamps and local-offset timestamps are
invalid. Graph construction preserves the original string; feature extraction
may parse it later.

`trust_level` describes the trust of the emitting source or channel for this
event. It is not a safety verdict on the content and it does not propagate
transitively. Provenance about untrusted inputs is carried by explicit edges
and metadata, not by rewriting the consumer's trust label.

### 4.2 Node record

A node is the original Canonical Event plus one derived attribute, `node_type`.
No other fields may be added or removed.

```json
{
  "event_id": "evt-0003",
  "session_id": "sess-tiny-0001",
  "timestamp": "2026-03-02T10:00:06+00:00",
  "event_type": "retrieved_context",
  "source_type": "document_store",
  "source_id": "kb-travel-policy-2026#section-4.2",
  "content": "Travel Policy 4.2 - Reimbursement: ...",
  "parent_event_id": "evt-0002",
  "trust_level": "untrusted",
  "metadata": {
    "document_id": "kb-travel-policy-2026",
    "chunk_id": "section-4.2"
  },
  "node_type": "retrieved_context"
}
```

`node_type` is derived exclusively from `event_type`. It MUST NOT be inferred
from `source_type`, `source_id`, `content`, `metadata`, or `trust_level`.

### 4.3 Event type to node type mapping

| `event_type` | `node_type` |
| --- | --- |
| `user_message` | `user` |
| `assistant_message` | `assistant` |
| `retrieved_context` | `retrieved_context` |
| `tool_call` | `tool` |
| `tool_result` | `tool` |
| `proposed_action` | `proposed_action` |
| `policy_decision` | `policy` |

The mapping is closed for schema version `1.0.0`. An unknown `event_type` MUST
be rejected with `GRAPH_UNKNOWN_NODE_TYPE`. There is no default or catch-all
node type. Adding a new event type requires updating this table, the fixture,
and the schema version.

---

## 5. Allowed node types

Only the following six node types are legal.

| Node type | Meaning | Typical event types | Must not be used for |
| --- | --- | --- | --- |
| `user` | A single human-authored input event in the session. | `user_message` | A persistent user identity or profile. |
| `assistant` | A single model-generated output event. | `assistant_message` | A persistent model or agent identity. |
| `retrieved_context` | A single event representing material retrieved from an external document or knowledge source. | `retrieved_context` | The document itself, a chunk collection, or a vector-store record. |
| `tool` | A single observed tool invocation or tool result event. | `tool_call`, `tool_result` | A proposed but unexecuted action. |
| `proposed_action` | A single action proposed by the assistant or runtime and awaiting policy evaluation. | `proposed_action` | An executed tool call or its result. |
| `policy` | A single policy evaluation or policy decision event. | `policy_decision` | A static policy document or configuration object. |

A `policy` node is an event node, not a configuration object. If policy text is
retrieved as context, it is represented by a `retrieved_context` node instead.

---

## 6. Edge record contract

Every edge record contains:

| Field | Type | Requirement |
| --- | --- | --- |
| `source_event_id` | `str` | Existing node ID in the same session. |
| `target_event_id` | `str` | Existing node ID in the same session. |
| `edge_type` | `str` | One of the four allowed edge types in section 7. |
| `session_id` | `str` | Must equal both endpoint session IDs. |
| `provenance` | `dict` | Structured evidence for why the edge exists. |

The canonical edge identity is the ordered triple:

```text
(source_event_id, edge_type, target_event_id)
```

Parallel edges are allowed only when their `edge_type` differs. Re-emitting the
same triple is an idempotent no-op. A self-loop is invalid for every edge type.
An edge with a missing endpoint is a dangling-edge error and MUST NOT be
persisted. Edges MUST NOT cross sessions.

`provenance` MUST identify the exact input field that justified the edge. It is
not free-form justification. Examples:

| Edge type | Required provenance evidence |
| --- | --- |
| `NEXT_EVENT` | `{"basis": "temporal_succession", "order_key": ["timestamp", "event_id"]}` |
| `RESPONDS_TO` | `{"basis": "parent_event_id", "parent_event_id": "evt-0002"}` |
| `RETRIEVED_FROM` | `{"basis": "metadata.retrieved_context_event_ids", "event_id": "evt-0003"}` |
| `REQUESTS_TOOL` | `{"basis": "metadata.requested_by_event_id", "requesting_event_id": "evt-0004"}` |

---

## 7. Allowed edge types

Only these four edge types are allowed.

| Edge type | Direction | Allowed source node type | Allowed target node type | Explicit creation rule |
| --- | --- | --- | --- | --- |
| `NEXT_EVENT` | Earlier event -> later event | Any of the six node types | Any of the six node types | Target is the immediately following event in the same session under the deterministic ordering rule. |
| `RESPONDS_TO` | Child event -> causal parent event | Any of the six node types | Any of the six node types | Source `parent_event_id` equals target `event_id`. No other basis is valid. |
| `RETRIEVED_FROM` | Consuming event -> retrieved-context event | `user`, `assistant`, `proposed_action`, or `policy` | `retrieved_context` | Source `metadata.retrieved_context_event_ids` explicitly contains target `event_id`. |
| `REQUESTS_TOOL` | Assistant event -> requested action/tool event | `assistant` | `proposed_action` or `tool` | Target `metadata.requested_by_event_id` equals source `event_id`. |

### 7.1 `NEXT_EVENT` (temporal succession)

`NEXT_EVENT` encodes temporal succession between events in the same session.

Rules:

- Direction is older event -> newer event.
- The ordering key is `(timestamp, event_id)` ascending. The event ID is the
  deterministic tie-breaker when timestamps are equal.
- In a fully materialised session with `n` event nodes, there are exactly `n - 1`
  `NEXT_EVENT` edges and they form a single ordered path.
- When an event is inserted out of order, the builder MUST re-link the affected
  predecessor and successor and MUST NOT leave a duplicate or stale edge.
- `NEXT_EVENT` means only "this event comes next". It does not imply influence,
  causality, agreement, conflict, or a response relationship.
- Temporal adjacency MUST NOT be used as evidence for any restricted edge.

### 7.2 `RESPONDS_TO` (explicit response relationship)

`RESPONDS_TO` encodes the causal relationship asserted by
`parent_event_id`.

Rules:

- Source is the child event; target is the event named by the child's
  `parent_event_id`.
- A non-null `parent_event_id` produces at most one `RESPONDS_TO` edge.
- A null `parent_event_id` produces no `RESPONDS_TO` edge.
- Parent and child MUST be in the same session.
- The parent MUST already exist, MUST NOT be the source itself, and MUST NOT
  create a cycle. The parent timestamp MUST NOT be later than the child
  timestamp.
- If the parent is not present, the child remains unlinked. The builder MUST
  NOT guess a different parent, and MUST NOT use the previous event as a
  substitute. The event may be deferred or rejected by the caller's ingestion
  policy, but no invented edge may be created.
- Shared source, shared content, temporal adjacency, and embedding similarity
  are not valid evidence for `RESPONDS_TO`.

`RESPONDS_TO` is not limited to assistant messages. A retrieved-context event
is generated in response to the query event named by its `parent_event_id`, so
that pair also produces a `RESPONDS_TO` edge. This is why the fixture's
retrieved-context event has a parent even though it is not an assistant
message.

### 7.3 `RETRIEVED_FROM` (context provenance)

`RETRIEVED_FROM` records that an event consumed material represented by a
`retrieved_context` event.

Phase 1 does not create document nodes. A document, chunk, or knowledge-base
record is not a node type. Instead:

- The `retrieved_context` event is the event-level materialisation of the
  retrieved material.
- Document identity is stored on that node's `metadata`, using fields such as
  `document_id`, `chunk_id`, and `source_uri`.
- `RETRIEVED_FROM` connects the event that used the material to that
  `retrieved_context` node.

Rules:

- Target MUST be a `retrieved_context` node.
- Source MUST be a non-`retrieved_context` node whose `metadata` contains a
  list named `retrieved_context_event_ids`.
- An edge is created only for each target `event_id` explicitly present in that
  list.
- A missing, empty, or malformed list produces no edge.
- Semantic similarity, citation-looking text, matching document names, and
  shared keywords MUST NOT be used to infer this edge.
- Creating a `document` node or an edge directly to an external document is a
  schema violation in Phase 1.

### 7.4 `REQUESTS_TOOL` (assistant tool request)

`REQUESTS_TOOL` records that an assistant event requested a tool action.

Rules:

- Source MUST be an `assistant` node.
- Target MUST be a `proposed_action` node for an unexecuted request, or a `tool`
  node for an observed tool invocation or result.
- The target's `metadata.requested_by_event_id` MUST equal the source
  `event_id`.
- One assistant event may request multiple actions; each target needs its own
  explicit `requested_by_event_id`.
- Do not infer a request from words such as "send", "search", "run", or
  "delete", from imperative content, or from adjacency.
- `REQUESTS_TOOL` does not assert that the action was authorised or executed.
  Execution and policy outcomes are separate events.

---

## 8. Restricted edge types

The following edge types are **restricted for Phase 1** and MUST NOT be
created:

| Edge type | Status | Reason |
| --- | --- | --- |
| `INFLUENCES` | Disabled | Requires semantic and causal evidence that is not encoded by event order or adjacency. |
| `CONFLICTS_WITH` | Disabled | Requires a claim-level comparison that Phase 1 does not model. |

These edge types are not part of the Phase 1 edge whitelist. A graph update that
attempts to create either one MUST fail with `GRAPH_RESTRICTED_EDGE_TYPE`.

### 8.1 Adjacency is not evidence

`INFLUENCES` and `CONFLICTS_WITH` MUST NOT be created purely based on
adjacency. "Adjacency" includes, but is not limited to:

- a `NEXT_EVENT` relationship,
- a `RESPONDS_TO` parent-child relationship,
- a `RETRIEVED_FROM` relationship,
- a `REQUESTS_TOOL` relationship,
- the same `session_id`,
- the same `source_type` or `source_id`,
- proximity in time,
- co-mention of the same document, person, URL, or keyword,
- embedding similarity or any model score.

Invalid inference examples:

| Invalid inference | Why it is invalid |
| --- | --- |
| "Event B follows event A, therefore A influences B." | Temporal succession is not causation or influence. |
| "The assistant responded to retrieved context, therefore the context influenced the response." | `RESPONDS_TO` only asserts the declared parent link; influence requires separate evidence. |
| "Two events mention the same document, therefore they conflict." | Shared references do not establish contradictory claims. |
| "A user message and an assistant message are adjacent, therefore they conflict." | Adjacency is not a semantic comparison. |
| "The embedding distance is small, therefore the events are in conflict." | Similarity is not disagreement. |

### 8.2 Conditions for future enablement

Enabling either restricted edge requires all of the following:

1. A schema version change and review by the graph, ingestion, and runtime
   owners.
2. A new explicit evidence field that records the claim-level or causal basis
   for the edge.
3. A dedicated validator that rejects adjacency-only construction.
4. Contract tests covering false-positive adjacency cases.
5. Runtime policy review confirming how the edge affects decisions.
6. An audit trail in the edge's `provenance`.

Until then, the restricted edges remain absent from all graphs, including test
fixtures.

---

## 9. Graph-level contract

### 9.1 Session scoping

- One graph represents one `session_id`.
- Every node in a graph has that session's `session_id`.
- Every edge connects nodes in the same graph and the same session.
- Cross-session edges are invalid, including edges between a user event in one
  session and a retrieved-context event in another.
- Cross-session identity resolution is out of scope.

### 9.2 Construction order and integrity

1. Validate every Canonical Event.
2. Create or upsert all event nodes with `node_type`.
3. Create `NEXT_EVENT` edges from the deterministic ordering.
4. Create `RESPONDS_TO` edges from explicit parent links.
5. Create `RETRIEVED_FROM` edges from explicit metadata references.
6. Create `REQUESTS_TOOL` edges from explicit request metadata.
7. Validate node and edge referential integrity.
8. Reject the update if any invariant is violated.

Node creation MUST precede edge creation. Edges MUST NOT refer to nodes that do
not exist in the same session.

### 9.3 Idempotent dynamic updates

- Upserting an existing `event_id` with identical content is a no-op.
- Upserting an existing `event_id` with different content is a conflict and
  MUST be rejected. Events are immutable.
- Adding an event may add one node, one or more semantic edges, and may modify
  the local `NEXT_EVENT` links.
- Replaying an identical event sequence MUST produce an identical graph.
- Removing or retracting an event is out of scope for Phase 1. If implemented
  later, it must also remove all incident edges and must be covered by a schema
  version change.

### 9.4 Ordering and serialisation

- Graph serialisation MUST use JSON-compatible values.
- Timestamps remain timezone-aware UTC strings in the serialised node record.
- Node and edge ordering in an export SHOULD be deterministic; when ordering is
  required, sort nodes by `(timestamp, event_id)` and edges by
  `(source_event_id, edge_type, target_event_id)`.
- `metadata` is preserved as supplied, subject to JSON compatibility. The graph
  builder MUST NOT insert derived model features into `metadata`.

---

## 10. Tiny fixture contract

The Phase 1 reference fixture is `TINY_SESSION_FIXTURE` in
`tests/graph/test_fixtures.py`. It contains exactly five Canonical Events in
chronological order for one session.

| Event ID | Event type | Node type | Source | Trust | Parent |
| --- | --- | --- | --- | --- | --- |
| `evt-0001` | `user_message` | `user` | `user-001` | `trusted` | `None` |
| `evt-0002` | `user_message` | `user` | `user-001` | `trusted` | `None` |
| `evt-0003` | `retrieved_context` | `retrieved_context` | `kb-travel-policy-2026#section-4.2` | `untrusted` | `evt-0002` |
| `evt-0004` | `assistant_message` | `assistant` | `assistant-primary` | `trusted` | `evt-0003` |
| `evt-0005` | `proposed_action` | `proposed_action` | `assistant-primary` | `untrusted` | `evt-0004` |

`evt-0002` is a follow-up user turn but has no `parent_event_id`. It is linked
to `evt-0001` only by temporal succession. This is intentional: a follow-up is
not automatically a `RESPONDS_TO` relationship without an explicit parent
assertion.

`evt-0004` consumes the retrieved context because its
`metadata.retrieved_context_event_ids` contains `evt-0003`. `evt-0005` is the
tool action requested by `evt-0004` because its
`metadata.requested_by_event_id` is `evt-0004`.

Expected graph for the fixture:

| Edge type | Source | Target | Basis |
| --- | --- | --- | --- |
| `NEXT_EVENT` | `evt-0001` | `evt-0002` | `(timestamp, event_id)` ordering |
| `NEXT_EVENT` | `evt-0002` | `evt-0003` | `(timestamp, event_id)` ordering |
| `NEXT_EVENT` | `evt-0003` | `evt-0004` | `(timestamp, event_id)` ordering |
| `NEXT_EVENT` | `evt-0004` | `evt-0005` | `(timestamp, event_id)` ordering |
| `RESPONDS_TO` | `evt-0003` | `evt-0002` | `parent_event_id` |
| `RESPONDS_TO` | `evt-0004` | `evt-0003` | `parent_event_id` |
| `RESPONDS_TO` | `evt-0005` | `evt-0004` | `parent_event_id` |
| `RETRIEVED_FROM` | `evt-0004` | `evt-0003` | `retrieved_context_event_ids` |
| `REQUESTS_TOOL` | `evt-0004` | `evt-0005` | `requested_by_event_id` |

Expected counts:

- 5 nodes.
- 9 edges: 4 `NEXT_EVENT`, 3 `RESPONDS_TO`, 1 `RETRIEVED_FROM`, 1
  `REQUESTS_TOOL`.
- 0 restricted edges.

The two edges from `evt-0004` to `evt-0003` are parallel edges with different
types: one asserts the declared parent relationship, the other asserts context
consumption. They are not duplicates.

The fixture deliberately contains an instruction-like sentence inside the
untrusted retrieved context. That text is data, not a policy instruction for the
graph builder. Graph construction MUST treat all `content` values as inert
strings.

---

## 11. Validation errors

The following error conditions are part of the contract. Phase 1 does not
require a specific exception class, but the condition MUST be detected and the
graph update MUST fail.

| Code | Condition |
| --- | --- |
| `GRAPH_UNKNOWN_NODE_TYPE` | `event_type` is absent from the closed mapping in section 4.3. |
| `GRAPH_NON_EVENT_NODE` | A node would represent a user entity, document, chunk, turn, or other non-event object. |
| `GRAPH_DUPLICATE_EVENT_ID` | The same `event_id` appears twice, or an existing event is re-upserted with different content. |
| `GRAPH_INVALID_SESSION_ID` | `session_id` is missing, empty, or inconsistent with the graph. |
| `GRAPH_CROSS_SESSION_EDGE` | An edge connects nodes with different `session_id` values. |
| `GRAPH_DANGLING_EDGE` | An edge endpoint does not exist in the graph. |
| `GRAPH_SELF_LOOP` | Source and target are the same event. |
| `GRAPH_UNKNOWN_PARENT` | A `RESPONDS_TO` edge would require a parent that is not present and cannot be resolved. |
| `GRAPH_PARENT_CYCLE` | Parent relationships form a cycle. |
| `GRAPH_UNGROUNDED_EDGE` | An edge has no explicit field-level basis from section 6. |
| `GRAPH_RESTRICTED_EDGE_TYPE` | A caller attempts `INFLUENCES` or `CONFLICTS_WITH`. |
| `GRAPH_INVALID_TIMESTAMP` | `timestamp` is not a parseable timezone-aware UTC datetime string. |

---

## 12. Change control

- This document and `tests/graph/test_fixtures.py` MUST change together.
- Adding a node type, edge type, or event-type mapping requires a schema version
  increment.
- Changing event-level granularity, node identity, or session scoping requires a
  major version change.
- Enabling a restricted edge type requires the full review described in section
  8.2.
- If Member 2 changes the Canonical Event schema, graph construction MUST NOT
  drift silently; the compatible version of this document and the fixture MUST
  be updated before implementation continues.
