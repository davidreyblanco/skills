---
name: rxnorm-query-skill
description: Use this skill when querying the RxNorm clinical vocabulary graph, including semantic aliases and intent-guided Cypher patterns.
---

# RxNorm Query Skill

## When To Use
Use this skill when a request targets `RxNorm` concepts, descriptions, relationships, or intent-style retrieval patterns.

## Vocabulary Identifier
Use `RxNorm` as the vocabulary string in all MCP tool calls.

## MCP Tools
Two tools are available via the `kdb-mcp` server:
- **`get_skills("RxNorm")`** — Fetches the live semantic layer for RxNorm. Call this first to get runtime-accurate node aliases, relationship aliases, and vocabulary-specific guidance.
- **`cypher_safe("RxNorm", query, limit)`** — Executes a read-only Cypher query against the RxNorm graph. Queries must start with `MATCH`, `OPTIONAL MATCH`, or `WITH`. All write operations are blocked by the server.

## Workflow
1. Read `references/data-model-and-query-practices.md` first.
2. Use semantic aliases and intent patterns before writing raw unconstrained Cypher.
3. Keep queries read-only and bounded with explicit `LIMIT`.
4. Return only graph-backed facts and include the evidence fields from query output.

## Semantic Layer Snapshot
- Alias groups detected: `has_description`, `relationship`, `rxnrel_rel`, `rxnrel_rela`
- Node alias labels detected: `Concept`, `Description`

## Intents catalog

Intents represent concrete use cases within this knowledge domain and can be used as query recipes.

- `rxnorm.code_lookup`: Resolve an RXCUI or medication term to RxNorm concepts and canonical names.
- `rxnorm.drug_composition`: Return ingredient, dose form, and tradename links for an RxNorm drug concept.

## Reference Files
- `references/ontology-skills.md` — Full semantic layer definition: node aliases, relationship aliases, Cypher cheatsheet, and usage examples. Consult when building queries or resolving alias names.
- `references/intent-catalog.md` — Query recipes with inputs, trigger phrases, Cypher plans, guardrails, and output schemas. Use to match user intent to a structured query pattern.
- `references/data-model-and-query-practices.md` — Base graph model and query discipline reference. Consult when exploring the schema or writing ad-hoc queries outside defined intents.

