---
name: snomedct-query-skill
description: Use this skill when querying the SNOMEDCT clinical vocabulary graph, including semantic aliases and intent-guided Cypher patterns.
---

# SNOMEDCT Query Skill

## When To Use
Use this skill when a request targets `SNOMEDCT` concepts, descriptions, relationships, or intent-style retrieval patterns.

## Vocabulary Identifier
Use `SNOMEDCT` as the vocabulary string in all MCP tool calls.

## MCP Tools
Two tools are available via the `kdb-mcp` server:
- **`get_skills("SNOMEDCT")`** — Fetches the live semantic layer for SNOMEDCT. Call this first to get runtime-accurate node aliases, relationship aliases, and vocabulary-specific guidance.
- **`cypher_safe("SNOMEDCT", query, limit)`** — Executes a read-only Cypher query against the SNOMEDCT graph. Queries must start with `MATCH`, `OPTIONAL MATCH`, or `WITH`. All write operations are blocked by the server.

## Workflow
1. Read `references/data-model-and-query-practices.md` first.
2. Use semantic aliases and intent patterns before writing raw unconstrained Cypher.
3. Keep queries read-only and bounded with explicit `LIMIT`.
4. Return only graph-backed facts and include the evidence fields from query output.

## Semantic Layer Snapshot
- Alias groups detected: `clinical`, `has_description`, `observation`, `procedure`, `product`, `temporal`
- Node alias labels detected: `Concept`

## Intents catalog

Intents represent concrete use cases within this knowledge domain and can be used as query recipes.

- `concept.description`: Return concept descriptions (FSN, preferred term, synonyms) for a given concept.
- `disease.anatomic_site_laterality`: Return finding site and laterality qualifiers for a disease concept.
- `mapping.icd10_to_snomed_site_laterality`: Given an ICD-10 code, return mapped SNOMED concept(s) with finding site and laterality when available.
- `procedure.context`: Return key context for a procedure, including method and procedure site.

## Reference Files
- `references/ontology-skills.md` — Full semantic layer definition: node aliases, relationship aliases, Cypher cheatsheet, and usage examples. Consult when building queries or resolving alias names.
- `references/intent-catalog.md` — Query recipes with inputs, trigger phrases, Cypher plans, guardrails, and output schemas. Use to match user intent to a structured query pattern.
- `references/data-model-and-query-practices.md` — Base graph model and query discipline reference. Consult when exploring the schema or writing ad-hoc queries outside defined intents.

