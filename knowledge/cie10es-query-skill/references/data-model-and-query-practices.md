# Data Model And Query Practices: CIE10ES

## Data Model Summary
- Source semantic layer: `CIE10ES.yaml`
- Primary node labels are typically `Concept` and `Description`.
- Core relationship labels commonly include `IS_A`, `RELATIONSHIP`, and `HAS_DESCRIPTION`.
- Semantic aliases in the YAML map user-facing names to relationship ids and node anchors.

## Data Model By Example (Cypher)
Inspect sample concept nodes:
```cypher
MATCH (c:Concept)
RETURN c
LIMIT 5
```

Inspect sample description nodes:
```cypher
MATCH (d:Description)
RETURN d
LIMIT 5
```

Concept to description linkage (`HAS_DESCRIPTION`):
```cypher
MATCH (c:Concept)-[:HAS_DESCRIPTION]->(d:Description)
RETURN c, d
LIMIT 10
```

Hierarchy traversal (`IS_A`):
```cypher
MATCH (child:Concept)-[:IS_A]->(parent:Concept)
RETURN child, parent
LIMIT 10
```

Typed semantic relationships (`RELATIONSHIP`):
```cypher
MATCH (a:Concept)-[r:RELATIONSHIP]->(b:Concept)
RETURN a, r, b
LIMIT 10
```

## Recommended Query Practices
- Start with focused `MATCH` patterns and explicit predicates on identifiers or aliases.
- Always enforce a bounded result size with `LIMIT`.
- Prefer semantic aliases from the configured layer over raw relationship ids when possible.
- Filter by `active` flags and language/term qualifiers when relevant.
- Use deterministic projections (`RETURN a,b,c`) and avoid broad `RETURN *` in production flows.
- Keep reads read-only; do not use `CREATE`, `MERGE`, `DELETE`, `SET`, `DROP`, `CALL`, or `APOC`.
