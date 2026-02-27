# Intent Catalog: RxNorm

{% if intent_sections %}
## Summary
- `rxnorm.code_lookup`: Resolve an RXCUI or medication term to RxNorm concepts and canonical names.
- `rxnorm.drug_composition`: Return ingredient, dose form, and tradename links for an RxNorm drug concept.

## Intent: rxnorm.code_lookup
**Goal:** Resolve an RXCUI or medication term to RxNorm concepts and canonical names.
**Triggers:**
- rxnorm code
- rxcui lookup
- find medication concept
- what drug is this code
**Inputs:**
- `text` (string; required=true; RXCUI fragment or medication name text.)
- `limit` (int; required=false; default=20; Maximum number of candidate concepts.)
**Required entities:**
- Concept
- Description
**Query plan:**
1. Match concepts by RXCUI or linked description strings.
```cypher
MATCH (c:Concept)
OPTIONAL MATCH (c)-[:HAS_DESCRIPTION]->(d:Description)
WHERE toLower(c.rxcui) CONTAINS toLower($text)
   OR (d.str IS NOT NULL AND toLower(d.str) CONTAINS toLower($text))
RETURN c.rxcui AS rxcui,
       c.tty AS tty,
       d.str AS term,
       d.tty AS termType
LIMIT $limit
```
**Evidence rules:**
- Prefer exact RXCUI matches before term-only matches.
- Always return the matched RXCUI.
**Guardrails:**
- Do not invent RXCUIs or terms.
- If no concept matches, return empty candidates.
**Output schema:**
- `rxcui`: `string`
- `term`: `string`
- `termType`: `string`
- `tty`: `string`

## Intent: rxnorm.drug_composition
**Goal:** Return ingredient, dose form, and tradename links for an RxNorm drug concept.
**Triggers:**
- drug ingredients
- what is in this medication
- rxnorm dose form
- brand and generic mapping
**Inputs:**
- `rxcui` (string; required=true; Target RxNorm RXCUI.)
- `limit` (int; required=false; default=25; Maximum number of relationship rows.)
**Required entities:**
- Concept
- Description
**Query plan:**
1. Traverse composition and naming relationships from the source drug.
```cypher
MATCH (src:Concept {rxcui: $rxcui})
MATCH (src)-[r:RELATIONSHIP]->(dst:Concept)
WHERE r.rela IN ['has_ingredient', 'has_precise_ingredient', 'has_dose_form', 'has_tradename']
OPTIONAL MATCH (dst)-[:HAS_DESCRIPTION]->(dd:Description)
RETURN src.rxcui AS sourceRxcui,
       r.rela AS relationType,
       dst.rxcui AS targetRxcui,
       dst.tty AS targetTty,
       dd.str AS targetTerm
LIMIT $limit
```
**Evidence rules:**
- Use RxNorm RELA semantics for composition and naming relationships.
- Return source and target RXCUI in every result row.
**Guardrails:**
- Only emit supported RELA values for this intent.
- If source RXCUI is missing, return empty results.
**Output schema:**
- `relationType`: `string`
- `sourceRxcui`: `string`
- `targetRxcui`: `string`
- `targetTerm`: `string`
- `targetTty`: `string`

{% else %}
No intents defined in the semantic layer.
{% endif %}
