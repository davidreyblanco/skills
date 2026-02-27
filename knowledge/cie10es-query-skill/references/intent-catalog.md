# Intent Catalog: CIE10ES

## Summary
- `cie10es.code_description`: Return CIE10-ES code details and preferred descriptions for a disease concept.
- `cie10es.neoplasia_behavior_site`: Retrieve neoplasia behavior and linked anatomic disease location context in CIE10-ES.

## Intent: cie10es.code_description
**Goal:** Return CIE10-ES code details and preferred descriptions for a disease concept.
**Triggers:**
- código cie10
- descripcion del codigo
- qué significa el código
- detalle diagnóstico
**Inputs:**
- `code` (string; required=true; CIE10-ES diagnostic code (e.g., A00, J18.9).)
- `limit` (int; required=false; default=20; Maximum number of descriptions to return.)
**Required entities:**
- Disease
- Description
**Query plan:**
1. Fetch code and active descriptions for the target disease concept.
```cypher
MATCH (d:Disease {code: $code})
MATCH (d)-[:HAS_DESCRIPTION]->(dsc:Description {active: 'true'})
RETURN d.code AS code,
       dsc.term AS term,
       dsc.language AS language,
       dsc.type AS descriptionType
ORDER BY dsc.type, dsc.term
LIMIT $limit
```
**Evidence rules:**
- Only return descriptions attached to the matched code.
- Prioritize active descriptions.
**Guardrails:**
- If code is not found, return empty results and ask for code validation.
- Do not infer definitions from external sources.
**Output schema:**
- `code`: `string`
- `descriptionType`: `string`
- `language`: `string`
- `term`: `string`

## Intent: cie10es.neoplasia_behavior_site
**Goal:** Retrieve neoplasia behavior and linked anatomic disease location context in CIE10-ES.
**Triggers:**
- neoplasia
- comportamiento tumoral
- tumor benigno/maligno
- localizacion neoplasia
**Inputs:**
- `limit` (int; required=false; default=20; Maximum number of neoplasia matches.)
**Required entities:**
- Neoplasm
- Disease
- Description
**Query plan:**
1. Resolve neoplasia concepts by relationship type (NEOPLASM_BENIGN or NEOPLASM_MALIGNANT_PRIMARY) .
```cypher
MATCH (n:Neoplasm)-[r:NEOPLASM_BENIGN]->(d:Disease)
OPTIONAL MATCH (n)-[:HAS_DESCRIPTION]->(nd:Description {active: 'true'})
OPTIONAL MATCH (d)-[:HAS_DESCRIPTION]->(dd:Description {active: 'true'})
RETURN n.code AS neoplasmCode,
       nd.term AS neoplasmTerm,
       r.type_id AS behaviorType,
       d.code AS diseaseCode,
       dd.term AS diseaseTerm
LIMIT $limit
```
**Evidence rules:**
- Behavior must come from neoplasia behavior aliases configured in semantic layer.
- Return both neoplasia and disease evidence terms when available.
**Guardrails:**
- Reject unknown behavior aliases.
- Do not fabricate behavior mappings when no relationship exists.
**Output schema:**
- `behaviorType`: `string`
- `diseaseCode`: `string`
- `diseaseTerm`: `string`
- `neoplasmCode`: `string`
- `neoplasmTerm`: `string`
