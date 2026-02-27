# Intent Catalog: SNOMEDCT

{% if intent_sections %}
## Summary
- `concept.description`: Return concept descriptions (FSN, preferred term, synonyms) for a given concept.
- `disease.anatomic_site_laterality`: Return finding site and laterality qualifiers for a disease concept.
- `mapping.icd10_to_snomed_site_laterality`: Given an ICD-10 code, return mapped SNOMED concept(s) with finding site and laterality when available.
- `procedure.context`: Return key context for a procedure, including method and procedure site.

## Intent: concept.description
**Goal:** Return concept descriptions (FSN, preferred term, synonyms) for a given concept.
**Triggers:**
- description
- define
- what is
- synonym
- fsn
**Inputs:**
- `concept_id` (string; required=true; SNOMED concept identifier.)
- `language` (string; required=false; default=en; Description language code.)
- `limit` (int; required=false; default=20; Maximum number of descriptions.)
**Required entities:**
- Concept
- Description
**Query plan:**
1. Fetch active descriptions linked to the target concept.
```cypher
MATCH (c:Concept {concept_id: '$concept_id', active: 'true'})
MATCH (c)-[:HAS_DESCRIPTION]->(d:Description {language: '$language', active: 'true'})
RETURN c.concept_id AS conceptId,
       d.description_id AS descriptionId,
       d.term AS term,
       d.type AS descType,
       d.language AS language
ORDER BY d.type, d.term
LIMIT $limit
```
**Evidence rules:**
- Always return description type and language with each term.
- Prefer active concept and active descriptions.
**Guardrails:**
- Do not infer descriptions not present in graph nodes.
- If concept_id is missing, require a prior grounding step.
**Output schema:**
- `conceptId`: `string`
- `descType`: `string`
- `descriptionId`: `string`
- `language`: `string`
- `term`: `string`

## Intent: disease.anatomic_site_laterality
**Goal:** Return finding site and laterality qualifiers for a disease concept.
**Triggers:**
- where is located
- site
- anatomic location
- laterality
- left or right
**Inputs:**
- `disease_concept_id` (string; required=true; SNOMED disease concept identifier.)
- `limit` (int; required=false; default=20; Maximum number of attribute results.)
**Required entities:**
- Disease
- BodyStructure
- Laterality
**Query plan:**
1. Retrieve anatomical finding site and optional laterality attributes.
```cypher
MATCH (d:Concept {active: 'true'})-[:HAS_MAPPING]->(m:Mapping {map_target: '$map_target'})
OPTIONAL MATCH (d)-[:FINDING_SITE]->(site:Concept)
OPTIONAL MATCH (d)-[:LATERALITY]->(lat:Concept)
OPTIONAL MATCH (site)-[:HAS_DESCRIPTION]->(sd:Description {language: '$language', active: 'true'})
OPTIONAL MATCH (lat)-[:HAS_DESCRIPTION]->(ld:Description {language: '$language', active: 'true'})
OPTIONAL MATCH (d)-[:PREFERRED_TERM]->(cd:Description {language: '$language', active: 'true'})
RETURN m.map_target,cd.term, sd.term,ld.term
LIMIT $limit
```
**Evidence rules:**
- Use SNOMED FINDING_SITE alias for anatomical location.
- Use SNOMED LATERALITY alias.
**Guardrails:**
- Return empty laterality fields when laterality is not encoded.
- Do not assume bilateral when no laterality relationship is present.
**Output schema:**
- `diseaseConceptId`: `string`
- `lateralityConceptId`: `string`
- `lateralityTerm`: `string`
- `siteConceptId`: `string`
- `siteTerm`: `string`

## Intent: mapping.icd10_to_snomed_site_laterality
**Goal:** Given an ICD-10 code, return mapped SNOMED concept(s) with finding site and laterality when available.
**Triggers:**
- icd10 to snomed
- map icd10 code
- snomed from icd10
- site and laterality from icd10
- crosswalk icd10 snomed
**Inputs:**
- `icd10_code` (string; required=true; ICD-10 code to resolve through SNOMED mapping rows (map_target).)
- `map_refset_id` (string; required=false; default=447562003; Optional SNOMED map refset identifier to constrain mapping rows.)
- `language` (string; required=false; default=en; Description language code.)
- `limit` (int; required=false; default=50; Maximum number of mapped SNOMED rows.)
**Required entities:**
- Concept
- Mapping
- BodyStructure
- Laterality
- Description
**Query plan:**
1. Find SNOMED concepts mapped to an ICD-10 code and enrich with site/laterality when present.
```cypher
MATCH (s:Concept {active: 'true'})-[:HAS_MAPPING]->(m:Mapping {active: 'true'})
WHERE m.map_target = '$icd10_code'
  AND ('$map_refset_id' = '' OR m.refset_id = '$map_refset_id')
OPTIONAL MATCH (s)-[:HAS_DESCRIPTION]->(sd:Description {language: '$language', active: 'true'})
OPTIONAL MATCH (s)-[:FINDING_SITE]->(site:Concept {active: 'true'})
OPTIONAL MATCH (site)-[:HAS_DESCRIPTION]->(siteDesc:Description {language: '$language', active: 'true'})
OPTIONAL MATCH (s)-[:LATERALITY]->(lat:Concept {active: 'true'})
OPTIONAL MATCH (lat)-[:HAS_DESCRIPTION]->(latDesc:Description {language: '$language', active: 'true'})
RETURN m.map_target AS icd10Code,
       m.refset_id AS mapRefsetId,
       s.concept_id AS snomedConceptId,
       sd.term AS snomedTerm,
       site.concept_id AS siteConceptId,
       siteDesc.term AS siteTerm,
       lat.concept_id AS lateralityConceptId,
       latDesc.term AS lateralityTerm
LIMIT $limit
```
**Evidence rules:**
- Use HAS_MAPPING rows where Mapping.map_target equals the ICD-10 code.
- Use FINDING_SITE and LATERALITY attributes directly from mapped SNOMED concept.
- Return site/laterality fields as null when those attributes are not present.
**Guardrails:**
- Do not infer laterality from ICD-10 text when no SNOMED laterality relationship exists.
- If map_refset_id is provided, restrict rows to that refset only.
- If no mapping exists for the ICD-10 code, return empty results.
**Output schema:**
- `icd10Code`: `string`
- `lateralityConceptId`: `string`
- `lateralityTerm`: `string`
- `mapRefsetId`: `string`
- `siteConceptId`: `string`
- `siteTerm`: `string`
- `snomedConceptId`: `string`
- `snomedTerm`: `string`

## Intent: procedure.context
**Goal:** Return key context for a procedure, including method and procedure site.
**Triggers:**
- procedure details
- how is procedure done
- method used
- procedure site
- surgical approach
**Inputs:**
- `procedure_concept_id` (string; required=true; SNOMED procedure concept identifier.)
- `limit` (int; required=false; default=20; Maximum number of attribute rows.)
**Required entities:**
- Procedure
- Concept
**Query plan:**
1. Fetch procedure method and procedure site attributes.
```cypher
MATCH (p:Concept {concept_id: '$procedure_concept_id', active: 'true'})
OPTIONAL MATCH (p)-[:METHOD]->(m:Concept)
OPTIONAL MATCH (p)-[:PROCEDURE_SITE]->(s:Concept)
OPTIONAL MATCH (m)-[:HAS_DESCRIPTION]->(md:Description {language: '$language', active: 'true'})
OPTIONAL MATCH (s)-[:HAS_DESCRIPTION]->(sd:Description {language: '$language', active: 'true'})
RETURN p.concept_id AS procedureConceptId,
       m.concept_id AS methodConceptId,
       md.term AS methodTerm,
       s.concept_id AS siteConceptId,
       sd.term AS siteTerm
LIMIT $limit
```
**Evidence rules:**
- Use SNOMED relationship alias METHOD
- Use SNOMED relationship alias PROCEDURE_SITE
**Guardrails:**
- Return null fields when attributes are absent.
- Do not infer method or site from term text alone.
**Output schema:**
- `methodConceptId`: `string`
- `methodTerm`: `string`
- `procedureConceptId`: `string`
- `siteConceptId`: `string`
- `siteTerm`: `string`

{% else %}
No intents defined in the semantic layer.
{% endif %}
