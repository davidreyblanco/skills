# Ontology Skills for `SNOMEDCT`

SNOMED CT (Systematized Nomenclature of Medicine—Clinical Terms) is a comprehensive, standardized clinical healthcare terminology used to represent medical concepts like diseases, symptoms, procedures, findings, and body structures in a consistent way. It enables interoperable electronic health records by providing unique codes and relationships between concepts, making clinical data easier to record, share, analyze, and reuse across systems and countries

This document explains how to exploit the `SNOMEDCT` clinical vocabulary using the project graph model.
Knowledge is stored as a graph data structure and accessed through Cypher Query Language, including the platform's Cypher extensions.
The guide first describes the base model, then the semantic layer extension that introduces semantic entities and relationships for more expressive queries.

## 1. Base Graph Model
The base model follows the project graph documentation and defines the essential entities and edges used across vocabularies.

Essential nodes:
- `Concept`: canonical clinical concept in a given vocabulary/version.
- `Description`: term strings for a concept (language, type, active state).
- `Mapping`: cross-vocabulary or logical mapping records when available.

Essential relationships:
- `IS_A`: hierarchy edge between concepts.
- `RELATIONSHIP`: typed concept-to-concept semantic edge.
- `HAS_DESCRIPTION`: concept-to-description lexical edge.
- `HAS_MAPPING`: concept-to-mapping edge when mapping content exists.

Model rules:
- Keep `vocabulary`, `version_id`, and `active` on nodes and relationships.
- Treat `Description.language` as explicit for multilingual filtering.
- Query the latest version when `version_id` is not pinned.

## 2. Semantic Layer Extension
The semantic layer is an extension over the base model. It adds semantic entities (node aliases) and semantic relationships (relationship aliases/modifiers) to make queries concise and domain-oriented.
These extensions do not require changing Cypher fundamentals; they translate to base graph patterns and properties.

## 3. Extended Cypher Cheatsheet
We added in some changes to make the data model more usable:
- Concept by ID: `MATCH (c:Concept:40733004) RETURN c`
- Concept by term: `MATCH (c:Concept:<Anthrax>) RETURN c`
- Description language: `MATCH (d:Description:es) RETURN d`
- Semantic alias: `MATCH (c:High) RETURN c`
- Relationship alias: `MATCH (a)-[r:HAS_INTERPRETATION]->(b) RETURN a,b`

## 4. Semantic Layer Definition
### 4.1 Entities
Node configuration:
| Node | Key Fields |
| --- | --- |
| `Concept` | `id`=concept_id, `key`=concept_id, `label`=Concept |
| `Description` | `key`=description_id, `label`=Description, `term-key`=term |

Node aliases:
| Label | Alias | Filters / Properties | Description |
| --- | --- | --- | --- |
| `Concept` | `Absent` | `id`=`2667000` | Qualifier value indicating absent. |
| `Concept` | `Acute` | `id`=`373933003` | Qualifier value indicating acute course. |
| `Concept` | `BacterialInfectiousDisease` | `id`=`87628006` | Bacterial infectious disease hierarchy anchor. |
| `Concept` | `BodyStructure` | `id`=`123037004` | Body structure hierarchy anchor. |
| `Concept` | `Brain` | `id`=`12738006` | Brain structure concept anchor. |
| `Concept` | `BrainStructure` | `id`=`12738006` | Alias of Brain structure concept. |
| `Concept` | `Chronic` | `id`=`90734009` | Qualifier value indicating chronic course. |
| `Concept` | `ClinicalFinding` | `id`=`404684003` | Top-level clinical finding hierarchy anchor. |
| `Concept` | `Disease` | `id`=`64572001` | Disease hierarchy anchor. |
| `Concept` | `Embolism` | `id`=`414086009` | Embolism disorder concept anchor. |
| `Concept` | `Heart` | `id`=`80891009` | Heart structure concept anchor. |
| `Concept` | `HeartDisease` | `id`=`56265001` | Heart disease hierarchy anchor. |
| `Concept` | `HeartStructure` | `id`=`80891009` | Alias of Heart structure concept. |
| `Concept` | `High` | `id`=`75540009` | Qualifier value indicating high. |
| `Concept` | `InfectiousDisease` | `id`=`40733004` | Infectious disease hierarchy anchor. |
| `Concept` | `Kidney` | `id`=`64033007` | Kidney structure concept anchor. |
| `Concept` | `KidneyStructure` | `id`=`64033007` | Alias of Kidney structure concept. |
| `Concept` | `LateralityLeft` | `id`=`7771000` | Left-sided laterality qualifier. |
| `Concept` | `LateralityRight` | `id`=`24028007` | Right-sided laterality qualifier. |
| `Concept` | `Liver` | `id`=`10200004` | Liver structure concept anchor. |
| `Concept` | `LiverStructure` | `id`=`10200004` | Alias of Liver structure concept. |
| `Concept` | `Low` | `id`=`62482003` | Qualifier value indicating low. |
| `Concept` | `Lung` | `id`=`39607008` | Lung structure concept anchor. |
| `Concept` | `LungStructure` | `id`=`39607008` | Alias of Lung structure concept. |
| `Concept` | `Mild` | `id`=`255604002` | Qualifier value indicating mild intensity. |
| `Concept` | `Moderate` | `id`=`6736007` | Qualifier value indicating moderate intensity. |
| `Concept` | `Negative` | `id`=`260385009` | Qualifier value indicating negative. |
| `Concept` | `Normal` | `id`=`17621005` | Qualifier value indicating normal. |
| `Concept` | `Organism` | `id`=`410607006` | Organism hierarchy anchor. |
| `Concept` | `PharmaceuticalProduct` | `id`=`373873005` | Pharmaceutical/biologic product hierarchy anchor. |
| `Concept` | `Pneumonia` | `id`=`233604007` | Pneumonia disorder concept anchor. |
| `Concept` | `Positive` | `id`=`10828004` | Qualifier value indicating positive. |
| `Concept` | `Present` | `id`=`52101004` | Qualifier value indicating present. |
| `Concept` | `Procedure` | `id`=`71388002` | Procedure hierarchy anchor. |
| `Concept` | `Sepsis` | `id`=`91302008` | Sepsis disorder concept anchor. |
| `Concept` | `Severe` | `id`=`24484000` | Qualifier value indicating severe intensity. |
| `Concept` | `Thrombosis` | `id`=`264579008` | Thrombosis disorder concept anchor. |
| `Concept` | `UrinaryTractInfection` | `id`=`68566005` | Urinary tract infection disorder concept anchor. |
| `Concept` | `ViralInfectiousDisease` | `id`=`34014006` | Viral infectious disease hierarchy anchor. |

### 4.2 Relationships
Relationship configuration:
| Relationship | Attributes |
| --- | --- |
| `HAS_DESCRIPTION` | `key`=['source_id', 'destination_id'], `type`=type_id |
| `HAS_MAPPING` | `key`=['concept_id', 'map_target'], `type`=type_id |
| `IS_A` | `key`=['source_id', 'destination_id'], `type`=type_id |
| `RELATIONSHIP` | `key`=['source_id', 'destination_id'], `type`=type_id |

Relationship aliases:
| Group | Alias | Description | Code |
| --- | --- | --- | --- |
| `clinical` | `ASSOCIATED_MORPHOLOGY` | Morphologic abnormality associated with a finding. | `116676008` |
| `clinical` | `ASSOCIATED_WITH` | General association between concepts. | `47429007` |
| `clinical` | `CAUSATIVE_AGENT` | Agent (organism or substance) causing the condition. | `246075003` |
| `clinical` | `CAUSED_BY` | Alias of CAUSATIVE_AGENT. | `246075003` |
| `clinical` | `CLINICAL_COURSE` | Clinical progression pattern over time. | `263502005` |
| `clinical` | `DUE_TO` | Relationship expressing underlying cause. | `42752001` |
| `clinical` | `FINDING_SITE` | Anatomical site associated with a clinical finding. | `363698007` |
| `clinical` | `HAS_REALIZATION` | Links an abstract disorder to a realized manifestation. | `719722006` |
| `clinical` | `LATERALITY` | Side of body qualifier (left/right/bilateral). | `272741003` |
| `clinical` | `MORPHOLOGY` | Alias of ASSOCIATED_MORPHOLOGY. | `116676008` |
| `clinical` | `OCCURRENCE` | Timing or onset occurrence qualifier. | `246454002` |
| `clinical` | `PATHOLOGICAL_PROCESS` | Pathophysiologic process involved in the condition. | `370135005` |
| `clinical` | `SEVERITY` | Severity qualifier for a clinical finding. | `246112005` |
| `clinical` | `SITE` | Alias of FINDING_SITE for anatomical location. | `363698007` |
| `has_description` | `DEFINITION` | Definition description type. | `900000000000550004` |
| `has_description` | `FSN` | Fully specified name description type. | `900000000000003001` |
| `has_description` | `FULLY_SPECIFIED_NAME` | Alias of FSN description type. | `900000000000003001` |
| `has_description` | `PREFERRED_TERM` | Preferred term description type alias. | `900000000000003001` |
| `has_description` | `SYNONYM` | Synonym description type. | `900000000000013009` |
| `observation` | `FINDING_METHOD` | Method used to obtain or determine the finding. | `418775008` |
| `observation` | `HAS_FOCUS` | Focus concept for an observation or finding. | `363702006` |
| `observation` | `HAS_INTERPRETATION` | Interpretation qualifier linked to an observation. | `363713009` |
| `observation` | `INTERPRETS` | Specifies what an observation interprets. | `363714003` |
| `procedure` | `METHOD` | Technique or method used by a procedure. | `260686004` |
| `procedure` | `PROCEDURE_SITE` | Anatomical site where the procedure is performed. | `363704007` |
| `product` | `HAS_ACTIVE_INGREDIENT` | Active ingredient of a medicinal product. | `127489000` |
| `product` | `HAS_MANUFACTURED_DOSE_FORM` | Manufactured pharmaceutical dose form of a product. | `411116001` |
| `product` | `ROUTE_OF_ADMINISTRATION` | Route by which a product is administered. | `410675002` |
| `temporal` | `AFTER` | Temporal relationship indicating sequence after another event. | `255234002` |
| `temporal` | `DURING` | Temporal relationship indicating coexistence during another event. | `371881003` |
| `temporal` | `TEMPORALLY_RELATED_TO` | General temporal association relationship. | `726633004` |

### 4.3 Description Qualifiers
`HAS_DESCRIPTION` qualifiers:
| Alias | Value |
| --- | --- |
| `DEFINITION` | `900000000000550004` |
| `FSN` | `900000000000003001` |
| `FULLY_SPECIFIED_NAME` | `900000000000003001` |
| `PREFERRED_TERM` | `900000000000003001` |
| `SYNONYM` | `900000000000013009` |

## 5. Usage Examples

```cypher
// Run against POST /cypher/SNOMEDCT
MATCH (c:Concept)
RETURN c
LIMIT 5
```

```cypher
MATCH (n:Absent)
RETURN n
LIMIT 20
```

```cypher
MATCH (a:Concept)-[r:ASSOCIATED_MORPHOLOGY]->(b:Concept)
RETURN a.concept_id, b.concept_id
LIMIT 20
```

```cypher
MATCH (c:Concept)-[:HAS_DESCRIPTION:DEFINITION]->(d:Description)
RETURN c.concept_id, d.term
LIMIT 20
```

Find disease concepts with lung as finding site.
```cypher
MATCH (d:Disease)-[r:FINDING_SITE]->(s:LungStructure)
WITH d
LIMIT 10
RETURN d
```

Find descendants of heart disease.
```cypher
MATCH (child:Concept)-[:IS_A]->(parent:HeartDisease)
WITH child
LIMIT 10
RETURN child
```

Find pharmaceutical products and their active ingredients.
```cypher
MATCH (p:PharmaceuticalProduct)-[r:HAS_ACTIVE_INGREDIENT]->(i:Concept)
WITH p, i
LIMIT 10
RETURN p, i
```

## 6. Intent Catalog

Intents are entry points for natural language questions. Each intent defines:
- a name
- triggers (keywords/patterns)
- required entities (concepts)
- query plan (one or more Cypher templates)
- evidence rules and guardrails
- output schema

### Intent: concept.description
**Goal:** Return concept descriptions (FSN, preferred term, synonyms) for a given concept.
**Triggers:**
- description
- define
- what is
- synonym
- fsn
**Inputs:**
- `concept_id` (string; required; SNOMED concept identifier.)
- `language` (string; optional; default=en; Description language code.)
- `limit` (int; optional; default=20; Maximum number of descriptions.)
**Required entities:**
- `Concept`
- `Description`
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

### Intent: disease.anatomic_site_laterality
**Goal:** Return finding site and laterality qualifiers for a disease concept.
**Triggers:**
- where is located
- site
- anatomic location
- laterality
- left or right
**Inputs:**
- `disease_concept_id` (string; required; SNOMED disease concept identifier.)
- `limit` (int; optional; default=20; Maximum number of attribute results.)
**Required entities:**
- `Disease`
- `BodyStructure`
- `Laterality`
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

### Intent: mapping.icd10_to_snomed_site_laterality
**Goal:** Given an ICD-10 code, return mapped SNOMED concept(s) with finding site and laterality when available.
**Triggers:**
- icd10 to snomed
- map icd10 code
- snomed from icd10
- site and laterality from icd10
- crosswalk icd10 snomed
**Inputs:**
- `icd10_code` (string; required; ICD-10 code to resolve through SNOMED mapping rows (map_target).)
- `map_refset_id` (string; optional; default=447562003; Optional SNOMED map refset identifier to constrain mapping rows.)
- `language` (string; optional; default=en; Description language code.)
- `limit` (int; optional; default=50; Maximum number of mapped SNOMED rows.)
**Required entities:**
- `Concept`
- `Mapping`
- `BodyStructure`
- `Laterality`
- `Description`
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

### Intent: procedure.context
**Goal:** Return key context for a procedure, including method and procedure site.
**Triggers:**
- procedure details
- how is procedure done
- method used
- procedure site
- surgical approach
**Inputs:**
- `procedure_concept_id` (string; required; SNOMED procedure concept identifier.)
- `limit` (int; optional; default=20; Maximum number of attribute rows.)
**Required entities:**
- `Procedure`
- `Concept`
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

## 7. Source
- Semantic layer file used: `SNOMEDCT.yaml`
