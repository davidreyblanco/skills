# Ontology Skills for `CIE10ES`

CIE10-ES (International Classification of Diseases, 10th revision, Spanish version) is a standardized classification system for diseases and health conditions in Spain. It is used for clinical documentation, epidemiological research, and health statistics.

This document explains how to exploit the `CIE10ES` clinical vocabulary using the project graph model.
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
| `Concept` | `id`=code, `key`=code, `label`=Concept |
| `Description` | `key`=description_id, `label`=Description, `term-key`=term |

Node aliases:
| Label | Alias | Filters / Properties | Description |
| --- | --- | --- | --- |
| `Concept` | `Disease` | `family`=`cie10-es` | Disease concept for CIE10-ES classification. |
| `Concept` | `Drug` | `source_kind`=`special_drug` | Drug |
| `Concept` | `Neoplasm` | `source_kind`=`special_neoplasia_node` | Neoplasm |
| `Concept` | `Procedure` | `family`=`cie10-pcs` | Procedure concept for CIE10-ES classification. |

### 4.2 Relationships
Relationship configuration:
| Relationship | Attributes |
| --- | --- |
| `HAS_DESCRIPTION` | `key`=['source_id', 'destination_id'], `type`=type_id |
| `IS_A` | `key`=['source_id', 'destination_id'], `type`=type_id |

Relationship aliases:
| Group | Alias | Description | Code |
| --- | --- | --- | --- |
| `drug_relationships` | `ACCIDENTAL_POISONING` | Envenenamiento Accidental | `DRUG_REL_B` |
| `drug_relationships` | `ASSAULT_POISONING` | Envenenamiento Asalto | `DRUG_REL_C` |
| `drug_relationships` | `HAS_ADVERSE_EFFECTS` | Efectos Adversos | `DRUG_REL_A` |
| `drug_relationships` | `SELF_POISONING` | Envenenamiento Intencional (Autolesión) | `DRUG_REL_E` |
| `drug_relationships` | `UNDERDOSIFICATION` | Infradosificación | `DRUG_REL_F` |
| `drug_relationships` | `UNDETERMINED_POISONING` | Envenenamiento Indeterminado | `DRUG_REL_D` |
| `neoplasia_behavior` | `NEOPLASM_BENIGN` | Benigna | `NEOPLASIA_CODE_A` |
| `neoplasia_behavior` | `NEOPLASM_IN_SITU` | In Situ | `NEOPLASIA_CODE_D` |
| `neoplasia_behavior` | `NEOPLASM_MALIGNANT_PRIMARY` | Maligna Primaria | `NEOPLASIA_CODE_E` |
| `neoplasia_behavior` | `NEOPLASM_MALIGNANT_SECONDARY` | Maligna Secundaria | `NEOPLASIA_CODE_F` |
| `neoplasia_behavior` | `NEOPLASM_UNCERTAIN_BEHAVIOR` | Comportamiento Incierto | `NEOPLASIA_CODE_B` |
| `neoplasia_behavior` | `NEOPLASM_UNSPECIFIED_BEHAVIOR` | Comportamiento No Especificado | `NEOPLASIA_CODE_C` |

### 4.3 Description Qualifiers
No `HAS_DESCRIPTION` qualifiers were found.

## 5. Usage Examples

```cypher
// Run against POST /cypher/CIE10ES
MATCH (c:Concept)
RETURN c
LIMIT 5
```

```cypher
MATCH (n:Disease)
RETURN n
LIMIT 20
```

```cypher
MATCH (a:Concept)-[r:ACCIDENTAL_POISONING]->(b:Concept)
RETURN a.concept_id, b.concept_id
LIMIT 20
```

Find underdosification relationships between drugs and diseases.
```cypher
MATCH (c:Drug)-[n:UNDERDOSIFICATION]->(x:Disease)
with n
limit 10
RETURN n
```

Find neoplasms with uncertain behavior.
```cypher
MATCH (c:Neoplasm)-[n:NEOPLASM_UNCERTAIN_BEHAVIOR]->(x:Disease)
with c
limit 10
RETURN c
```

Find neoplasms with uncertain behavior.
```cypher
MATCH (child:Concept:'A00')-[:IS_A]->(parent:Disease)
RETURN child.code, parent.code
LIMIT 1
```

## 6. Intent Catalog

Intents are entry points for natural language questions. Each intent defines:
- a name
- triggers (keywords/patterns)
- required entities (concepts)
- query plan (one or more Cypher templates)
- evidence rules and guardrails
- output schema

### Intent: cie10es.code_description
**Goal:** Return CIE10-ES code details and preferred descriptions for a disease concept.
**Triggers:**
- código cie10
- descripcion del codigo
- qué significa el código
- detalle diagnóstico
**Inputs:**
- `code` (string; required; CIE10-ES diagnostic code (e.g., A00, J18.9).)
- `limit` (int; optional; default=20; Maximum number of descriptions to return.)
**Required entities:**
- `Disease`
- `Description`
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

### Intent: cie10es.neoplasia_behavior_site
**Goal:** Retrieve neoplasia behavior and linked anatomic disease location context in CIE10-ES.
**Triggers:**
- neoplasia
- comportamiento tumoral
- tumor benigno/maligno
- localizacion neoplasia
**Inputs:**
- `limit` (int; optional; default=20; Maximum number of neoplasia matches.)
**Required entities:**
- `Neoplasm`
- `Disease`
- `Description`
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

## 7. Source
- Semantic layer file used: `CIE10ES.yaml`
