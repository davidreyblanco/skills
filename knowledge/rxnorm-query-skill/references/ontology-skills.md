# Ontology Skills for `RxNorm`

RxNorm is a standardized nomenclature for clinical drugs produced by the U.S. National Library of Medicine. It provides normalized names and unique identifiers (RXCUIs) for medications, including their ingredients, dose forms, and brand names. RxNorm also defines relationships between drug concepts, enabling interoperability in electronic health records and clinical applications.

This document explains how to exploit the `RxNorm` clinical vocabulary using the project graph model.
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
| `Concept` | `attributes-key`=attributes, `id`=concept_id, `key`=rxcui, `label`=Concept |
| `Description` | `key`=desc_id, `label`=Description, `term-key`=str |

Node aliases:
| Label | Alias | Filters / Properties | Description |
| --- | --- | --- | --- |
| `Concept` | `BrandName` | `id`=`BN` | Brand name concept anchor. |
| `Concept` | `BrandedDrug` | `id`=`SBD` | Semantic branded drug concept anchor. |
| `Concept` | `BrandedDrugComponent` | `id`=`SBDC` | Semantic branded drug component concept anchor. |
| `Concept` | `BrandedDrugForm` | `id`=`SBDF` | Semantic branded drug form concept anchor. |
| `Concept` | `BrandedPack` | `id`=`BPCK` | Branded pack concept anchor. |
| `Concept` | `ClinicalDrug` | `id`=`SCD` | Semantic clinical drug concept anchor. |
| `Concept` | `ClinicalDrugComponent` | `id`=`SCDC` | Semantic clinical drug component concept anchor. |
| `Concept` | `ClinicalDrugForm` | `id`=`SCDF` | Semantic clinical drug form concept anchor. |
| `Concept` | `DoseForm` | `id`=`DF` | Dose form concept anchor. |
| `Concept` | `Example_Acetaminophen` | `id`=`161` | Example RXCUI alias for acetaminophen. |
| `Concept` | `Example_Amoxicillin` | `id`=`723` | Example RXCUI alias for amoxicillin. |
| `Concept` | `Example_Aspirin` | `id`=`1191` | Example RXCUI alias for aspirin. |
| `Concept` | `Example_Ibuprofen` | `id`=`5640` | Example RXCUI alias for ibuprofen. |
| `Concept` | `Example_Metformin` | `id`=`6809` | Example RXCUI alias for metformin. |
| `Concept` | `GenericPack` | `id`=`GPCK` | Generic pack concept anchor. |
| `Concept` | `Ingredient` | `id`=`IN` | Ingredient concept anchor. |
| `Concept` | `MultipleIngredients` | `id`=`MIN` | Multiple ingredients concept anchor. |
| `Concept` | `PreciseIngredient` | `id`=`PIN` | Precise ingredient concept anchor. |
| `Description` | `English` | `id`=`ENG` | English language description alias. |

### 4.2 Relationships
Relationship configuration:
| Relationship | Attributes |
| --- | --- |
| `HAS_DESCRIPTION` | `qualifier-key`=tty, `type`=HAS_DESCRIPTION |
| `IS_A` | `derived-from`=RELA, `type`=IS_A |
| `RELATIONSHIP` | `key`=['source_id', 'destination_id'], `type`=rela |

Relationship aliases:
| Group | Alias | Description | Code |
| --- | --- | --- | --- |
| `has_description` | `BN` | Brand name term type. | `BN` |
| `has_description` | `BPCK` | Branded pack term type. | `BPCK` |
| `has_description` | `DF` | Dose form term type. | `DF` |
| `has_description` | `ET` | Entry term type. | `ET` |
| `has_description` | `GPCK` | Generic pack term type. | `GPCK` |
| `has_description` | `IN` | Ingredient term type. | `IN` |
| `has_description` | `MIN` | Multiple ingredients term type. | `MIN` |
| `has_description` | `PIN` | Precise ingredient term type. | `PIN` |
| `has_description` | `SBD` | Semantic branded drug term type. | `SBD` |
| `has_description` | `SBDC` | Semantic branded drug component term type. | `SBDC` |
| `has_description` | `SBDC_OLD` | Legacy alias for semantic branded drug component. | `SBDC` |
| `has_description` | `SBDF` | Semantic branded drug form term type. | `SBDF` |
| `has_description` | `SCD` | Semantic clinical drug term type. | `SCD` |
| `has_description` | `SCDC` | Semantic clinical drug component term type. | `SCDC` |
| `has_description` | `SCDF` | Semantic clinical drug form term type. | `SCDF` |
| `relationship` | `HAS_DESCRIPTION` | Relationship from concept to description term entries. | `HAS_DESCRIPTION` |
| `relationship` | `IS_A` | Hierarchical relationship derived from selected RELA values. | `IS_A` |
| `relationship` | `RELATIONSHIP` | Generic relationship edge in the RxNorm graph model. | `RELATIONSHIP` |
| `rxnrel_rel` | `RB` | Broader-than relationship category. | `RB` |
| `rxnrel_rel` | `RN` | Narrower-than relationship category. | `RN` |
| `rxnrel_rel` | `RO` | Other relationship category. | `RO` |
| `rxnrel_rel` | `SY` | Synonymy-like relationship category. | `SY` |
| `rxnrel_rela` | `CONSTITUTED_FROM` | Concept is constituted from another concept. | `constituted_from` |
| `rxnrel_rela` | `CONSTITUTES` | Concept constitutes another RxNorm concept. | `constitutes` |
| `rxnrel_rela` | `CONTAINED_IN` | Concept is contained in a pack or other concept. | `contained_in` |
| `rxnrel_rela` | `CONTAINS` | Pack or concept contains another concept. | `contains` |
| `rxnrel_rela` | `DOSE_FORM_OF` | Dose form belongs to a drug concept. | `dose_form_of` |
| `rxnrel_rela` | `FORM_OF` | Concept is a form of another concept. | `form_of` |
| `rxnrel_rela` | `HAS_DOSE_FORM` | Drug concept has a dose form. | `has_dose_form` |
| `rxnrel_rela` | `HAS_FORM` | Concept has pharmaceutical form. | `has_form` |
| `rxnrel_rela` | `HAS_INGREDIENT` | Drug concept has ingredient. | `has_ingredient` |
| `rxnrel_rela` | `HAS_PRECISE_INGREDIENT` | Drug concept has precise ingredient. | `has_precise_ingredient` |
| `rxnrel_rela` | `HAS_TRADENAME` | Clinical drug has tradename relationship. | `has_tradename` |
| `rxnrel_rela` | `INGREDIENT_OF` | Ingredient belongs to a drug concept. | `ingredient_of` |
| `rxnrel_rela` | `PRECISE_INGREDIENT_OF` | Precise ingredient belongs to a drug concept. | `precise_ingredient_of` |
| `rxnrel_rela` | `TRADENAME_OF` | Tradename belongs to a clinical drug. | `tradename_of` |

### 4.3 Description Qualifiers
`HAS_DESCRIPTION` qualifiers:
| Alias | Value |
| --- | --- |
| `BN` | `BN` |
| `BPCK` | `BPCK` |
| `DF` | `DF` |
| `ET` | `ET` |
| `GPCK` | `GPCK` |
| `IN` | `IN` |
| `MIN` | `MIN` |
| `PIN` | `PIN` |
| `SBD` | `SBD` |
| `SBDC` | `SBDC` |
| `SBDC_OLD` | `SBDC` |
| `SBDF` | `SBDF` |
| `SCD` | `SCD` |
| `SCDC` | `SCDC` |
| `SCDF` | `SCDF` |

## 5. Usage Examples

```cypher
// Run against POST /cypher/RxNorm
MATCH (c:Concept)
RETURN c
LIMIT 5
```

```cypher
MATCH (n:BrandName)
RETURN n
LIMIT 20
```

```cypher
MATCH (a:Concept)-[r:BN]->(b:Concept)
RETURN a.concept_id, b.concept_id
LIMIT 20
```

```cypher
MATCH (c:Concept)-[:HAS_DESCRIPTION:BN]->(d:Description)
RETURN c.concept_id, d.term
LIMIT 20
```

## 6. Intent Catalog

Intents are entry points for natural language questions. Each intent defines:
- a name
- triggers (keywords/patterns)
- required entities (concepts)
- query plan (one or more Cypher templates)
- evidence rules and guardrails
- output schema

### Intent: rxnorm.code_lookup
**Goal:** Resolve an RXCUI or medication term to RxNorm concepts and canonical names.
**Triggers:**
- rxnorm code
- rxcui lookup
- find medication concept
- what drug is this code
**Inputs:**
- `text` (string; required; RXCUI fragment or medication name text.)
- `limit` (int; optional; default=20; Maximum number of candidate concepts.)
**Required entities:**
- `Concept`
- `Description`
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

### Intent: rxnorm.drug_composition
**Goal:** Return ingredient, dose form, and tradename links for an RxNorm drug concept.
**Triggers:**
- drug ingredients
- what is in this medication
- rxnorm dose form
- brand and generic mapping
**Inputs:**
- `rxcui` (string; required; Target RxNorm RXCUI.)
- `limit` (int; optional; default=25; Maximum number of relationship rows.)
**Required entities:**
- `Concept`
- `Description`
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

## 7. Source
- Semantic layer file used: `RxNorm.yaml`
