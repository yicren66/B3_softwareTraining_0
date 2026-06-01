// ============================================================================
// Migration 001: Initial Ontology — Constraints & Indexes
// Description: Creates uniqueness constraints and fulltext indexes for the
//              jujube pest & disease knowledge graph core entity labels.
// ============================================================================

// ---------------------------------------------------------------------------
// 1. Uniqueness constraints on entity_id for all entity labels
//    Ensures every entity (pest, disease, symptom, method, etc.) has a unique
//    identifier so MERGE operations are safe and the graph stays consistent.
// ---------------------------------------------------------------------------

CREATE CONSTRAINT pest_disease_unique_id IF NOT EXISTS
FOR (n:PestDisease) REQUIRE n.entity_id IS UNIQUE;

CREATE CONSTRAINT prevention_method_unique_id IF NOT EXISTS
FOR (n:PreventionMethod) REQUIRE n.entity_id IS UNIQUE;

CREATE CONSTRAINT pesticide_unique_id IF NOT EXISTS
FOR (n:Pesticide) REQUIRE n.entity_id IS UNIQUE;

CREATE CONSTRAINT symptom_unique_id IF NOT EXISTS
FOR (n:Symptom) REQUIRE n.entity_id IS UNIQUE;

CREATE CONSTRAINT growth_cycle_unique_id IF NOT EXISTS
FOR (n:GrowthCycle) REQUIRE n.entity_id IS UNIQUE;

CREATE CONSTRAINT geographic_region_unique_id IF NOT EXISTS
FOR (n:GeographicRegion) REQUIRE n.entity_id IS UNIQUE;

CREATE CONSTRAINT climate_condition_unique_id IF NOT EXISTS
FOR (n:ClimateCondition) REQUIRE n.entity_id IS UNIQUE;

CREATE CONSTRAINT control_strategy_unique_id IF NOT EXISTS
FOR (n:ControlStrategy) REQUIRE n.entity_id IS UNIQUE;

// ---------------------------------------------------------------------------
// 2. Fulltext indexes for name / description search
//    Powers the semantic search and auto-complete features in the knowledge-
//    graph service, matching against both Chinese and English text fields.
// ---------------------------------------------------------------------------

CREATE FULLTEXT INDEX pest_disease_fulltext IF NOT EXISTS
FOR (n:PestDisease)
ON EACH [n.name_cn, n.name_en, n.description, n.scientific_name]
OPTIONS {indexConfig: {`fulltext.analyzer`: 'cjk'}};

CREATE FULLTEXT INDEX symptom_fulltext IF NOT EXISTS
FOR (n:Symptom)
ON EACH [n.name_cn, n.description, n.affected_parts]
OPTIONS {indexConfig: {`fulltext.analyzer`: 'cjk'}};

CREATE FULLTEXT INDEX prevention_method_fulltext IF NOT EXISTS
FOR (n:PreventionMethod)
ON EACH [n.name_cn, n.description, n.category]
OPTIONS {indexConfig: {`fulltext.analyzer`: 'cjk'}};

// ---------------------------------------------------------------------------
// 3. Single-property indexes for high-frequency lookup fields
// ---------------------------------------------------------------------------

CREATE INDEX pest_disease_name_cn IF NOT EXISTS
FOR (n:PestDisease) ON (n.name_cn);

CREATE INDEX pest_disease_category IF NOT EXISTS
FOR (n:PestDisease) ON (n.category);

// ============================================================================
// End of Migration 001
// ============================================================================
