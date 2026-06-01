// ============================================================================
// Migration 002: Additional Constraints & Composite Indexes
// Description: Adds composite / single-property indexes for high-cardinality
//              query patterns used by the query planner after the initial
//              ontology constraints are in place.
// ============================================================================

// ---------------------------------------------------------------------------
// 1. Composite index — PestDisease(category, name_cn)
//    Optimises queries that filter by category first (e.g., "病害" / "虫害")
//    and then sort or filter by Chinese name.
// ---------------------------------------------------------------------------

CREATE INDEX pest_disease_category_name_cn IF NOT EXISTS
FOR (n:PestDisease) ON (n.category, n.name_cn);

// ---------------------------------------------------------------------------
// 2. Index on GrowthCycle.stage_name
//    Powers the phenology timeline — queries that join pests to the growth
//    stage in which they are active (e.g., "what pests threaten 花期?").
// ---------------------------------------------------------------------------

CREATE INDEX growth_cycle_stage_name IF NOT EXISTS
FOR (n:GrowthCycle) ON (n.stage_name);

// ---------------------------------------------------------------------------
// 3. Index on GeographicRegion.county
//    Regional drill-down queries (e.g., "list all pests in 沧县") hit this
//    index for fast filtering before traversing the graph.
// ---------------------------------------------------------------------------

CREATE INDEX geographic_region_county IF NOT EXISTS
FOR (n:GeographicRegion) ON (n.county);

// ---------------------------------------------------------------------------
// 4. Index on Pesticide.name_cn
//    Speeds up pesticide lookup by Chinese name in the treatment
//    recommendation engine and the pesticide-detail views.
// ---------------------------------------------------------------------------

CREATE INDEX pesticide_name_cn IF NOT EXISTS
FOR (n:Pesticide) ON (n.name_cn);

// ============================================================================
// End of Migration 002
// ============================================================================
