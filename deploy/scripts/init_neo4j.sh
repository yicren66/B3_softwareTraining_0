#!/bin/bash
# =============================================================================
# Neo4j initialisation for Jujube Platform
# Runs Cypher migrations on first startup.
# =============================================================================
set -e

echo "[init_neo4j] Waiting for Neo4j to be ready..."

# Wait for Neo4j HTTP endpoint
until wget -qO- http://localhost:7474 > /dev/null 2>&1; do
    sleep 2
done

echo "[init_neo4j] Neo4j is ready. Running migrations..."

for cypher_file in /app/services/knowledge-graph/src/neo4j/migrations/*.cypher; do
    if [ -f "$cypher_file" ]; then
        echo "[init_neo4j] Running: $cypher_file"
        cat "$cypher_file" | cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" || true
    fi
done

echo "[init_neo4j] Migration complete."
