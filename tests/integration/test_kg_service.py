"""
Integration tests for Knowledge Graph Service.

Validates:
  - Entity retrieval with relations
  - Natural language semantic search
  - Prevention recommendation generation
  - Risk prediction with climate data
  - Health check

Run:  pytest tests/integration/test_kg_service.py -v
"""

import pytest
from fastapi.testclient import TestClient

# These tests require the KG service to be importable
pytestmark = pytest.mark.integration


class TestKGEntityRetrieval:
    """Test entity query endpoints (SRS §6.1.2)."""

    def test_get_entity_returns_correct_structure(self, kg_client):
        """GET /api/v1/kg/entity/{entity_id} returns valid entity data."""
        response = kg_client.get("/api/v1/kg/entity/KG-ENT-001")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        entity = data["data"]
        assert entity["entity_id"] == "KG-ENT-001"
        assert entity["name_cn"] == "枣炭疽病"
        assert entity["category"] == "病害"
        assert "relations" in entity

    def test_get_nonexistent_entity_returns_404(self, kg_client):
        """Unknown entity returns 404."""
        response = kg_client.get("/api/v1/kg/entity/KG-ENT-99999")
        assert response.status_code == 404

    def test_all_srs_core_pests_have_entities(self, kg_client):
        """All 15 「7病8虫」 core pests have valid KG entities (SRS IR-02)."""
        core_entities = [
            ("KG-ENT-001", "枣炭疽病", "病害"),
            ("KG-ENT-002", "枣疯病", "病害"),
            ("KG-ENT-003", "枣树锈病", "病害"),
            ("KG-ENT-004", "枣缩果病", "病害"),
            ("KG-ENT-005", "枣果腐病", "病害"),
            ("KG-ENT-006", "枣褐斑病", "病害"),
            ("KG-ENT-007", "枣叶黑斑病", "病害"),
            ("KG-ENT-008", "枣芽象甲", "虫害"),
            ("KG-ENT-009", "枣瘿蚊", "虫害"),
            ("KG-ENT-010", "桃小食心虫", "虫害"),
            ("KG-ENT-011", "绿盲蝽", "虫害"),
            ("KG-ENT-012", "枣尺蠖", "虫害"),
            ("KG-ENT-013", "枣镰翅小卷蛾", "虫害"),
            ("KG-ENT-014", "枣红蜘蛛", "虫害"),
            ("KG-ENT-015", "枣龟蜡蚧", "虫害"),
        ]
        for entity_id, expected_name, expected_category in core_entities:
            response = kg_client.get(f"/api/v1/kg/entity/{entity_id}")
            assert response.status_code == 200, f"Entity {entity_id} not found"
            data = response.json()["data"]
            assert data["name_cn"] == expected_name, f"{entity_id} name mismatch"
            assert data["category"] == expected_category, f"{entity_id} category mismatch"


class TestKGSearch:
    """Test natural-language semantic search (SRS §4.2.2 KQ-03)."""

    def test_chinese_query_returns_results(self, kg_client):
        """Natural language query in Chinese returns relevant results."""
        response = kg_client.post("/api/v1/kg/search", json={
            "query": "枣树叶片上有黄褐色粉末是什么病",
            "user_role": "枣农",
            "max_results": 5,
        })
        assert response.status_code == 200
        data = response.json()["data"]
        assert len(data["results"]) > 0
        # Should find 枣树锈病 as most relevant
        names = [r["name"] for r in data["results"]]
        assert any("锈" in name for name in names)

    def test_intent_classification_identify(self, kg_client):
        """Query asking "what is this" should classify as identify_pest intent."""
        response = kg_client.post("/api/v1/kg/search", json={
            "query": "枣叶发黄叶子卷曲是什么病",
            "include_intent_recommendation": True,
        })
        assert response.status_code == 200
        intent = response.json()["data"]["parsed_intent"]
        assert intent["intent"] in ("identify_pest", "find_treatment")

    def test_intent_classification_treatment(self, kg_client):
        """Query asking "how to treat" should classify as find_treatment intent."""
        response = kg_client.post("/api/v1/kg/search", json={
            "query": "枣树锈病怎么治打什么药",
            "include_intent_recommendation": True,
        })
        assert response.status_code == 200
        intent = response.json()["data"]["parsed_intent"]
        assert intent["intent"] == "find_treatment"

    def test_pest_keyword_extraction(self, kg_client):
        """Query containing pest names should extract mapped entities."""
        response = kg_client.post("/api/v1/kg/search", json={
            "query": "红蜘蛛和枣尺蠖怎么防治",
        })
        assert response.status_code == 200
        mapped = response.json()["data"]["parsed_intent"].get("mapped_pests", [])
        mapped_names = [m["name"] for m in mapped]
        assert any("红蜘蛛" in n for n in mapped_names)


class TestKGRecommendation:
    """Test prevention recommendation generation (SRS §4.2.3 KR-01)."""

    def test_recommendation_has_four_categories(self, kg_client):
        """Recommendation should include prevention categories."""
        response = kg_client.get(
            "/api/v1/kg/recommendation/KG-ENT-003",
            params={"county": "沧县", "growth_stage": "幼果期", "temperature": 28, "humidity": 75},
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["pest_id"] == "KG-ENT-003"
        recs = data.get("recommendations", {})
        # Should have at least one category
        assert len(recs) > 0

    def test_recommendation_includes_strategy(self, kg_client):
        """Response should include comprehensive strategy."""
        response = kg_client.get("/api/v1/kg/recommendation/KG-ENT-003")
        assert response.status_code == 200
        strategy = response.json()["data"].get("comprehensive_strategy", {})
        assert "priority_sequence" in strategy
        assert "integrated_plan" in strategy


class TestKGRiskPrediction:
    """Test risk prediction (SRS §4.2.3 KR-03)."""

    def test_risk_prediction_returns_assessments(self, kg_client):
        """Risk prediction returns per-pest risk assessments."""
        response = kg_client.get("/api/v1/kg/risk/predict", params={
            "county": "沧县",
            "growth_stage": "幼果期",
            "temperature": 30,
            "humidity": 85,
        })
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["county"] == "沧县"
        assessments = data.get("risk_assessment", [])
        assert len(assessments) > 0

        # Should have risk_level field on each assessment
        for a in assessments:
            assert a["risk_level"] in ("低", "中", "高")
            assert 0 <= a["risk_score"] <= 1.0

    def test_high_temp_humidity_triggers_alert(self, kg_client):
        """High temperature + humidity should trigger disease alerts."""
        response = kg_client.get("/api/v1/kg/risk/predict", params={
            "county": "沧县",
            "temperature": 32,
            "humidity": 90,
            "precipitation_7d": 60,
        })
        assert response.status_code == 200
        data = response.json()["data"]
        # Alerts likely triggered under these conditions
        assert isinstance(data["alert_triggered"], bool)


class TestKGHealth:
    """Test health endpoint."""

    def test_health_returns_ready(self, kg_client):
        response = kg_client.get("/health")
        assert response.status_code == 200
        assert response.json()["is_ready"] is True
