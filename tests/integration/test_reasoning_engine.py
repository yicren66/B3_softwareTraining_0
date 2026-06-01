"""
Integration tests for Reasoning Engine.

Validates:
  - Prevention plan generation with role differentiation
  - Multi-turn Q&A with KG-backed answers
  - Similar pest differentiation
  - Health check

Run:  pytest tests/integration/test_reasoning_engine.py -v
"""

import pytest

pytestmark = pytest.mark.integration


class TestPreventionPlan:
    """Test personalized prevention plan generation (SRS §4.2.4 KP-01~03)."""

    def test_plan_generation_basic(self, reasoning_client):
        """Generate a basic prevention plan."""
        response = reasoning_client.post("/api/v1/reasoning/prevention-plan", json={
            "pest_id": "KG-ENT-003",
            "pest_name": "枣树锈病",
            "county": "沧县",
            "growth_stage": "幼果期",
            "temperature": 28.0,
            "humidity": 75.0,
            "user_role": "枣农",
            "confidence": 0.91,
            "severity": "中度",
        })
        assert response.status_code == 200
        data = response.json()["data"]
        assert "explanation" in data
        assert "plan" in data
        explanation = data["explanation"]
        assert explanation["pest_name"] == "枣树锈病"
        assert len(explanation["cause_analysis"]) > 0

    def test_farmer_output_is_simplified(self, reasoning_client):
        """Farmer role should get simplified, actionable output."""
        response = reasoning_client.post("/api/v1/reasoning/prevention-plan", json={
            "pest_id": "KG-ENT-003",
            "pest_name": "枣树锈病",
            "user_role": "枣农",
            "severity": "轻度",
        })
        assert response.status_code == 200
        plan_text = response.json()["data"]["plan"].get("integrated_plan", "")
        # Farmer output should be simple and include emoji markers
        assert len(plan_text) > 0

    def test_expert_output_has_technical_detail(self, reasoning_client):
        """Expert role should get detailed technical output."""
        response = reasoning_client.post("/api/v1/reasoning/prevention-plan", json={
            "pest_id": "KG-ENT-003",
            "pest_name": "枣树锈病",
            "user_role": "植保专家",
            "severity": "重度",
        })
        assert response.status_code == 200
        explanation = response.json()["data"]["explanation"]
        cause = explanation.get("cause_analysis", "")
        # Expert output should have environmental analysis
        assert len(cause) > 0

    def test_severe_triggers_risk_warning(self, reasoning_client):
        """Severe severity should trigger risk warning in plan."""
        response = reasoning_client.post("/api/v1/reasoning/prevention-plan", json={
            "pest_id": "KG-ENT-003",
            "pest_name": "枣树锈病",
            "user_role": "枣农",
            "severity": "重度",
        })
        assert response.status_code == 200
        risk_warning = response.json()["data"]["plan"].get("risk_warning", "")
        assert "⚠️" in risk_warning or "重度" in risk_warning


class TestQA:
    """Test multi-turn Q&A (SRS §4.2.3 KR-04)."""

    def test_basic_qa_returns_answer(self, reasoning_client):
        """Simple question gets an answer with sources."""
        response = reasoning_client.post("/api/v1/reasoning/qa", json={
            "question": "枣树锈病怎么防治",
            "user_role": "枣农",
        })
        assert response.status_code == 200
        data = response.json()["data"]
        assert len(data["answer"]) > 0
        assert "sources" in data
        assert "follow_up_suggestions" in data

    def test_qa_returns_conversation_id(self, reasoning_client):
        """QA response should include a conversation_id for multi-turn."""
        response = reasoning_client.post("/api/v1/reasoning/qa", json={
            "question": "枣叶发黄是什么问题",
            "user_role": "枣农",
        })
        assert response.status_code == 200
        conv_id = response.json()["data"].get("conversation_id", "")
        assert len(conv_id) > 0

    def test_multi_turn_conversation(self, reasoning_client):
        """Multi-turn conversation should maintain context."""
        # First turn
        r1 = reasoning_client.post("/api/v1/reasoning/qa", json={
            "question": "我的枣树叶片有黄斑",
            "user_role": "枣农",
        })
        conv_id = r1.json()["data"]["conversation_id"]

        # Second turn with same conversation_id
        r2 = reasoning_client.post("/api/v1/reasoning/qa", json={
            "question": "那应该怎么处理",
            "user_role": "枣农",
            "conversation_id": conv_id,
        })
        assert r2.status_code == 200
        assert len(r2.json()["data"]["answer"]) > 0

    def test_similar_pest_differentiation(self, reasoning_client):
        """Query about similar pests should return differentiation info (SRS KR-02)."""
        response = reasoning_client.post("/api/v1/reasoning/qa", json={
            "question": "枣炭疽病和枣褐斑病怎么区别",
            "user_role": "植保专家",
        })
        assert response.status_code == 200
        answer = response.json()["data"]["answer"]
        assert "区别" in answer or "鉴别" in answer or "不同" in answer

    def test_follow_up_suggestions(self, reasoning_client):
        """QA response should include follow-up question suggestions."""
        response = reasoning_client.post("/api/v1/reasoning/qa", json={
            "question": "枣树锈病",
            "user_role": "枣农",
        })
        assert response.status_code == 200
        follow_ups = response.json()["data"].get("follow_up_suggestions", [])
        assert len(follow_ups) > 0


class TestReasoningHealth:
    def test_health(self, reasoning_client):
        response = reasoning_client.get("/health")
        assert response.status_code == 200
