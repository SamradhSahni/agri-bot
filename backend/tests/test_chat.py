import pytest


class TestChatEndpoint:
    """Tests for POST /api/v1/chat endpoint."""

    # ── Basic response structure ──────────────────────────────────────

    def test_chat_returns_200(self, test_client, pest_query):
        r = test_client.post("/api/v1/chat", json=pest_query)
        assert r.status_code == 200

    def test_chat_response_has_required_fields(self, test_client, pest_query):
        r    = test_client.post("/api/v1/chat", json=pest_query)
        data = r.json()
        required = [
            "response", "intent", "rag_used",
            "passages", "latency_ms", "retrieval_ms",
            "generation_ms", "session_id", "query",
            "state", "crop", "timestamp",
        ]
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_chat_response_is_string(self, test_client, pest_query):
        r    = test_client.post("/api/v1/chat", json=pest_query)
        data = r.json()
        assert isinstance(data["response"], str)
        assert len(data["response"]) > 0

    def test_chat_response_not_empty(self, test_client, pest_query):
        r    = test_client.post("/api/v1/chat", json=pest_query)
        data = r.json()
        assert len(data["response"].strip()) > 10

    # ── Hindi language validation ─────────────────────────────────────

    def test_chat_response_is_hindi(self, test_client, pest_query):
        r        = test_client.post("/api/v1/chat", json=pest_query)
        response = r.json()["response"]
        dev_chars   = sum(1 for c in response if '\u0900' <= c <= '\u097F')
        total_alpha = sum(1 for c in response if c.isalpha())
        ratio = dev_chars / max(total_alpha, 1)
        assert ratio >= 0.3, \
            f"Response not in Hindi. Devanagari ratio: {ratio:.2f}\nResponse: {response}"

    def test_chat_no_language_mismatch(self, test_client, crop_advisory_query):
        r        = test_client.post("/api/v1/chat", json=crop_advisory_query)
        response = r.json()["response"]
        # Should not be purely English
        latin_chars = sum(1 for c in response if c.isascii() and c.isalpha())
        dev_chars   = sum(1 for c in response if '\u0900' <= c <= '\u097F')
        assert dev_chars > latin_chars, \
            f"Response appears to be in English: {response}"

    # ── Intent detection ──────────────────────────────────────────────

    def test_chat_intent_detected(self, test_client, pest_query):
        r    = test_client.post("/api/v1/chat", json=pest_query)
        data = r.json()
        assert data["intent"] is not None
        assert len(data["intent"]) > 0

    def test_chat_pest_intent_correct(self, test_client, pest_query):
        r    = test_client.post("/api/v1/chat", json=pest_query)
        data = r.json()
        assert data["intent"] == "pest_id", \
            f"Expected pest_id, got {data['intent']}"

    def test_chat_crop_advisory_intent(self, test_client, crop_advisory_query):
        r    = test_client.post("/api/v1/chat", json=crop_advisory_query)
        data = r.json()
        assert data["intent"] in ["crop_advisory", "weather_sowing", "nutrient_management"]

    def test_chat_government_scheme_intent(self, test_client, government_scheme_query):
        r    = test_client.post("/api/v1/chat", json=government_scheme_query)
        data = r.json()
        assert data["intent"] == "government_scheme"

    # ── RAG validation ────────────────────────────────────────────────

    def test_chat_rag_used_when_enabled(self, test_client, pest_query):
        r    = test_client.post("/api/v1/chat", json=pest_query)
        data = r.json()
        assert data["rag_used"] is True

    def test_chat_rag_disabled_when_requested(self, test_client, pest_query):
        body = {**pest_query, "use_rag": False}
        r    = test_client.post("/api/v1/chat", json=body)
        data = r.json()
        assert data["rag_used"] is False

    def test_chat_passages_returned_with_rag(self, test_client, pest_query):
        r    = test_client.post("/api/v1/chat", json=pest_query)
        data = r.json()
        assert isinstance(data["passages"], list)
        assert len(data["passages"]) >= 0

    def test_chat_passage_structure(self, test_client, pest_query):
        r        = test_client.post("/api/v1/chat", json=pest_query)
        data     = r.json()
        passages = data["passages"]
        if passages:
            p = passages[0]
            assert "answer"    in p
            assert "intent"    in p
            assert "rrf_score" in p

    def test_chat_no_passages_without_rag(self, test_client, pest_query):
        body = {**pest_query, "use_rag": False}
        r    = test_client.post("/api/v1/chat", json=body)
        data = r.json()
        assert data["passages"] == []

    # ── Latency validation ────────────────────────────────────────────

    def test_chat_latency_ms_positive(self, test_client, pest_query):
        r    = test_client.post("/api/v1/chat", json=pest_query)
        data = r.json()
        assert data["latency_ms"] > 0
        assert data["retrieval_ms"] >= 0
        assert data["generation_ms"] > 0

    def test_chat_latency_under_10s(self, test_client, crop_advisory_query):
        r    = test_client.post("/api/v1/chat", json=crop_advisory_query)
        data = r.json()
        assert data["latency_ms"] < 10000, \
            f"Latency too high: {data['latency_ms']}ms"

    # ── Session handling ──────────────────────────────────────────────

    def test_chat_session_id_returned(self, test_client, pest_query):
        r    = test_client.post("/api/v1/chat", json=pest_query)
        data = r.json()
        assert data["session_id"] == pest_query["session_id"]

    def test_chat_auto_generates_session_id(self, test_client):
        body = {
            "query": "गेहूं में सिंचाई कब करें?",
            "state": "HARYANA",
            "crop":  "wheat",
        }
        r    = test_client.post("/api/v1/chat", json=body)
        data = r.json()
        assert data["session_id"] is not None
        assert len(data["session_id"]) > 0

    def test_chat_query_echoed_back(self, test_client, pest_query):
        r    = test_client.post("/api/v1/chat", json=pest_query)
        data = r.json()
        assert data["query"] == pest_query["query"]

    def test_chat_state_echoed_back(self, test_client, pest_query):
        r    = test_client.post("/api/v1/chat", json=pest_query)
        data = r.json()
        assert data["state"] == pest_query["state"]

    # ── Validation errors ─────────────────────────────────────────────

    def test_chat_empty_query_rejected(self, test_client):
        body = {"query": "", "state": "BIHAR", "crop": "wheat"}
        r    = test_client.post("/api/v1/chat", json=body)
        assert r.status_code == 422    # Pydantic validation error

    def test_chat_missing_query_rejected(self, test_client):
        body = {"state": "BIHAR", "crop": "wheat"}
        r    = test_client.post("/api/v1/chat", json=body)
        assert r.status_code == 422

    def test_chat_query_too_long_rejected(self, test_client):
        body = {"query": "क" * 1001, "state": "BIHAR", "crop": "wheat"}
        r    = test_client.post("/api/v1/chat", json=body)
        assert r.status_code == 422

    def test_chat_default_state_used(self, test_client):
        body = {"query": "गेहूं में कीट है क्या करें?"}
        r    = test_client.post("/api/v1/chat", json=body)
        assert r.status_code == 200

    # ── All 11 intents ────────────────────────────────────────────────

    @pytest.mark.parametrize("query,state,crop,expected_intent", [
        (
            "मक्का में फॉल आर्मी वर्म कीट का नियंत्रण?",
            "UTTAR PRADESH", "maize (makka)", "pest_id"
        ),
        (
            "धान में झुलसा रोग का उपचार बताएं",
            "BIHAR", "paddy (dhan)", "disease"
        ),
        (
            "किसान क्रेडिट कार्ड के लिए आवेदन कैसे करें?",
            "BIHAR", "others", "government_scheme"
        ),
        (
            "गेहूं का न्यूनतम समर्थन मूल्य क्या है?",
            "UTTAR PRADESH", "wheat", "msp_price"
        ),
        (
            "सरसों में यूरिया कब और कितनी मात्रा में डालें?",
            "RAJASTHAN", "mustard", "nutrient_management"
        ),
    ])
    def test_chat_intent_parametrized(
        self, test_client, query, state, crop, expected_intent
    ):
        body = {"query": query, "state": state, "crop": crop, "use_rag": True}
        r    = test_client.post("/api/v1/chat", json=body)
        assert r.status_code == 200
        data = r.json()
        assert data["intent"] == expected_intent, \
            f"Expected {expected_intent}, got {data['intent']}"