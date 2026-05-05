import pytest
import time


class TestFullCycleIntegration:
    """
    End-to-end integration tests simulating a real farmer session.
    Tests the complete flow: chat → feedback → MSP lookup.
    """

    def test_full_farmer_session(self, test_client):
        """
        Simulate a complete farmer session:
        1. Ask a crop advisory question
        2. Ask a follow-up pest question
        3. Look up MSP price
        4. Submit feedback
        """
        session_id = "integration_test_full_session"

        # ── Step 1: Crop advisory ──
        r1 = test_client.post("/api/v1/chat", json={
            "query":      "गेहूं की बुवाई का सही समय क्या है?",
            "state":      "HARYANA",
            "crop":       "wheat",
            "session_id": session_id,
            "use_rag":    True,
        })
        assert r1.status_code == 200
        d1 = r1.json()
        assert len(d1["response"]) > 10
        first_response = d1["response"]

        # ── Step 2: Pest query ──
        r2 = test_client.post("/api/v1/chat", json={
            "query":      "गेहूं में माहू कीट का नियंत्रण कैसे करें?",
            "state":      "HARYANA",
            "crop":       "wheat",
            "session_id": session_id,
            "use_rag":    True,
        })
        assert r2.status_code == 200
        d2 = r2.json()
        assert len(d2["response"]) > 10

        # ── Step 3: MSP lookup ──
        r3 = test_client.get("/api/v1/msp", params={"crop": "wheat"})
        assert r3.status_code == 200
        d3 = r3.json()
        assert d3["found"] is True
        assert d3["msp_price"] == 2275

        # ── Step 4: Feedback on first response ──
        r4 = test_client.post("/api/v1/feedback", json={
            "session_id": session_id,
            "query":      "गेहूं की बुवाई का सही समय क्या है?",
            "response":   first_response,
            "rating":     4,
            "comment":    "उपयोगी जानकारी",
            "intent":     d1["intent"],
            "state":      "HARYANA",
            "crop":       "wheat",
        })
        assert r4.status_code == 200
        assert r4.json()["success"] is True

    def test_rag_vs_no_rag_both_succeed(self, test_client):
        """Both RAG and non-RAG should return valid Hindi responses."""
        query = {
            "query": "मक्का में फॉल आर्मी वर्म कीट का नियंत्रण?",
            "state": "UTTAR PRADESH",
            "crop":  "maize (makka)",
        }

        # With RAG
        r_rag = test_client.post("/api/v1/chat",
                                  json={**query, "use_rag": True})
        assert r_rag.status_code == 200
        resp_rag = r_rag.json()["response"]

        # Without RAG
        r_no = test_client.post("/api/v1/chat",
                                 json={**query, "use_rag": False})
        assert r_no.status_code == 200
        resp_no = r_no.json()["response"]

        # Both should be in Hindi
        for resp in [resp_rag, resp_no]:
            dev = sum(1 for c in resp if '\u0900' <= c <= '\u097F')
            tot = sum(1 for c in resp if c.isalpha())
            assert dev / max(tot, 1) >= 0.3

    def test_multiple_states_same_query(self, test_client):
        """Same query from different states should return valid responses."""
        states = [
            "UTTAR PRADESH", "BIHAR", "RAJASTHAN",
            "HARYANA", "MADHYA PRADESH",
        ]
        for state in states:
            r = test_client.post("/api/v1/chat", json={
                "query": "गेहूं की बुवाई कब करें?",
                "state": state,
                "crop":  "wheat",
            })
            assert r.status_code == 200, f"Failed for state: {state}"
            data = r.json()
            assert len(data["response"]) > 0

    def test_all_msp_crops_return_200(self, test_client):
        """All major crops should return 200 from MSP endpoint."""
        crops = [
            "wheat", "paddy", "maize", "mustard",
            "soybean", "cotton", "groundnut", "arhar",
            "moong", "bajra", "barley", "gram",
        ]
        for crop in crops:
            r = test_client.get("/api/v1/msp", params={"crop": crop})
            assert r.status_code == 200, f"MSP failed for: {crop}"
            assert r.json()["found"] is True, f"Not found: {crop}"

    def test_all_ratings_stored_successfully(self, test_client):
        """Submit one feedback for each rating value."""
        base = {
            "session_id": "rating_test_session",
            "query":      "गेहूं में कीट है",
            "response":   "नीम तेल डालें",
        }
        for rating in [1, 2, 3, 4, 5]:
            r = test_client.post("/api/v1/feedback",
                                  json={**base, "rating": rating})
            assert r.status_code == 200
            assert r.json()["success"] is True

    def test_concurrent_chat_requests(self, test_client):
        """
        Simulate multiple farmers asking different questions.
        All should succeed (tests thread safety of pipeline).
        """
        import threading

        queries = [
            {"query": "गेहूं में कीट है?",         "state": "HARYANA",       "crop": "wheat"},
            {"query": "धान में रोग लगा है?",       "state": "BIHAR",         "crop": "paddy (dhan)"},
            {"query": "किसान क्रेडिट कार्ड?",      "state": "UTTAR PRADESH", "crop": "others"},
            {"query": "सरसों की बुवाई कब करें?",  "state": "RAJASTHAN",     "crop": "mustard"},
        ]

        results = []
        errors  = []

        def make_request(q):
            try:
                r = test_client.post("/api/v1/chat", json=q, timeout=30)
                results.append(r.status_code)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=make_request, args=(q,)) for q in queries]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert all(s == 200 for s in results), f"Non-200 responses: {results}"