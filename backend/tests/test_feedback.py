import pytest


class TestFeedbackEndpoint:
    """Tests for POST /api/v1/feedback endpoint."""

    # ── Basic response ────────────────────────────────────────────────

    def test_feedback_returns_200(self, test_client, feedback_body):
        r = test_client.post("/api/v1/feedback", json=feedback_body)
        assert r.status_code == 200

    def test_feedback_response_structure(self, test_client, feedback_body):
        r    = test_client.post("/api/v1/feedback", json=feedback_body)
        data = r.json()
        assert "success"     in data
        assert "message"     in data
        assert "feedback_id" in data

    def test_feedback_success_true(self, test_client, feedback_body):
        r    = test_client.post("/api/v1/feedback", json=feedback_body)
        data = r.json()
        assert data["success"] is True

    def test_feedback_message_in_hindi(self, test_client, feedback_body):
        r       = test_client.post("/api/v1/feedback", json=feedback_body)
        message = r.json()["message"]
        dev_chars = sum(1 for c in message if '\u0900' <= c <= '\u097F')
        assert dev_chars > 0, f"Message not in Hindi: {message}"

    def test_feedback_id_is_integer(self, test_client, feedback_body):
        r    = test_client.post("/api/v1/feedback", json=feedback_body)
        data = r.json()
        assert isinstance(data["feedback_id"], int)
        assert data["feedback_id"] > 0

    def test_feedback_ids_increment(self, test_client, feedback_body):
        r1 = test_client.post("/api/v1/feedback", json=feedback_body)
        r2 = test_client.post("/api/v1/feedback", json=feedback_body)
        id1 = r1.json()["feedback_id"]
        id2 = r2.json()["feedback_id"]
        assert id2 > id1, f"IDs not incrementing: {id1}, {id2}"

    # ── Rating validation ─────────────────────────────────────────────

    @pytest.mark.parametrize("rating", [1, 2, 3, 4, 5])
    def test_feedback_valid_ratings(self, test_client, feedback_body, rating):
        body = {**feedback_body, "rating": rating}
        r    = test_client.post("/api/v1/feedback", json=body)
        assert r.status_code == 200
        assert r.json()["success"] is True

    def test_feedback_rating_zero_rejected(self, test_client, feedback_body):
        body = {**feedback_body, "rating": 0}
        r    = test_client.post("/api/v1/feedback", json=body)
        assert r.status_code == 422

    def test_feedback_rating_six_rejected(self, test_client, feedback_body):
        body = {**feedback_body, "rating": 6}
        r    = test_client.post("/api/v1/feedback", json=body)
        assert r.status_code == 422

    def test_feedback_negative_rating_rejected(self, test_client, feedback_body):
        body = {**feedback_body, "rating": -1}
        r    = test_client.post("/api/v1/feedback", json=body)
        assert r.status_code == 422

    # ── Optional fields ───────────────────────────────────────────────

    def test_feedback_without_comment(self, test_client, feedback_body):
        body = {**feedback_body}
        body.pop("comment", None)
        r    = test_client.post("/api/v1/feedback", json=body)
        assert r.status_code == 200

    def test_feedback_without_intent(self, test_client, feedback_body):
        body = {**feedback_body}
        body.pop("intent", None)
        r    = test_client.post("/api/v1/feedback", json=body)
        assert r.status_code == 200

    def test_feedback_comment_too_long_rejected(self, test_client, feedback_body):
        body = {**feedback_body, "comment": "क" * 501}
        r    = test_client.post("/api/v1/feedback", json=body)
        assert r.status_code == 422

    # ── Required fields ───────────────────────────────────────────────

    def test_feedback_missing_session_id_rejected(self, test_client, feedback_body):
        body = {**feedback_body}
        body.pop("session_id")
        r    = test_client.post("/api/v1/feedback", json=body)
        assert r.status_code == 422

    def test_feedback_missing_query_rejected(self, test_client, feedback_body):
        body = {**feedback_body}
        body.pop("query")
        r    = test_client.post("/api/v1/feedback", json=body)
        assert r.status_code == 422

    def test_feedback_missing_rating_rejected(self, test_client, feedback_body):
        body = {**feedback_body}
        body.pop("rating")
        r    = test_client.post("/api/v1/feedback", json=body)
        assert r.status_code == 422