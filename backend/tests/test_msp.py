import pytest


class TestMSPEndpoint:
    """Tests for GET /api/v1/msp endpoint."""

    # ── Basic response ────────────────────────────────────────────────

    def test_msp_returns_200(self, test_client):
        r = test_client.get("/api/v1/msp", params={"crop": "wheat"})
        assert r.status_code == 200

    def test_msp_response_structure(self, test_client):
        r    = test_client.get("/api/v1/msp", params={"crop": "wheat"})
        data = r.json()
        required = ["crop", "msp_price", "unit", "season", "year", "source", "found"]
        for field in required:
            assert field in data, f"Missing field: {field}"

    # ── Known crops ───────────────────────────────────────────────────

    @pytest.mark.parametrize("crop,expected_price", [
        ("wheat",    2275),
        ("paddy",    2300),
        ("mustard",  5650),
        ("cotton",   7121),
        ("maize",    2225),
        ("soybean",  4892),
        ("groundnut",6783),
        ("sugarcane", 340),
        ("arhar",    7550),
        ("moong",    8682),
        ("bajra",    2625),
        ("barley",   1735),
    ])
    def test_msp_known_crops(self, test_client, crop, expected_price):
        r    = test_client.get("/api/v1/msp", params={"crop": crop})
        data = r.json()
        assert data["found"] is True, f"Crop not found: {crop}"
        assert data["msp_price"] == expected_price, \
            f"{crop}: expected ₹{expected_price}, got ₹{data['msp_price']}"

    def test_msp_wheat_details(self, test_client):
        r    = test_client.get("/api/v1/msp", params={"crop": "wheat"})
        data = r.json()
        assert data["found"]     is True
        assert data["msp_price"] == 2275
        assert data["season"]    == "Rabi"
        assert data["year"]      == "2024-25"
        assert "quintal"         in data["unit"]

    # ── Unknown crop ──────────────────────────────────────────────────

    def test_msp_unknown_crop_not_found(self, test_client):
        r    = test_client.get("/api/v1/msp", params={"crop": "unknowncrop_xyz"})
        data = r.json()
        assert data["found"]     is False
        assert data["msp_price"] is None

    def test_msp_unknown_crop_returns_200(self, test_client):
        r = test_client.get("/api/v1/msp", params={"crop": "unknowncrop_xyz"})
        assert r.status_code == 200    # not a 404 — graceful not-found

    # ── Case insensitivity ────────────────────────────────────────────

    def test_msp_case_insensitive(self, test_client):
        r1 = test_client.get("/api/v1/msp", params={"crop": "WHEAT"})
        r2 = test_client.get("/api/v1/msp", params={"crop": "wheat"})
        r3 = test_client.get("/api/v1/msp", params={"crop": "Wheat"})
        assert r1.json()["msp_price"] == r2.json()["msp_price"] == r3.json()["msp_price"]

    # ── Missing param ─────────────────────────────────────────────────

    def test_msp_missing_crop_param(self, test_client):
        r = test_client.get("/api/v1/msp")
        assert r.status_code == 422    # crop is required

    # ── Source field ──────────────────────────────────────────────────

    def test_msp_source_is_government(self, test_client):
        r    = test_client.get("/api/v1/msp", params={"crop": "wheat"})
        data = r.json()
        assert "Government of India" in data["source"] or "CCEA" in data["source"]

    # ── Redis cache integration ───────────────────────────────────────

    def test_msp_second_call_is_faster(self, test_client):
        import time

        # First call — may hit DB
        start1 = time.time()
        test_client.get("/api/v1/msp", params={"crop": "mustard"})
        t1 = time.time() - start1

        # Second call — should hit Redis cache
        start2 = time.time()
        test_client.get("/api/v1/msp", params={"crop": "mustard"})
        t2 = time.time() - start2

        # Cache should be at least as fast (within 50ms)
        assert t2 <= t1 + 0.05, \
            f"Second call not faster: {t1*1000:.0f}ms vs {t2*1000:.0f}ms"