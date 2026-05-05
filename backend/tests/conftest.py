import sys
import pytest
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")

# ── Shared test fixtures ──────────────────────────────────────────────

@pytest.fixture(scope="session")
def test_client():
    """
    Create a TestClient with the full FastAPI app.
    Loads the RAG pipeline once for the entire test session.
    scope="session" means model loads once — not per test.
    """
    from fastapi.testclient import TestClient
    from backend.main import app
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="session")
def base_url():
    return "http://testserver"


# ── Sample request bodies ─────────────────────────────────────────────

@pytest.fixture
def pest_query():
    return {
        "query":      "मक्का में फॉल आर्मी वर्म कीट का नियंत्रण कैसे करें?",
        "state":      "UTTAR PRADESH",
        "crop":       "maize (makka)",
        "session_id": "pytest_session_001",
        "use_rag":    True,
    }


@pytest.fixture
def crop_advisory_query():
    return {
        "query":      "गेहूं की बुवाई का सही समय और बीज दर क्या है?",
        "state":      "HARYANA",
        "crop":       "wheat",
        "session_id": "pytest_session_002",
        "use_rag":    True,
    }


@pytest.fixture
def government_scheme_query():
    return {
        "query":      "किसान क्रेडिट कार्ड के लिए आवेदन कैसे करें?",
        "state":      "BIHAR",
        "crop":       "others",
        "session_id": "pytest_session_003",
        "use_rag":    True,
    }


@pytest.fixture
def feedback_body():
    return {
        "session_id": "pytest_session_001",
        "query":      "मक्का में कीट नियंत्रण",
        "response":   "नीम का तेल 5 मिलीलीटर प्रति लीटर पानी में छिड़काव करें",
        "rating":     4,
        "comment":    "अच्छी जानकारी",
        "intent":     "pest_id",
        "state":      "UTTAR PRADESH",
        "crop":       "maize (makka)",
    }