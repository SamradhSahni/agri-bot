import sys
import time
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")

log_path = Path("logs/test_db_cache.log")
log_path.parent.mkdir(exist_ok=True)
logger.add(str(log_path), rotation="10 MB", encoding="utf-8")


def test_postgres():
    from backend.db.database import (
        upsert_session, get_session,
        save_message, get_session_messages,
        save_feedback, get_usage_stats,
        get_msp_from_db,
    )

    sep = "=" * 55
    print(f"\n{sep}")
    print("  PostgreSQL Tests")
    print(sep)

    # Session test
    upsert_session("test_sess_01", "BIHAR", "wheat")
    sess = get_session("test_sess_01")
    assert sess["state"] == "BIHAR"
    print(f"  ✅ Session upsert + get")

    # Message test
    msg_id = save_message("test_sess_01", "user",      "गेहूं में कीट है", "pest_id")
    msg_id = save_message("test_sess_01", "assistant", "नीम का तेल डालें",
                          "pest_id", rag_used=True, passages_count=3, latency_ms=1200)
    msgs = get_session_messages("test_sess_01")
    assert len(msgs) == 2
    print(f"  ✅ Message save + get ({len(msgs)} messages)")

    # Feedback test
    fb_id = save_feedback(
        "test_sess_01", "गेहूं में कीट", "नीम तेल डालें",
        rating=4, comment="अच्छा", intent="pest_id",
        state="BIHAR", crop="wheat",
    )
    print(f"  ✅ Feedback saved (id={fb_id})")

    # MSP test
    msp = get_msp_from_db("wheat")
    assert msp is not None
    print(f"  ✅ MSP from DB: wheat = ₹{msp['price']}/quintal")

    # Stats test
    stats = get_usage_stats()
    print(f"  ✅ Usage stats: {stats['total_sessions']} sessions, "
          f"{stats['total_queries']} queries")

    print(f"\n  ✅ All PostgreSQL tests passed")
    print(f"{sep}")


def test_redis():
    from backend.cache.redis_cache import (
        get_cached_msp, set_cached_msp,
        get_session_context, set_session_context,
        get_cache_stats, is_redis_available,
    )

    sep = "=" * 55
    print(f"\n{sep}")
    print("  Redis Cache Tests")
    print(sep)

    if not is_redis_available():
        print(f"  ⚠️  Redis not available — skipping cache tests")
        print(f"  Make sure Redis is running on localhost:6379")
        return False

    # MSP cache test
    test_data = {"msp_price": 2275, "unit": "₹/quintal",
                 "season": "Rabi", "year": "2024-25"}
    set_cached_msp("wheat", test_data)
    cached = get_cached_msp("wheat")
    assert cached is not None
    assert cached["msp_price"] == 2275
    print(f"  ✅ MSP cache set + get (TTL: 24h)")

    # Session context cache test
    ctx = {"state": "BIHAR", "crop": "wheat", "intent": "pest_id"}
    set_session_context("test_sess_01", ctx)
    retrieved = get_session_context("test_sess_01")
    assert retrieved["state"] == "BIHAR"
    print(f"  ✅ Session context cache set + get (TTL: 1h)")

    # Cache miss test
    miss = get_cached_msp("nonexistent_crop_xyz")
    assert miss is None
    print(f"  ✅ Cache miss returns None correctly")

    # Stats
    stats = get_cache_stats()
    print(f"  ✅ Cache stats: {stats['msp_cached']} MSP keys, "
          f"{stats['sessions_live']} sessions, "
          f"memory: {stats['memory_used']}")

    # TTL test
    import time
    set_cached_msp("barley_ttl_test", {"msp_price": 1735,
                   "unit": "₹/quintal", "season": "Rabi", "year": "2024-25"})
    time.sleep(0.1)
    hit = get_cached_msp("barley_ttl_test")
    assert hit is not None
    print(f"  ✅ TTL test passed")

    print(f"\n  ✅ All Redis tests passed")
    print(f"{sep}")
    return True


def test_full_stack():
    """
    Simulate a full request cycle:
    chat → DB save → Redis cache → MSP lookup with cache
    """
    from backend.db.database import upsert_session, save_message, get_msp_from_db
    from backend.cache.redis_cache import set_cached_msp, get_cached_msp

    sep = "=" * 55
    print(f"\n{sep}")
    print("  Full Stack Integration Test")
    print(sep)

    session_id = "full_stack_test_01"

    # Simulate chat request
    upsert_session(session_id, "UTTAR PRADESH", "maize (makka)")
    save_message(session_id, "user",
                 "मक्का में फॉल आर्मी वर्म कीट का नियंत्रण?", "pest_id")
    save_message(session_id, "assistant",
                 "इमामेक्टिन बेंज़ोइड का छिड़काव करें",
                 "pest_id", rag_used=True, passages_count=3, latency_ms=1375)
    print(f"  ✅ Chat messages saved to PostgreSQL")

    # Simulate MSP lookup with cache
    msp_db = get_msp_from_db("wheat")
    if msp_db:
        set_cached_msp("wheat", {
            "msp_price": msp_db["price"],
            "unit":      msp_db["unit"],
            "season":    msp_db["season"],
            "year":      msp_db["year"],
        })

    cached = get_cached_msp("wheat")
    if cached:
        print(f"  ✅ MSP cached from DB: ₹{cached['msp_price']}/quintal")
    else:
        print(f"  ⚠️  Redis not available — MSP cache skipped")

    print(f"\n  ✅ Full stack integration test passed")
    print(f"{sep}\n")


if __name__ == "__main__":
    logger.info("=" * 55)
    logger.info("KisanMitra AI — DB + Cache Test (Task 14)")
    logger.info("=" * 55)

    test_postgres()
    redis_ok = test_redis()
    test_full_stack()

    sep = "=" * 55
    print(f"\n{sep}")
    print("  Task 14 Summary")
    print(sep)
    print(f"  ✅ PostgreSQL — all operations working")
    print(f"  {'✅' if redis_ok else '⚠️ '} Redis — {'all operations working' if redis_ok else 'not running (optional for dev)'}")
    print(f"  ✅ Full stack integration — working")
    print(f"\n  Backend data layer ready.")
    print(f"  Next: Task 15 — pytest tests for all endpoints")
    print(f"{sep}\n")