import uuid
from fastapi import APIRouter, HTTPException, Request
from loguru import logger
from backend.api.models.schemas import ChatRequest, ChatResponse, PassageItem
from backend.db.database import upsert_session, save_message
from backend.cache.redis_cache import set_session_context, refresh_session_ttl
from datetime import datetime

router = APIRouter()


@router.post("/chat", response_model=ChatResponse, summary="Chat with KisanMitra AI")
async def chat(request: Request, body: ChatRequest):
    pipeline   = request.app.state.pipeline
    if not pipeline.is_ready():
        raise HTTPException(status_code=503, detail="Model pipeline not ready.")

    session_id = body.session_id or str(uuid.uuid4())

    try:
        result = pipeline.chat(
            query=body.query,
            state=body.state,
            crop=body.crop,
            intent=body.intent,
            session_id=session_id,
            use_rag=body.use_rag,
        )
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    # ── Save to DB (non-blocking best-effort) ──
    try:
        upsert_session(session_id, body.state, body.crop)
        save_message(session_id, "user",      body.query,        result["intent"])
        save_message(session_id, "assistant", result["response"],
                     intent=result["intent"],
                     rag_used=result["rag_used"],
                     passages_count=len(result.get("passages", [])),
                     latency_ms=result["latency_ms"])

        # Cache session context in Redis
        set_session_context(session_id, {
            "state":  body.state,
            "crop":   body.crop,
            "intent": result["intent"],
        })
    except Exception as e:
        logger.warning(f"DB/cache save warning (non-fatal): {e}")

    passages = [
        PassageItem(
            answer=p["answer"], intent=p["intent"],
            crop=p["crop"],     state=p["state"],
            rrf_score=p["rrf_score"],
        )
        for p in result.get("passages", [])
    ]

    return ChatResponse(
        response=result["response"],
        intent=result["intent"],
        rag_used=result["rag_used"],
        passages=passages,
        latency_ms=result["latency_ms"],
        retrieval_ms=result["retrieval_ms"],
        generation_ms=result["generation_ms"],
        session_id=session_id,
        query=body.query,
        state=body.state,
        crop=body.crop,
        timestamp=datetime.utcnow(),
    )