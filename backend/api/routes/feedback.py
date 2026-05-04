from datetime import datetime
from fastapi import APIRouter, HTTPException
from loguru import logger
from backend.api.models.schemas import FeedbackRequest, FeedbackResponse
from backend.db.database import save_feedback

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse,
             summary="Submit feedback for a response")
async def submit_feedback(body: FeedbackRequest):
    try:
        feedback_id = save_feedback(
            session_id=body.session_id,
            query=body.query,
            response=body.response,
            rating=body.rating,
            comment=body.comment,
            intent=body.intent,
            state=body.state,
            crop=body.crop,
        )
        logger.info(f"Feedback #{feedback_id} saved | Rating: {body.rating}/5")
        return FeedbackResponse(
            success=True,
            message="फीडबैक सफलतापूर्वक प्राप्त हुआ। धन्यवाद!",
            feedback_id=feedback_id,
        )
    except Exception as e:
        logger.error(f"Feedback DB error: {e}")
        raise HTTPException(status_code=500, detail="Feedback save failed")