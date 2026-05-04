from fastapi import APIRouter, Query
from loguru import logger
from backend.api.models.schemas import MSPResponse
from backend.cache.redis_cache import get_cached_msp, set_cached_msp
from backend.db.database import get_msp_from_db

router = APIRouter()

# ── Fallback in-memory MSP data ───────────────────────────────────────
MSP_FALLBACK = {
    "wheat":       {"price": 2275, "unit": "₹/quintal", "season": "Rabi",   "year": "2024-25"},
    "paddy":       {"price": 2300, "unit": "₹/quintal", "season": "Kharif", "year": "2024-25"},
    "mustard":     {"price": 5650, "unit": "₹/quintal", "season": "Rabi",   "year": "2024-25"},
    "cotton":      {"price": 7121, "unit": "₹/quintal", "season": "Kharif", "year": "2024-25"},
    "maize":       {"price": 2225, "unit": "₹/quintal", "season": "Kharif", "year": "2024-25"},
    "soybean":     {"price": 4892, "unit": "₹/quintal", "season": "Kharif", "year": "2024-25"},
    "groundnut":   {"price": 6783, "unit": "₹/quintal", "season": "Kharif", "year": "2024-25"},
    "sugarcane":   {"price": 340,  "unit": "₹/quintal", "season": "Annual", "year": "2024-25"},
    "arhar":       {"price": 7550, "unit": "₹/quintal", "season": "Kharif", "year": "2024-25"},
    "moong":       {"price": 8682, "unit": "₹/quintal", "season": "Kharif", "year": "2024-25"},
    "bajra":       {"price": 2625, "unit": "₹/quintal", "season": "Kharif", "year": "2024-25"},
    "barley":      {"price": 1735, "unit": "₹/quintal", "season": "Rabi",   "year": "2024-25"},
    "gram":        {"price": 5440, "unit": "₹/quintal", "season": "Rabi",   "year": "2024-25"},
    "lentil":      {"price": 6425, "unit": "₹/quintal", "season": "Rabi",   "year": "2024-25"},
}


@router.get("/msp", response_model=MSPResponse, summary="Get MSP price for a crop")
async def get_msp(crop: str = Query(..., description="Crop name", example="wheat")):
    crop_key = crop.lower().strip()

    # ── Layer 1: Redis cache ──
    cached = get_cached_msp(crop_key)
    if cached:
        logger.info(f"MSP cache HIT: {crop_key}")
        return MSPResponse(found=True, crop=crop,
                           source="CCEA, Government of India (cached)", **cached)

    # ── Layer 2: PostgreSQL ──
    try:
        db_data = get_msp_from_db(crop_key)
        if db_data:
            set_cached_msp(crop_key, {
                "msp_price": db_data["price"],
                "unit":      db_data["unit"],
                "season":    db_data["season"],
                "year":      db_data["year"],
            })
            logger.info(f"MSP DB HIT: {crop_key}")
            return MSPResponse(
                crop=crop, found=True,
                msp_price=db_data["price"],
                unit=db_data["unit"],
                season=db_data["season"],
                year=db_data["year"],
                source="CCEA, Government of India",
            )
    except Exception as e:
        logger.warning(f"DB lookup failed, using fallback: {e}")

    # ── Layer 3: In-memory fallback ──
    data = MSP_FALLBACK.get(crop_key)
    if not data:
        # Fuzzy match
        for key, val in MSP_FALLBACK.items():
            if crop_key in key or key in crop_key:
                data = val
                break

    if data:
        set_cached_msp(crop_key, {"msp_price": data["price"], **data})
        return MSPResponse(
            crop=crop, found=True,
            msp_price=data["price"],
            unit=data["unit"],
            season=data["season"],
            year=data["year"],
            source="CCEA, Government of India",
        )

    return MSPResponse(
        crop=crop, found=False,
        msp_price=None, unit="₹/quintal",
        season="N/A", year="2024-25",
        source="CCEA, Government of India",
    )