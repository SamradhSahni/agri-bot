import os
import sys
import time
import json
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")
from utils.config_loader import CONFIG

# ── Config ────────────────────────────────────────────────────────────
TOP_K              = CONFIG["rag"]["top_k"]          # 3
MIN_RRF_THRESHOLD  = 0.0160    # drop passages below this RRF score
MIN_ANSWER_LEN     = 20       # drop passages with very short answers
MAX_CONTEXT_CHARS  = 600      # max total context chars fed to model


# ── Intent detector (reuse from inference.py) ─────────────────────────
def detect_intent_from_query(query: str) -> str:
    intents     = CONFIG["intents"]
    query_lower = query.lower()
    for intent_name, kw_banks in intents.items():
        for lang, words in kw_banks.items():
            for word in words:
                if word.lower() in query_lower:
                    return intent_name
    return "unknown"


# ── Passage quality filter ────────────────────────────────────────────
def filter_passages(passages: list) -> list:
    """
    Remove low-quality passages before feeding to the model.
    Filters:
    1. RRF score below threshold (poor match)
    2. Answer too short to be useful
    3. Answer is the same as a previous passage (dedup)
    """
    filtered  = []
    seen_answers = set()

    for p in passages:
        answer    = str(p.get("answer", "")).strip()
        rrf_score = p.get("rrf_score", 0.0)

        # Filter 1: RRF score threshold
        if rrf_score < MIN_RRF_THRESHOLD:
            logger.debug(f"Dropping passage — low RRF: {rrf_score:.5f}")
            continue

        # Filter 2: Answer too short
        if len(answer) < MIN_ANSWER_LEN:
            logger.debug(f"Dropping passage — answer too short: {len(answer)} chars")
            continue

        # Filter 3: Duplicate answer
        answer_key = answer[:80]    # use first 80 chars as dedup key
        if answer_key in seen_answers:
            logger.debug("Dropping passage — duplicate answer")
            continue

        seen_answers.add(answer_key)
        filtered.append(p)

    return filtered


# ── Context builder ───────────────────────────────────────────────────
def build_rag_context(passages: list, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """
    Build a clean Hindi context string from filtered passages.
    Truncates to max_chars to avoid overflowing model input.
    """
    if not passages:
        return ""

    lines  = []
    total  = 0

    for i, p in enumerate(passages, 1):
        answer = str(p.get("answer", "")).strip()
        intent = str(p.get("intent", "")).strip()

        # Clean up answer — remove excessive whitespace
        answer = " ".join(answer.split())

        line = f"{i}. {answer}"

        if total + len(line) > max_chars:
            # Truncate this passage to fit
            remaining = max_chars - total - 4
            if remaining > 50:
                line = f"{i}. {answer[:remaining]}..."
                lines.append(line)
            break

        lines.append(line)
        total += len(line)

    return "\n".join(lines)


# ── KisanMitraRAGPipeline ─────────────────────────────────────────────
class KisanMitraRAGPipeline:
    """
    Full RAG pipeline combining:
    1. Intent detection
    2. Hybrid retrieval (BM25 + vector)
    3. Passage filtering
    4. Context building
    5. Model generation

    This is the main class the FastAPI backend calls for /chat endpoint.

    Usage:
        pipeline = KisanMitraRAGPipeline()
        pipeline.load()
        result = pipeline.chat(
            query="गेहूं में कीट नियंत्रण कैसे करें?",
            state="HARYANA",
            crop="wheat",
            session_id="user123",
        )
    """

    def __init__(self):
        self.inference_engine = None
        self.retriever        = None
        self._ready           = False

    def load(self):
        """
        Load both inference engine and retriever.
        Called once at FastAPI startup.
        """
        if self._ready:
            logger.info("RAG pipeline already loaded")
            return

        logger.info("Loading KisanMitra RAG Pipeline...")

        # Load inference engine
        from backend.inference import KisanMitraInference
        self.inference_engine = KisanMitraInference()
        self.inference_engine.load()

        # Load retriever
        from backend.rag.retriever import KisanMitraRetriever
        self.retriever = KisanMitraRetriever()
        self.retriever.connect()

        # Pre-warm embedding model with dummy query
        logger.info("Pre-warming embedding model...")
        self.retriever.retrieve("गेहूं की बुवाई", top_k=1)
        logger.success("Embedding model warmed up")

        self._ready = True
        logger.success("KisanMitra RAG Pipeline ready")

    def chat(
        self,
        query:      str,
        state:      str  = "UTTAR PRADESH",
        crop:       str  = "others",
        intent:     str  = None,
        session_id: str  = None,
        use_rag:    bool = True,
    ) -> dict:
        """
        Main chat method — full RAG pipeline.

        Args:
            query      : Hindi farmer question
            state      : farmer's state (English uppercase)
            crop       : crop name
            intent     : override intent (optional — auto-detected if None)
            session_id : for logging/tracking
            use_rag    : set False to bypass RAG (useful for testing)

        Returns:
            dict with:
                response      : Hindi advisory text
                intent        : detected intent
                passages      : list of retrieved passages (for frontend sources panel)
                rag_used      : whether RAG was applied
                latency_ms    : total response time in milliseconds
                retrieval_ms  : retrieval time in milliseconds
                generation_ms : generation time in milliseconds
        """
        if not self._ready:
            raise RuntimeError("Pipeline not loaded. Call pipeline.load() first.")

        total_start = time.time()

        # ── Step 1: Detect intent ──────────────────────────────────────
        if intent is None:
            intent = detect_intent_from_query(query)

        logger.debug(f"Query: '{query[:60]}' | Intent: {intent} | State: {state}")

        # ── Step 2: Retrieve passages (RAG) ───────────────────────────
        passages        = []
        rag_context     = None
        retrieval_ms    = 0

        if use_rag and self.retriever.is_ready():
            retrieval_start = time.time()

            raw_passages = self.retriever.retrieve(
                query=query,
                top_k=TOP_K,
                intent=intent,
                state=state,
            )

            # Filter low-quality passages
            passages = filter_passages(raw_passages)

            # Build context string
            if passages:
                rag_context = build_rag_context(passages)

            retrieval_ms = int((time.time() - retrieval_start) * 1000)
            logger.debug(
                f"Retrieved {len(raw_passages)} passages → "
                f"{len(passages)} after filtering | {retrieval_ms}ms"
            )

        # ── Step 3: Generate response ──────────────────────────────────
        generation_start = time.time()

        result = self.inference_engine.generate(
            query=query,
            state=state,
            crop=crop,
            intent=intent,
            rag_context=rag_context,
        )

        generation_ms = int((time.time() - generation_start) * 1000)
        total_ms      = int((time.time() - total_start) * 1000)

        # ── Step 4: Build response payload ────────────────────────────
        response_payload = {
            "response":       result["response"],
            "intent":         intent,
            "rag_used":       rag_context is not None,
            "passages":       [
                {
                    "answer":    p.get("answer", "")[:300],
                    "intent":    p.get("intent", ""),
                    "crop":      p.get("crop", ""),
                    "state":     p.get("state", ""),
                    "rrf_score": round(p.get("rrf_score", 0), 5),
                }
                for p in passages
            ],
            "latency_ms":     total_ms,
            "retrieval_ms":   retrieval_ms,
            "generation_ms":  generation_ms,
            "session_id":     session_id,
            "query":          query,
            "state":          state,
            "crop":           crop,
        }

        logger.info(
            f"Chat complete | Intent: {intent} | "
            f"RAG: {rag_context is not None} | "
            f"Passages: {len(passages)} | "
            f"Total: {total_ms}ms "
            f"(retrieval: {retrieval_ms}ms, gen: {generation_ms}ms)"
        )

        return response_payload

    def is_ready(self) -> bool:
        return self._ready


# ── Module-level singleton ────────────────────────────────────────────
_pipeline: KisanMitraRAGPipeline = None


def get_pipeline() -> KisanMitraRAGPipeline:
    """
    Return the singleton RAG pipeline.
    Loads on first call — reuses on all subsequent calls.
    FastAPI lifespan should call this at startup.
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = KisanMitraRAGPipeline()
        _pipeline.load()
    return _pipeline