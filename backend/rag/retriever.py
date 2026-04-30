import os
import sys
import torch
import numpy as np
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()
sys.path.insert(0, ".")
from utils.config_loader import CONFIG

# ── Config ────────────────────────────────────────────────────────────
ES_HOST        = os.getenv("ES_HOST",       "http://localhost:9200")
ES_INDEX       = os.getenv("ES_INDEX_NAME", "kisanmitra_kb")
EMBEDDING_MODEL = CONFIG["rag"]["embedding_model"]
EMBEDDING_DIM   = CONFIG["rag"]["embedding_dim"]
TOP_K           = CONFIG["rag"]["top_k"]        # 3
BM25_WEIGHT     = CONFIG["rag"]["bm25_weight"]  # 2.0


# ── Singleton embedding model ─────────────────────────────────────────
_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _embed_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        logger.info(f"Embedding model loaded on {device}")
    return _embed_model


# ── KisanMitraRetriever ───────────────────────────────────────────────
class KisanMitraRetriever:
    """
    Hybrid retriever combining BM25 + cosine vector similarity.
    Used by the RAG pipeline at inference time.

    Usage:
        retriever = KisanMitraRetriever()
        passages  = retriever.retrieve("मक्का में कीट नियंत्रण कैसे करें?")
    """

    def __init__(self):
        self.es     = None
        self._ready = False

    def connect(self):
        self.es = Elasticsearch(
            ES_HOST,
            request_timeout=30,
        )
        if not self.es.ping():
            raise ConnectionError(
                f"Cannot connect to Elasticsearch at {ES_HOST}"
            )
        self._ready = True
        logger.success("KisanMitraRetriever connected to Elasticsearch")

    # ── Embed query ───────────────────────────────────────────────────
    def _embed_query(self, query: str) -> list:
        model = get_embed_model()
        # Prefix 'query: ' as required by multilingual-e5-small
        text  = f"query: {query}"
        emb   = model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return emb[0].tolist()

    # ── Vector search ──────────────────────────────────────────────────
    def _vector_search(self, query: str, top_k: int) -> list:
        """KNN cosine similarity search using dense_vector field."""

        q_emb = self._embed_query(query)

        result = self.es.search(
            index=ES_INDEX,
            body={
                "knn": {
                    "field":          "embedding",
                    "query_vector":   q_emb,
                    "k":              top_k * 2,
                    "num_candidates": top_k * 10,
                },
                "size": top_k * 2,
                "_source": ["query", "answer", "intent", "crop", "state", "doc_id"],
            }
        )

        hits = []
        for hit in result["hits"]["hits"]:
            hits.append({
                "doc_id":       hit["_source"].get("doc_id", -1),
                "query":        hit["_source"].get("query", ""),
                "answer":       hit["_source"].get("answer", ""),
                "intent":       hit["_source"].get("intent", ""),
                "crop":         hit["_source"].get("crop", ""),
                "state":        hit["_source"].get("state", ""),
                "bm25_score":   0.0,
                "vector_score": hit["_score"],
            })
        return hits

    # ── BM25 / keyword search ───────────────────────────────────────────
    def _bm25_search(
        self,
        query: str,
        top_k: int,
        intent: str = None,
        state: str = None,
        crop: str = None,
    ) -> list:
        """Full-text BM25 search using Elasticsearch multi_match query."""

        must_clause = {
            "multi_match": {
                "query":  query,
                "fields": ["query^2", "answer"],
                "type":   "best_fields",
            }
        }

        filters = []
        if intent:
            filters.append({"term": {"intent": intent}})
        if state:
            filters.append({"term": {"state": state}})
        if crop:
            filters.append({"term": {"crop": crop}})

        es_query = {
            "query": {
                "bool": {
                    "must":   must_clause,
                    "filter": filters,
                }
            },
            "size": top_k * 2,
            "_source": ["query", "answer", "intent", "crop", "state", "doc_id"],
        }

        result = self.es.search(index=ES_INDEX, body=es_query)

        hits = []
        for hit in result["hits"]["hits"]:
            hits.append({
                "doc_id":       hit["_source"].get("doc_id", -1),
                "query":        hit["_source"].get("query", ""),
                "answer":       hit["_source"].get("answer", ""),
                "intent":       hit["_source"].get("intent", ""),
                "crop":         hit["_source"].get("crop", ""),
                "state":        hit["_source"].get("state", ""),
                "bm25_score":   hit["_score"],
                "vector_score": 0.0,
            })
        return hits

    # ── Hybrid fusion ─────────────────────────────────────────────────
    def _fuse_results(self, bm25_hits: list, vector_hits: list, top_k: int) -> list:
        """
        Reciprocal Rank Fusion (RRF) to combine BM25 and vector results.
        RRF score = sum(1 / (rank + 60)) across both result lists.
        """
        RRF_K = 60
        scores = {}

        # BM25 ranks
        for rank, hit in enumerate(bm25_hits, 1):
            doc_id = hit["doc_id"]
            if doc_id not in scores:
                scores[doc_id] = {"hit": hit, "rrf": 0.0}
            scores[doc_id]["rrf"] += 1 / (rank + RRF_K)

        # Vector ranks
        for rank, hit in enumerate(vector_hits, 1):
            doc_id = hit["doc_id"]
            if doc_id not in scores:
                scores[doc_id] = {"hit": hit, "rrf": 0.0}
            scores[doc_id]["rrf"] += 1 / (rank + RRF_K)

        # Sort by RRF score descending
        ranked = sorted(scores.values(), key=lambda x: -x["rrf"])

        results = []
        for item in ranked[:top_k]:
            hit = item["hit"]
            hit["rrf_score"] = round(item["rrf"], 6)
            results.append(hit)

        return results

    # ── Static helpers ────────────────────────────────────────────────
    @staticmethod
    def normalize_crop_for_filter(crop: str):
        """Extract primary crop keyword for ES filtering."""
        if not crop or crop == "others":
            return None
        return crop.split("(")[0].strip().lower()

    @staticmethod
    def extract_crop_from_query(query: str):
        """Detect crop name from Hindi query text."""
        crop_map = {
            "गेहूं":   "wheat",   "गेहूँ":   "wheat",
            "धान":     "paddy",   "मक्का":   "maize",
            "सरसों":   "mustard", "धनिया":   "coriander",
            "प्याज":   "onion",   "आलू":     "potato",
            "टमाटर":   "tomato",  "कपास":    "cotton",
            "गन्ना":   "sugarcane","मूंग":   "moong",
            "अरहर":    "arhar",   "चना":     "gram",
            "सोयाबीन": "soybean", "बाजरा":  "bajra",
            "मूंगफली": "groundnut","आम":     "mango",
            "केला":    "banana",  "अमरूद":  "guava",
            "मिर्च":   "chilli",  "बैंगन":  "brinjal",
            "जौ":      "barley",  "मसूर":   "lentil",
        }
        for hindi_name, english_name in crop_map.items():
            if hindi_name in query:
                return english_name
        return None
    # ── Main retrieve method ───────────────────────────────────────────
    def retrieve(
        self,
        query:  str,
        top_k:  int  = TOP_K,
        intent: str  = None,
        state:  str  = None,
        crop:   str  = None,
    ) -> list:
        """
        Main retrieval with crop-aware filtering.
        Detects crop from query if not provided.
        """
        if not self._ready:
            raise RuntimeError("Retriever not connected.")

        # ── Detect crop from query if not explicit ──
        query_crop = self.extract_crop_from_query(query)
        filter_crop = self.normalize_crop_for_filter(crop) or query_crop

        try:
            bm25_hits   = self._bm25_search(
                query, top_k, intent, state, filter_crop
            )
            vector_hits = self._vector_search(query, top_k)
            fused       = self._fuse_results(bm25_hits, vector_hits, top_k)

            # ── Post-filter: if crop detected, drop passages for wrong crop ──
            if filter_crop and filter_crop not in ["others", "other"]:
                crop_filtered = [
                    p for p in fused
                    if filter_crop in p.get("crop", "").lower()
                    or p.get("crop", "") in ["others", ""]
                ]
                # Only apply crop filter if we get enough results
                if len(crop_filtered) >= 1:
                    fused = crop_filtered

            return fused
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []

    # ── Format for model input ─────────────────────────────────────────
    @staticmethod
    def format_context(passages: list) -> str:
        """
        Format retrieved passages into a single Hindi context string
        to prepend to the model's instruction prompt.
        """
        if not passages:
            return ""

        lines = []
        for i, p in enumerate(passages, 1):
            answer = str(p.get("answer", "")).strip()
            intent = str(p.get("intent", "")).strip()
            if answer:
                lines.append(f"{i}. [{intent}] {answer}")

        return "\n".join(lines)

    def is_ready(self) -> bool:
        return self._ready


# ── Singleton retriever ───────────────────────────────────────────────
_retriever: KisanMitraRetriever = None

def get_retriever() -> KisanMitraRetriever:
    global _retriever
    if _retriever is None:
        _retriever = KisanMitraRetriever()
        _retriever.connect()
    return _retriever