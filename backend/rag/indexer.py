import os
import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers

load_dotenv()
sys.path.insert(0, ".")
from utils.config_loader import CONFIG

log_path = Path("logs/indexer.log")
log_path.parent.mkdir(exist_ok=True)
logger.add(str(log_path), rotation="10 MB", encoding="utf-8")

# ── Config ────────────────────────────────────────────────────────────
ES_HOST       = os.getenv("ES_HOST",       "http://localhost:9200")
ES_INDEX      = os.getenv("ES_INDEX_NAME", "kisanmitra_kb")
EMBEDDING_DIM = CONFIG["rag"]["embedding_dim"]   # 384
BATCH_SIZE    = 500

KB_JSONL_PATH   = "./data/embeddings/kb_records.jsonl"
EMBEDDINGS_PATH = "./data/embeddings/kb_embeddings.npy"


# ── Connect to Elasticsearch ──────────────────────────────────────────
def connect_es() -> Elasticsearch:
    es = Elasticsearch(
        ES_HOST,
        request_timeout=60,
        max_retries=3,
        retry_on_timeout=True,
    )
    if not es.ping():
        raise ConnectionError(
            f"Cannot connect to Elasticsearch at {ES_HOST}\n"
            "Make sure Elasticsearch is running."
        )
    info = es.info()
    logger.success(f"Connected to Elasticsearch {info['version']['number']} at {ES_HOST}")
    return es


# ── Create index with dense_vector mapping ────────────────────────────
def create_index(es: Elasticsearch):
    """
    Create ES index with:
    - BM25 text fields for keyword search
    - dense_vector field for cosine similarity search
    """
    if es.indices.exists(index=ES_INDEX):
        logger.warning(f"Index '{ES_INDEX}' already exists — deleting and recreating")
        es.indices.delete(index=ES_INDEX)

    mapping = {
        "settings": {
            "number_of_shards":   1,
            "number_of_replicas": 0,    # local dev — no replicas needed
            "similarity": {
                "default": {
                    "type": "BM25",
                    "b":    0.75,
                    "k1":   1.2,
                }
            }
        },
        "mappings": {
            "properties": {
                # ── Text fields for BM25 search ──
                "query": {
                    "type":     "text",
                    "analyzer": "standard"
                },
                "answer": {
                    "type":     "text",
                    "analyzer": "standard",
                },
                # ── Keyword fields for filtering ──
                "intent": {
                    "type": "keyword",
                },
                "crop": {
                    "type": "keyword",
                },
                "state": {
                    "type": "keyword",
                },
                "source": {
                    "type": "keyword",
                },
                "doc_id": {
                    "type": "integer",
                },
                # ── Dense vector for cosine similarity ──
                "embedding": {
                    "type":       "dense_vector",
                    "dims":       EMBEDDING_DIM,     # 384
                    "index":      True,
                    "similarity": "cosine",
                },
            }
        }
    }

    es.indices.create(index=ES_INDEX, body=mapping)
    logger.success(f"Index '{ES_INDEX}' created with dense_vector mapping")


# ── Load KB records and embeddings ───────────────────────────────────
def load_kb_data() -> tuple:
    records = []
    with open(KB_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading KB records"):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    embeddings = np.load(EMBEDDINGS_PATH)
    logger.info(f"Loaded {len(records):,} records and embeddings {embeddings.shape}")
    assert len(records) == len(embeddings), \
        f"Mismatch: {len(records)} records vs {len(embeddings)} embeddings"
    return records, embeddings


# ── Index documents in bulk ───────────────────────────────────────────
def index_documents(es: Elasticsearch, records: list, embeddings: np.ndarray):
    """
    Bulk index all KB records with their embeddings into Elasticsearch.
    """
    logger.info(f"Indexing {len(records):,} documents into '{ES_INDEX}'...")

    def generate_actions():
        for i, (record, emb) in enumerate(zip(records, embeddings)):
            yield {
                "_index": ES_INDEX,
                "_id":    i,
                "_source": {
                    "doc_id":    i,
                    "query":     str(record.get("query",  "")).strip(),
                    "answer":    str(record.get("answer", "")).strip(),
                    "crop":      str(record.get("crop",   "")).strip(),
                    "state":     str(record.get("state",  "")).strip(),
                    "intent":    str(record.get("intent", "")).strip(),
                    "source":    str(record.get("source", "kcc")).strip(),
                    "embedding": emb.tolist(),
                }
            }

    # Bulk index with progress bar
    success = 0
    errors  = 0

    for ok, action in tqdm(
        helpers.streaming_bulk(
            es,
            generate_actions(),
            chunk_size=BATCH_SIZE,
            raise_on_error=False,
        ),
        total=len(records),
        desc="Indexing",
    ):
        if ok:
            success += 1
        else:
            errors += 1

    es.indices.refresh(index=ES_INDEX)
    logger.success(f"Indexed {success:,} docs — {errors} errors")
    return success, errors


# ── Verify index ──────────────────────────────────────────────────────
def verify_index(es: Elasticsearch):
    count = es.count(index=ES_INDEX)["count"]
    stats = es.indices.stats(index=ES_INDEX)
    size_mb = stats["indices"][ES_INDEX]["total"]["store"]["size_in_bytes"] / 1024**2

    sep = "=" * 65
    print(f"\n{sep}")
    print("  Elasticsearch Index Verification")
    print(sep)
    print(f"  Index name    : {ES_INDEX}")
    print(f"  Document count: {count:,}")
    print(f"  Index size    : {size_mb:.1f} MB")

    # Test a sample BM25 search
    print(f"\n  ── BM25 Test Search ─────────────────────────────────")
    result = es.search(
        index=ES_INDEX,
        body={
            "query": {
                "multi_match": {
                    "query":  "मक्का में कीट नियंत्रण",
                    "fields": ["query^2", "answer"],
                }
            },
            "size": 3,
        }
    )
    for hit in result["hits"]["hits"]:
        print(f"  Score: {hit['_score']:.4f} | "
              f"Intent: {hit['_source']['intent']} | "
              f"Q: {hit['_source']['query'][:60]}")

    print(f"\n  ✅ Index verified and searchable")
    print(f"{sep}\n")


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 65)
    logger.info("KisanMitra AI — ES Indexing (Task 11b part 1)")
    logger.info("=" * 65)

    es = connect_es()
    create_index(es)
    records, embeddings = load_kb_data()
    success, errors = index_documents(es, records, embeddings)
    verify_index(es)

    logger.success(f"Indexing complete: {success:,} docs in '{ES_INDEX}'")