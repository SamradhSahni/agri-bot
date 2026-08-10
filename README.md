<div align="center">

# 🌾 KisanMitra AI — किसानमित्र AI

**A Hindi-first agricultural advisory platform for Indian farmers**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-6.0-3178C6?style=flat-square&logo=typescript&logoColor=white)](https://typescriptlang.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Elasticsearch](https://img.shields.io/badge/Elasticsearch-9.x-005571?style=flat-square&logo=elasticsearch&logoColor=white)](https://elastic.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

</div>

---

## 📖 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Core Concepts Explained](#core-concepts-explained)
  - [Fine-tuned mT5 + QLoRA](#1-fine-tuned-mt5--qlora)
  - [Hybrid RAG Pipeline](#2-hybrid-rag-pipeline--bm25--vector-search)
  - [Crop Disease Detection (TRMS-ViT)](#3-crop-disease-detection--trms-vit)
  - [MSP Price Engine](#4-msp-price-engine)
  - [Intent Detection System](#5-intent-detection-system)
- [API Reference](#api-reference)
- [Getting Started](#getting-started)
- [Environment Variables](#environment-variables)
- [Frontend Architecture](#frontend-architecture)
- [Data & Knowledge Base](#data--knowledge-base)
- [Testing](#testing)

---

## Overview

**KisanMitra AI** (किसानमित्र AI — "Farmer's Friend AI") is a full-stack, AI-powered agricultural advisory platform designed specifically for **Hindi-speaking Indian farmers**. The platform bridges the gap between modern agricultural science and rural farmers who primarily communicate in Hindi.

The system combines a **fine-tuned multilingual language model** (mT5-base with QLoRA) with a **Hybrid Retrieval-Augmented Generation (RAG) pipeline** to deliver accurate, context-aware crop advisory, pest management guidance, MSP pricing, and government scheme information — all in conversational Hindi.

Additionally, a custom **TRMS-ViT** (Token-Refined Multi-Scale Vision Transformer) model detects 38 crop diseases from plant images, offering on-the-spot visual diagnostics.

### Who is it for?

- 🧑‍🌾 Indian farmers in the Hindi belt (UP, Rajasthan, MP, Bihar, Haryana, etc.)
- 🏛️ Agricultural extension officers and Kisan Call Centres
- 🔬 Researchers working on Indian agriculture AI
- 🚀 Developers building AgriTech products in India

---

## Key Features

| Feature | Description |
|---|---|
| 🗣️ **Hindi-first Chat** | Conversational advisory in Hindi using a QLoRA fine-tuned mT5 model |
| 🔍 **Hybrid RAG** | Combines BM25 keyword search + semantic vector search with Reciprocal Rank Fusion (RRF) |
| 🌿 **Crop Disease Detection** | Upload a plant image → get disease class + treatment advice (38 disease classes) |
| 💰 **MSP Price Lookup** | Real-time Minimum Support Prices for 14+ crops with Redis caching |
| 🧠 **Intent Detection** | Automatic classification into 11 agricultural intents (weather, pests, schemes, etc.) |
| 🗺️ **State-aware Responses** | Answers tailored to the farmer's state (UP, Rajasthan, Haryana, etc.) |
| 🌾 **Crop-aware Retrieval** | Knowledge base retrieval filtered by detected crop from the Hindi query |
| 💬 **Feedback Loop** | Thumbs up/down feedback collection stored to PostgreSQL for model improvement |
| 📊 **Latency Telemetry** | Per-request breakdown of retrieval time vs. generation time |
| 🔄 **Session Persistence** | Conversation sessions stored in PostgreSQL + cached in Redis |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND (React + Vite + TypeScript)    │
│  ┌──────────────┐  ┌─────────────────┐  ┌───────────────────┐  │
│  │  Chat Panel  │  │  Disease Detect │  │   MSP Widget      │  │
│  │  (Hindi UI)  │  │  (Image Upload) │  │   (Crop Prices)   │  │
│  └──────┬───────┘  └────────┬────────┘  └────────┬──────────┘  │
└─────────┼───────────────────┼────────────────────┼─────────────┘
          │ REST (Axios)       │                    │
          ▼                   ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FASTAPI BACKEND  (Python 3.10+)              │
│                                                                 │
│   /api/v1/chat          /api/v1/disease-predict                 │
│   /api/v1/msp           /api/v1/generate-advisory               │
│   /api/v1/feedback      /health                                 │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              KisanMitraRAGPipeline                       │   │
│  │                                                          │   │
│  │  1. Intent Detection  ──►  11-class keyword classifier   │   │
│  │  2. KisanMitraRetriever                                  │   │
│  │     ├─ BM25 Search (Elasticsearch)                       │   │
│  │     ├─ Vector Search (multilingual-e5-small embeddings)  │   │
│  │     └─ RRF Fusion + Quality Filtering                    │   │
│  │  3. Prompt Builder  ──►  Hindi instruction prompt        │   │
│  │  4. KisanMitraInference (mT5-base QLoRA)                 │   │
│  │     └─ Beam Search → Hindi response                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────┐  ┌────────────────┐  ┌─────────────────┐  │
│  │  PostgreSQL DB  │  │   Redis Cache  │  │  TRMS-ViT Model │  │
│  │  (sessions,     │  │  (MSP prices,  │  │  (38-class crop │  │
│  │   messages,     │  │   sessions)    │  │   disease det.) │  │
│  │   feedback)     │  │                │  │                 │  │
│  └─────────────────┘  └────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│              Elasticsearch Knowledge Base  (kisanmitra_kb)      │
│  ~100K+ farmer Q&A records from Kisan Call Centre (KCC) data    │
│  ● BM25 text fields:    query, answer                           │
│  ● Keyword filters:     crop, state, intent                     │
│  ● Dense vector field:  embedding (384-dim cosine similarity)   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

### Backend

| Technology | Version | Why Used |
|---|---|---|
| **Python** | 3.10+ | Core language for ML/AI workloads |
| **FastAPI** | 0.111 | High-performance async API framework with auto-generated OpenAPI docs |
| **PyTorch** | 2.2.0 | Deep learning framework for both LM inference and ViT disease detection |
| **Transformers (HuggingFace)** | 4.40.0 | Load and run the fine-tuned mT5 model for Hindi text generation |
| **PEFT + BitsAndBytes** | 0.10 / 0.43 | QLoRA fine-tuning — allows training mT5 on consumer GPUs with 4-bit quantization |
| **sentence-transformers** | 2.7.0 | `multilingual-e5-small` for 384-dim Hindi/English embeddings for semantic search |
| **Elasticsearch** | 9.x / SDK 8.13 | Dual-purpose: BM25 full-text search + KNN dense vector search in a single index |
| **PostgreSQL + psycopg2** | — | Persistent storage for chat sessions, messages, feedback, and MSP data |
| **Redis** | 5.0.4 | Fast in-memory cache for MSP prices and session context |
| **timm** | 0.9.16 | Provides the `vit_base_patch16_224` backbone for the TRMS-ViT disease model |
| **Uvicorn** | 0.29.0 | ASGI server for running FastAPI |
| **Mangum** | 0.17.0 | AWS Lambda adapter for serverless deployment |
| **Loguru** | 0.7.2 | Structured logging with file rotation |
| **indic-nlp-library** | 0.92 | Indic language processing utilities for Hindi text normalization |

### Frontend

| Technology | Version | Why Used |
|---|---|---|
| **React** | 19 | Component-based UI with concurrent features for smoother UX |
| **TypeScript** | 6.0 | Type safety across the frontend — catches API contract mismatches at compile time |
| **Vite** | 8.0 | Blazing-fast build tool with HMR for development |
| **React Router DOM** | 7.x | Client-side routing between Chat and Disease Detection pages |
| **Axios** | 1.15 | HTTP client for API calls |
| **Lucide React** | 1.8 | Tree-shakable, consistent icon library |
| **react-hot-toast** | 2.6 | Lightweight, accessible toast notifications |
| **Tailwind CSS** | 4.x | Utility-first CSS for rapid, consistent styling |
| **clsx** | 2.1 | Conditional className utility |
| **Google Fonts** | — | Fraunces (display), Plus Jakarta Sans (UI), Noto Sans Devanagari (Hindi text) |

### ML / AI Models

| Model | Purpose | Why This Model |
|---|---|---|
| **mT5-base** | Hindi text generation | Multilingual T5 pre-trained on mC4 corpus including large Hindi data; seq2seq fits instruction-following Q&A |
| **QLoRA** | Fine-tuning efficiency | 4-bit quantization + low-rank adapters allow fine-tuning 580M params on a single GPU |
| **multilingual-e5-small** | Semantic embeddings | 384-dim multilingual embeddings with strong Hindi performance; fast inference |
| **TRMS-ViT (custom)** | Crop disease detection | Custom ViT-Base + CNN multi-scale tokens with cross-attention for 38-class plant disease classification |
| **Llama 3.1 8B (NVIDIA NIM)** | Disease advisory generation | Generates detailed Hindi treatment advisories post-detection via NVIDIA API |

---

## Project Structure

```
kisanmitra-ai/
├── backend/                    # FastAPI application
│   ├── main.py                 # App entrypoint, lifespan, middleware, routers
│   ├── inference.py            # KisanMitraInference — mT5 model loading & generation
│   ├── trms_model.py           # TRMS-ViT model definition, plant disease classes
│   ├── api/
│   │   ├── models/
│   │   │   └── schemas.py      # Pydantic request/response models
│   │   └── routes/
│   │       ├── chat.py         # POST /api/v1/chat
│   │       ├── disease.py      # POST /api/v1/disease-predict, /generate-advisory
│   │       ├── msp.py          # GET  /api/v1/msp
│   │       └── feedback.py     # POST /api/v1/feedback
│   ├── rag/
│   │   ├── pipeline.py         # KisanMitraRAGPipeline — orchestrates full RAG flow
│   │   ├── retriever.py        # KisanMitraRetriever — BM25 + vector + RRF fusion
│   │   └── indexer.py          # One-time script to bulk-index KB into Elasticsearch
│   ├── db/
│   │   ├── database.py         # PostgreSQL connection pool, CRUD helpers
│   │   └── init_db.py          # Schema initialization (sessions, messages, feedback)
│   └── cache/
│       └── redis_cache.py      # Redis helpers for MSP caching + session context
│
├── frontend/                   # React + TypeScript + Vite
│   └── src/
│       ├── App.tsx             # Root component, global styles, chat interface
│       ├── config.ts           # API URLs, states, crops, intent labels
│       ├── types.ts            # TypeScript interfaces
│       ├── api/
│       │   └── client.ts       # Axios API client functions
│       ├── components/
│       │   ├── ChatPanel.tsx   # Message list + input bar
│       │   ├── Header.tsx      # App header with health status indicator
│       │   ├── MSPWidget.tsx   # MSP price lookup widget
│       │   ├── ProfilePanel.tsx# Farmer profile (state, crop selection)
│       │   └── SourcesPanel.tsx# RAG retrieved passages display
│       └── pages/
│           ├── HomePage.tsx    # Main chat + sidebar layout
│           └── CropDiseasePage.tsx  # Image upload + disease detection UI
│
├── Crop_Disease_detection/
│   └── best_trms_vit.pth       # Trained TRMS-ViT weights (~340MB)
│
├── IndicTrans2/                 # IndicTrans2 translation model (Hindi <> English)
├── IndicTransToolkit/           # Toolkit for IndicTrans2 preprocessing
│
├── data/
│   └── embeddings/             # Pre-generated KB embeddings (kb_records.jsonl, .npy)
│
├── model/
│   └── final/                  # Fine-tuned mT5 model weights + tokenizer
│
├── notebooks/                  # Jupyter notebooks for training & experimentation
├── scripts/                    # Data processing, training, eval scripts
├── utils/
│   └── config_loader.py        # Loads config.yaml as CONFIG singleton
│
├── config.yaml                 # Master config: intents, crops, RAG params, states
├── requirements.txt            # Python dependencies
└── .env                        # Environment variables (not committed)
```

---

## Core Concepts Explained

### 1. Fine-tuned mT5 + QLoRA

**What it is:** The primary language model that generates Hindi advisory text.

**Why mT5-base?**
- `mT5-base` (Multilingual T5) is pre-trained on the mC4 corpus which includes hundreds of gigabytes of Hindi web text. This gives it a strong prior understanding of Hindi grammar, agricultural vocabulary, and Devanagari script — without needing to train from scratch.
- Its encoder-decoder (seq2seq) architecture is naturally suited for the **instruction → response** format: the encoder reads the structured Hindi prompt (with farmer context: state, crop, intent, RAG context), and the decoder generates a targeted advisory response.

**Why QLoRA for fine-tuning?**
- Fine-tuning all 580M parameters of mT5-base in full precision requires ~10GB+ VRAM. QLoRA solves this via:
  1. **4-bit quantization** (via `bitsandbytes`) — loads the base model at 4-bit precision, reducing memory footprint ~4x
  2. **Low-Rank Adapters (LoRA)** (via `peft`) — trains small rank-decomposition matrices (`r=16`, `alpha=32`) inserted at attention layers instead of updating all weights
- This allows fine-tuning on consumer-grade GPUs (16GB VRAM) while achieving near-full-precision performance on domain-specific tasks.

**Prompt structure:**
```
निर्देश: आप एक कृषि विशेषज्ञ हैं। किसान की समस्या का उत्तर हिंदी में दें।
राज्य: हरियाणा
फसल: गेहूं
समस्या का प्रकार: कीट प्रबंधन
संदर्भ जानकारी:
1. गेहूं में माहू कीट नियंत्रण के लिए इमिडाक्लोप्रिड 17.8 SL...
किसान का प्रश्न: गेहूं में माहू का प्रकोप है, क्या करें?
उत्तर:
```

**Generation strategy:** Beam search (`num_beams=4`) with `no_repeat_ngram_size=3` to balance response quality and diversity.

---

### 2. Hybrid RAG Pipeline — BM25 + Vector Search

**What it is:** A Retrieval-Augmented Generation system that retrieves relevant farmer Q&A pairs from a knowledge base before generating a response.

**Why RAG?**
- The fine-tuned mT5 model has excellent language generation ability but its **parametric knowledge is frozen** at training time. RAG solves this by **retrieving factual grounding** at inference time from a curated knowledge base of real farmer queries and expert answers (sourced from the Kisan Call Centre dataset).

**Why Hybrid (BM25 + Vector)?**
- **BM25** excels at **exact keyword matching** — if a farmer asks about "गेहूं में माहू", BM25 finds passages that literally contain those words.
- **Vector search** (via `multilingual-e5-small` embeddings) captures **semantic similarity** — it understands that "गेहूं में कीट नियंत्रण" and "wheat pest management" mean the same thing, even without keyword overlap.
- Neither alone is optimal: BM25 misses paraphrases; vector search can return semantically related but factually wrong passages. **Together they are complementary.**

**Reciprocal Rank Fusion (RRF):**
```
RRF_score(doc) = SUM( 1 / (rank_in_list + 60) )  over all lists
```
Each document gets a score based on its rank position in both result lists. This rewards documents that consistently appear near the top of both BM25 and vector results, without needing to normalize incompatible score scales (BM25 scores are not comparable to cosine similarity scores directly).

**Crop-aware Filtering:**
After RRF fusion, the retriever detects the crop mentioned in the Hindi query (e.g., "गेहूं" → `wheat`) and filters out passages for unrelated crops, ensuring the retrieved context is crop-specific.

**Elasticsearch as the unified backend:**
Elasticsearch serves dual duty — it stores both the BM25 inverted index (via standard `text` analyzer) and the dense vector index (via `dense_vector` field with cosine similarity). This avoids running a separate FAISS or pgvector instance.

---

### 3. Crop Disease Detection — TRMS-ViT

**What it is:** A custom hybrid vision model that classifies plant images into 38 disease categories across multiple crops (Rice, Wheat, Corn, Tomato, Potato, Cotton, Apple, Sugarcane, Pepper).

**Architecture — TRMSViT (Token-Refined Multi-Scale ViT):**

```
Input Image (224×224)
        |
        |----------------------------------------------|
        v                                              v
  ViT-Base (patch16)                    CNN Token Extractor
  (pretrained backbone)                 Conv2d(3->64->128->256)
  Global CLS token [768-dim]            AdaptiveAvgPool -> Linear
        |                               -> [768-dim local feature]
        |-------------------|--------------------------|
                            v
                  Cross-Attention Module
                  (ViT features as Query,
                   CNN features as Key/Value)
                  Fuses local texture info
                  into global ViT representation
                            |
                            v
                  Classifier Head
                  LayerNorm -> Linear(768->512) -> ReLU
                  -> Dropout(0.5) -> Linear(512->38)
```

**Why this hybrid CNN + ViT approach?**
- **ViT alone** captures long-range global patterns but can miss fine-grained local textures (lesion edges, color gradients) critical for disease identification.
- **CNN alone** captures local features but lacks global context.
- **Cross-attention fusion** lets the ViT global representation attend to the CNN's local features — particularly important for distinguishing visually similar diseases (e.g., Early Blight vs. Late Blight in tomatoes).

**Post-detection Advisory:**
After classification, the system calls NVIDIA NIM (Llama 3.1 8B Instruct) to generate a structured, farmer-friendly treatment advisory in Hindi covering: disease cause, how it spreads, chemical treatment, organic treatment, preventive measures, and practical advice.

---

### 4. MSP Price Engine

**What it is:** A three-layer cached lookup system for Minimum Support Prices of 14+ crops.

**Why three layers?**
```
Request -> Layer 1: Redis Cache (sub-millisecond)
               | MISS
               v
           Layer 2: PostgreSQL (persisted, accurate)
               | MISS / DB down
               v
           Layer 3: In-memory fallback dict (always available)
```
This guarantees **100% availability** of MSP data even if the database is temporarily unreachable, while Redis ensures fast responses for frequently queried crops.

---

### 5. Intent Detection System

**What it is:** A keyword-based classifier that categorizes farmer queries into one of **11 agricultural intents**.

| Intent | Hindi Label | Example Query |
|---|---|---|
| `weather_sowing` | मौसम एवं बुवाई | "इस सप्ताह बारिश होगी क्या?" |
| `crop_advisory` | फसल सलाह | "गेहूं की बुवाई कब करें?" |
| `pest_id` | कीट प्रबंधन | "मेरी फसल में कीड़े लग रहे हैं" |
| `disease` | रोग प्रबंधन | "पत्तियों पर धब्बे आ रहे हैं" |
| `nutrient_management` | पोषक तत्व प्रबंधन | "यूरिया कितनी डालें?" |
| `msp_price` | मूल्य एवं बाजार | "गेहूं का MSP क्या है?" |
| `government_scheme` | सरकारी योजना | "PM किसान योजना में कैसे रजिस्टर करें?" |
| `horticulture` | बागवानी | "आम के पेड़ की देखभाल कैसे करें?" |
| `soil_water` | मृदा एवं जल | "मिट्टी की जांच कैसे करें?" |
| `animal_husbandry` | पशुपालन | "गाय को क्या खिलाएं?" |
| `equipment_machinery` | कृषि यंत्र | "रोटावेटर कहां मिलेगा?" |

**Why keyword-based (not ML-based)?**
- Covers both Hindi and English terms for Hinglish queries common among farmers
- Zero latency — no model inference needed
- Transparent and debuggable
- Feeds intent as a **filter to Elasticsearch BM25 search**, improving retrieval precision

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | API health check — model status, RAG status, uptime |
| `POST` | `/api/v1/chat` | Main chat endpoint — Hindi Q&A with RAG |
| `GET` | `/api/v1/msp?crop=wheat` | MSP price lookup for a crop |
| `POST` | `/api/v1/disease-predict` | Upload plant image → disease classification |
| `POST` | `/api/v1/generate-advisory` | Generate detailed Hindi advisory for a disease |
| `POST` | `/api/v1/feedback` | Submit thumbs up/down feedback for a response |

### Chat Request / Response

```json
// POST /api/v1/chat  — Request
{
  "query": "गेहूं में माहू कीट का प्रकोप है, क्या उपाय करें?",
  "state": "HARYANA",
  "crop": "wheat",
  "session_id": "uuid-optional",
  "use_rag": true
}

// Response
{
  "response": "गेहूं में माहू (एफिड) के नियंत्रण के लिए...",
  "intent": "pest_id",
  "rag_used": true,
  "passages": [
    {
      "answer": "...",
      "intent": "pest_id",
      "crop": "wheat",
      "state": "HARYANA",
      "rrf_score": 0.03214
    }
  ],
  "latency_ms": 1240,
  "retrieval_ms": 180,
  "generation_ms": 1060,
  "session_id": "...",
  "timestamp": "2024-08-10T11:23:00Z"
}
```

Interactive API docs available at **`http://localhost:8000/docs`** (Swagger UI) and **`http://localhost:8000/redoc`**.

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- Elasticsearch 9.x (running locally or remote)
- PostgreSQL 14+
- Redis 7+
- CUDA GPU (optional, recommended for faster inference)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/kisanmitra-ai.git
cd kisanmitra-ai
```

### 2. Backend Setup

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Copy environment template and configure
cp .env.example .env
# Edit .env with your DB credentials, API keys, etc.
```

### 3. Initialize the Database

```bash
python -m backend.db.init_db
```

### 4. Build the Knowledge Base Index

> This step requires the KB embeddings to be pre-generated in `data/embeddings/`.

```bash
# Start Elasticsearch first, then run:
python -m backend.rag.indexer
```

### 5. Start the Backend

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Frontend Setup

```bash
cd frontend
npm install
npm run dev    # Starts at http://localhost:5173
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
# Model
FINETUNED_MODEL_PATH=./model/final
MAX_INPUT_LENGTH=512
MAX_OUTPUT_LENGTH=128

# Elasticsearch
ES_HOST=http://localhost:9200
ES_INDEX_NAME=kisanmitra_kb

# PostgreSQL
DB_HOST=localhost
DB_PORT=5432
DB_NAME=kisanmitra
DB_USER=postgres
DB_PASSWORD=your_password_here

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# External APIs
NVIDIA_API_KEY=your_nvidia_nim_api_key
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=meta/llama-3.1-8b-instruct

# HuggingFace (for model downloads)
HUGGINGFACE_TOKEN=your_hf_token
```

---

## Frontend Architecture

The frontend is a **React 19 + TypeScript + Vite** SPA with two routes:

### Pages

- **`/` — HomePage**: The main chat interface.
  - Farmer profile sidebar (state + crop selection)
  - Chat panel with message history, typing indicator, RAG sources panel
  - MSP widget for quick price lookups
  - Voice input support (Web Speech API)
  - Feedback buttons per message

- **`/disease` — CropDiseasePage**: The disease detection interface.
  - Drag-and-drop image upload
  - Real-time disease classification results
  - AI-generated Hindi advisory panel

### Design System

The UI uses a custom **agricultural green color palette** with CSS custom properties:
- Forest green (`#0f3b1f`) for dark backgrounds
- Sage green (`#2d7a46`) for primary UI
- Warm cream (`#fefdf8`) for chat backgrounds
- Amber gold (`#d97706`) for highlights and prices

**Typography:** Fraunces (display headings) + Plus Jakarta Sans (UI text) + Noto Sans Devanagari (Hindi content)

---

## Data & Knowledge Base

The knowledge base is sourced from the **Kisan Call Centre (KCC)** dataset — a government agricultural helpline database containing hundreds of thousands of real farmer queries and expert responses.

**Data processing pipeline:**
1. **Cleaning**: Remove noise patterns (blank queries, call artifacts, "farmer did not respond")
2. **State filtering**: Keep only Hindi-belt states (UP, Rajasthan, MP, Bihar, etc.)
3. **Intent labeling**: Keyword-based intent assignment to all records
4. **Deduplication**: MinHash LSH for near-duplicate removal (`datasketch`)
5. **Language detection**: Confirm Hindi content using `langdetect`
6. **Embedding generation**: Batch encode with `multilingual-e5-small` → save as `.npy`
7. **Indexing**: Bulk upload to Elasticsearch with both text fields and dense vectors

---

## Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest backend/tests/test_pipeline.py

# Run with coverage report
pytest --cov=backend
```

Test configuration is defined in `pytest.ini`.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with love for Indian farmers**

*किसान भारत की रीढ़ हैं — Farmers are the backbone of India*

</div>
