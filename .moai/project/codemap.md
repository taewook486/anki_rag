# Anki RAG - Architecture Codemap

> Updated for GraphRAG v2.0 | 2026-04-19

## System Overview

Anki 플래시카드(.apkg) 기반 영어 학습 RAG 시스템.
BGE-M3 하이브리드 임베딩 + Qdrant 벡터 DB + Multi-LLM 지원 + WordNet 기반 지식 그래프(GraphRAG v2.0).

```
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Streamlit   │  │   FastAPI    │  │     CLI      │      │
│  │  (web/app)   │  │  (api/main)  │  │ (__main__)   │      │
│  │ + Graph Tab  │  │  + /adaptive │  │              │      │
│  │   (v2.0)     │  │  + /graph    │  │              │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
├─────────┼──────────────────┼──────────────────┼─────────────┤
│         │      Application Layer              │             │
│         ├────────────────────┬────────────────┤             │
│         ▼                    ▼                ▼             │
│  ┌─────────────────┐   ┌────────────┐   ┌──────────────┐  │
│  │ RAG Pipeline    │   │ AdaptiveRAG│   │ LearningAgent│  │
│  │ (rag.py)        │   │(adaptive)  │   │ (agent.py)   │  │
│  └────────┬────────┘   └──────┬─────┘   └──────────────┘  │
│           │                   │                             │
│           │        ┌──────────▼──────────┐                  │
│           │        ▼                     ▼                  │
│           │  ┌──────────────────┐  ┌──────────────┐        │
│           └─▶│ GraphRAG Fusion  │  │HybridRetriever│       │
│              │(graph.py v2.0)   │  │(retriever.py)│       │
│              └─────────┬────────┘  └──────┬───────┘        │
├─────────────────────────┼──────────────────┼──────────────┤
│              Retrieval & Knowledge Layer   │               │
│  ┌────────────────────────┬────────────────┴────────┐     │
│  │                        ▼                         │     │
│  │  ┌──────────────────┐  ┌──────────────────────┐ │     │
│  │  │Knowledge Graph   │  │ HybridRetriever      │ │     │
│  │  │(WordNet+CO_OCCURS)  │(Dense+Sparse+RRF)    │ │     │
│  │  │pickle+GraphML    │  │(k=60, Exact Match)   │ │     │
│  │  └────────┬─────────┘  └──────┬───────────────┘ │     │
├──────────────┼──────────────────┼──────────────────┼──────┤
│   Data Layer │                  │                  │      │
│  ┌──────────▼─┐  ┌─────────────▼──┐  ┌──────────┐ │      │
│  │BGEEmbedder │  │ QdrantIndexer  │  │AnkiParser│ │      │
│  │(embedder)  │  │  (indexer)     │  │(parser)  │ │      │
│  └────────────┘  └────────┬───────┘  └──────────┘ │      │
│                           │                       │      │
│                    ┌──────▼────────┐               │      │
│                    │  Qdrant DB    │               │      │
│                    │(qdrant_data)  │               │      │
│                    └───────────────┘               │      │
└──────────────────────────────────────────────────────────┘
```

---

## Module Map

### Core Pipeline

| Module | File | Lines | Purpose |
|--------|------|-------|---------|
| Document | [models.py](src/models.py) | 33 | Pydantic v2 데이터 모델 (Document, SearchResult) |
| AnkiParser | [parser.py](src/parser.py) | 257 | .apkg / .txt 파일 파싱 → Document 리스트 |
| BGEEmbedder | [embedder.py](src/embedder.py) | 161 | BGE-M3 하이브리드 임베딩 (Dense 1024d + Sparse) |
| QdrantIndexer | [indexer.py](src/indexer.py) | 99 | Qdrant 벡터 DB 인덱싱 + 그래프 자동 빌드 (v2.0) |
| HybridRetriever | [retriever.py](src/retriever.py) | 238 | 하이브리드 검색 + RRF 퓨전 + search_dense_only() (v1.3) |
| RAGPipeline | [rag.py](src/rag.py) | 398 | Multi-LLM RAG 파이프라인 |
| AudioPlayer | [audio.py](src/audio.py) | ~50 | 크로스 플랫폼 오디오 재생 |
| **QueryClassifier** | **[adaptive.py](src/adaptive.py)** | **~250** | **쿼리 분류 (Simple/Moderate/Complex) + AdaptiveRAG (v1.3)** |
| **WordNetGraph** | **[graph.py](src/graph.py)** | **~400** | **WordNet 관계 추출 + CO_OCCURS 엣지 + GraphRAG Fusion (v2.0)** |

### API Layer

| Route | File | Endpoint | Method |
|-------|------|----------|--------|
| Search | [routes/search.py](src/api/routes/search.py) | `/api/search` | POST |
| Query | [routes/query.py](src/api/routes/query.py) | `/api/query` | POST |
| Audio | [routes/audio.py](src/api/routes/audio.py) | `/api/audio/{id}` | GET |
| Index | [routes/index.py](src/api/routes/index.py) | `/api/index` | POST/GET |
| **Adaptive** | **[routes/adaptive.py](src/api/routes/adaptive.py)** | **`/api/adaptive`** | **POST (v1.3)** |
| **Graph** | **[routes/graph.py](src/api/routes/graph.py)** | **`/api/graph/related/{word}`, `/api/graph/stats`** | **GET (v2.0)** |

### Entry Points

| Entry | File | Purpose |
|-------|------|---------|
| CLI | [__main__.py](src/__main__.py) | Click CLI (index, search, query, chat) |
| API | [api/main.py](src/api/main.py) | FastAPI 앱 (port 8000) |
| Web | [web/app.py](src/web/app.py) | Streamlit UI (port 8501) |
| Launcher | [start.bat](start.bat) | FastAPI + Streamlit 동시 실행 |

---

## Dependency Graph

```
__main__.py ──┬── parser.py ──── models.py
              ├── embedder.py
              ├── indexer.py ─── embedder.py, models.py, graph.py (v2.0)
              ├── retriever.py ─ embedder.py, models.py
              ├── rag.py ─────── retriever.py, models.py
              ├── agent.py ───── rag.py, retriever.py
              ├── adaptive.py ── retriever.py, rag.py, agent.py, graph.py (v1.3+v2.0)
              ├── graph.py ───── models.py (v2.0)
              └── audio.py

api/main.py ──┬── routes/search.py ── retriever.py
              ├── routes/query.py ─── rag.py, retriever.py
              ├── routes/agent.py ─── agent.py
              ├── routes/audio.py
              ├── routes/index.py ─── parser.py, embedder.py, indexer.py
              ├── routes/adaptive.py ─ adaptive.py (v1.3)
              └── routes/graph.py ─── graph.py (v2.0)

web/app.py ──── HTTP requests → api/main.py (+ graph visualization v2.0)
```

---

## Key Data Models

### Document (models.py)
```
Document
├── word: str               # 단어/표현
├── meaning: str            # 번역/정의
├── source: str             # 데이터 소스 (toefl, xfer, sentences)
├── deck: str               # 덱 이름
├── pronunciation: str?     # IPA 발음
├── example: str?           # 예문
├── example_translation: str? # 예문 번역
├── tags: list[str]         # 태그
├── note_type: str?         # Anki 노트 타입
├── audio_path: str?        # 오디오 파일 경로
├── difficulty: str?        # 난이도
└── synonyms: list[str]     # 동의어
```

### SearchResult (models.py)
```
SearchResult
├── document: Document      # 검색된 문서
├── score: float            # RRF 점수
└── rank: int               # 순위
```

---

## Data Flow Pipelines

### 1. Indexing Pipeline
```
.apkg / .txt ─→ [Parser] ─→ Document[] ─→ [BGEEmbedder] ─→ (Dense+Sparse)[] ─→ [QdrantIndexer] ─→ Qdrant DB
```

### 2. Search Pipeline
```
Query ─→ [embed_query] ─→ ┬─ Dense Search (top_k*3)
                           └─ Sparse Search (top_k*3)
                                    ↓
                              [RRF Fusion (k=60)]
                                    ↓
                          [Exact Match Boosting]
                                    ↓
                            [Deduplication]
                                    ↓
                          SearchResult[] (ranked)
```

### 3. RAG Pipeline
```
Question ─→ [Extract Keywords] ─→ [HybridRetriever.search]
                                         ↓
                                  [Build Context]
                                   (score≥0.005, max 2000 chars)
                                         ↓
                              [System Prompt + Few-shot]
                                         ↓
                                  [LLM Generate]
                                   (Claude/GLM/OpenAI)
                                         ↓
                                    Answer (str)
```

---

## External Dependencies

### Runtime
| Package | Version | Purpose |
|---------|---------|---------|
| flagembedding | >=1.2 | BGE-M3 임베딩 모델 |
| qdrant-client | >=1.9 | 벡터 DB 클라이언트 |
| openai | >=1.30 | OpenAI 호환 LLM API |
| pydantic | >=2.7 | 데이터 검증 |
| click | >=8.1 | CLI 프레임워크 |
| beautifulsoup4 | >=4.12 | HTML 태그 제거 |
| torch | >=2.3 | PyTorch (임베딩) |
| transformers | >=4.44,<4.52 | HuggingFace 모델 |
| python-dotenv | >=1.0 | 환경 변수 |
| **nltk** | **>=3.9.4** | **WordNet 코퍼스 (v2.0)** |
| **networkx** | **>=3.0** | **그래프 구조 (v2.0)** |

### Optional
| Package | Purpose |
|---------|---------|
| fastapi >=0.104 | REST API |
| uvicorn >=0.24 | ASGI 서버 |
| streamlit >=1.28 | Web UI |
| anthropic >=0.28 | Claude API |

---

## Configuration

### Environment Variables
| Variable | Default | Purpose |
|----------|---------|---------|
| `ANTHROPIC_API_KEY` | - | Anthropic Claude (우선순위 1) |
| `LLM_API_KEY` | - | OpenAI 호환 API 키 (우선순위 2) |
| `LLM_BASE_URL` | - | API 엔드포인트 |
| `LLM_MODEL` | gpt-4o-mini | LLM 모델명 |
| `QDRANT_LOCATION` | ./qdrant_data | Qdrant DB 경로 |
| `AUDIO_MEDIA_DIR` | ./data/media | 오디오 파일 디렉토리 |

---

## Critical Paths (ANCHOR)

### HybridRetriever.search (retriever.py)
- 하이브리드 검색의 핵심 메서드
- Dense + Sparse 결과를 RRF로 퓨전
- `fetch_multiplier=3`으로 후보 풀 확장
- Exact match 부스팅 (2x/1.5x)
- 단어 기반 중복 제거

### RAGPipeline.query (rag.py)
- 질문 → 키워드 추출 → 검색 → 컨텍스트 구성 → LLM 호출
- Few-shot 프롬프트 (2개 예시)
- 컨텍스트 제한: score>=0.005, 최대 2000자
- Multi-LLM 프로바이더 지원

### **AdaptiveRAG.query (adaptive.py) - v1.3+**
- 쿼리 분류: 휴리스틱(정규식) + LLM 2단계
- Simple: Dense-only 검색만
- Moderate: HybridRetriever (RRF)
- Complex: LearningAgent (ReAct) + GraphRAG Fusion

### **WordNetGraph.build_from_documents (graph.py) - v2.0**
- SYNONYM/ANTONYM 추출 (WordNet)
- CO_OCCURS 엣지 문서당 상한 (k=10)
- 그래프 영속화: pickle + GraphML
- GraphRAG Fusion: Complex 경로에서 자동 적용

---

## Test Coverage

| Test File | Target Module | Tests |
|-----------|---------------|-------|
| [test_models.py](tests/test_models.py) | Document, SearchResult 검증 | 7 |
| [test_parser.py](tests/test_parser.py) | AnkiParser, TextParser | 8 |
| [test_embedder.py](tests/test_embedder.py) | BGEEmbedder | 6 |
| [test_indexer.py](tests/test_indexer.py) | QdrantIndexer | 8 |
| [test_retriever.py](tests/test_retriever.py) | HybridRetriever, RRF | 12 |
| [test_rag.py](tests/test_rag.py) | RAGPipeline | 14 |
| [test_audio.py](tests/test_audio.py) | AudioPlayer | 4 |
| **[test_agent.py](tests/test_agent.py)** | **LearningAgent, ReAct** | **30 (v1.2)** |
| **[test_adaptive.py](tests/test_adaptive.py)** | **QueryClassifier, AdaptiveRAG** | **27 (v1.3)** |
| **[test_graph.py](tests/test_graph.py)** | **WordNetGraph, GraphRAG Fusion** | **78 (v2.0)** |

**Total**: 78 → 160 tests (+82 신규)
**Coverage Target**: **85%+** (pyproject.toml)
- graph.py: 86%, adaptive.py: 96%, indexer.py: 90%, api/routes/graph.py: 89%

---

## Stats

- **총 소스 코드**: ~2,700+ 라인 (src/) — v2.0에서 ~533줄 증가 (graph.py 400+, adaptive.py 250+)
- **총 테스트 코드**: ~1,500+ 라인 (tests/) — v2.0에서 ~637줄 증가 (test_graph.py 300+)
- **핵심 모듈**: 9개 (v1.1:7개 → v1.3+v2.0:9개)
- **API 엔드포인트**: 8개 (v1.1:5개 → v2.0:8개)
- **Entry Points**: 3개 (CLI, API, Web)
- **그래프 노드**: 5,000~10,000 (Anki 덱 크기에 따라 변동)
- **그래프 엣지**: 50,000~100,000 (SYNONYM+ANTONYM+DERIVATION+CO_OCCURS)
