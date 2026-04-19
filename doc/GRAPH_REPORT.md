# Graph Report - src/+tests/+doc/  (2026-04-19)

## Corpus Check
- 43 files · ~41,592 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 738 nodes · 2387 edges · 19 communities detected
- Extraction: 38% EXTRACTED · 62% INFERRED · 0% AMBIGUOUS · INFERRED: 1490 edges (avg confidence: 0.58)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Adaptive RAG Engine|Adaptive RAG Engine]]
- [[_COMMUNITY_LLM Providers & Strategy Mix|LLM Providers & Strategy Mix]]
- [[_COMMUNITY_Streamlit Web UI|Streamlit Web UI]]
- [[_COMMUNITY_Audio Playback|Audio Playback]]
- [[_COMMUNITY_Knowledge Graph (GraphRAG)|Knowledge Graph (GraphRAG)]]
- [[_COMMUNITY_Adaptive API Layer|Adaptive API Layer]]
- [[_COMMUNITY_ReAct Agent Internals|ReAct Agent Internals]]
- [[_COMMUNITY_BGE-M3 Embedding|BGE-M3 Embedding]]
- [[_COMMUNITY_REST API Routes|REST API Routes]]
- [[_COMMUNITY_Data Sources & Citations|Data Sources & Citations]]
- [[_COMMUNITY_RAG Query API|RAG Query API]]
- [[_COMMUNITY_MD-to-HWP Converter|MD-to-HWP Converter]]
- [[_COMMUNITY_Test Init Files|Test Init Files]]
- [[_COMMUNITY_FastAPI Root|FastAPI Root]]
- [[_COMMUNITY_Project Definition|Project Definition]]
- [[_COMMUNITY_Audio Backward Compat|Audio Backward Compat]]
- [[_COMMUNITY_Orphan __init__ (1)|Orphan __init__ (1)]]
- [[_COMMUNITY_Orphan __init__ (2)|Orphan __init__ (2)]]
- [[_COMMUNITY_Orphan __init__ (3)|Orphan __init__ (3)]]

## God Nodes (most connected - your core abstractions)
1. `Document` - 219 edges
2. `SearchResult` - 163 edges
3. `AgentResult` - 96 edges
4. `HybridRetriever` - 90 edges
5. `RAGPipeline` - 84 edges
6. `LearningAgent` - 81 edges
7. `AgentStep` - 80 edges
8. `AdaptiveRAG` - 54 edges
9. `AdaptiveResult` - 52 edges
10. `QueryComplexity` - 48 edges

## Surprising Connections (you probably didn't know these)
- `임베딩 - BGE-M3 Dense+Sparse 벡터 생성` --uses--> `Document`  [INFERRED]
  src\embedder.py → src\models.py
- `Args:             model_name: HuggingFace 모델명             device: 실행 디바이스 (cuda/` --uses--> `Document`  [INFERRED]
  src\embedder.py → src\models.py
- `단일 문서 임베딩 (문서 인덱싱용 — instruction 없음)          Args:             doc: Document 객체` --uses--> `Document`  [INFERRED]
  src\embedder.py → src\models.py
- `배치 임베딩 (문서 인덱싱용 — instruction 없음)          Args:             docs: Document 리스트` --uses--> `Document`  [INFERRED]
  src\embedder.py → src\models.py
- `검색 쿼리 임베딩 (BGE-M3 query instruction prefix 적용)          문서 임베딩(embed/embed_batch` --uses--> `Document`  [INFERRED]
  src\embedder.py → src\models.py

## Hyperedges (group relationships)
- **Indexing pipeline (Parser -> Embedder -> Indexer -> Qdrant)** — spec_module_parser, spec_module_embedder, spec_module_indexer, spec_concept_qdrant [EXTRACTED 1.00]
- **Search pipeline (Embedder -> Retriever -> RAG -> LLM)** — spec_module_embedder, spec_module_retriever, spec_module_rag, spec_concept_rrf_fusion [EXTRACTED 1.00]
- **RAG paradigm family (Naive/Advanced/Modular/Self/Corrective/Adaptive/GraphRAG)** — spec_concept_naive_rag, spec_concept_advanced_rag, spec_concept_modular_rag, spec_concept_self_rag, spec_concept_corrective_rag, spec_concept_adaptive_rag, spec_concept_graph_rag [EXTRACTED 1.00]

## Communities

### Community 0 - "Adaptive RAG Engine"
Cohesion: 0.07
Nodes (100): AdaptiveRAG, AdaptiveResult, classify_query(), classify_query_heuristic(), classify_query_llm(), QueryComplexity, Adaptive RAG API 라우트  설계서 13.4 — POST /api/adaptive 엔드포인트:     쿼리 복잡도를 자동 분류하여 최, 쿼리 복잡도 분류 (휴리스틱 우선, 불확실 시 LLM)      Args:         question: 사용자 질문         provi (+92 more)

### Community 1 - "LLM Providers & Strategy Mix"
Cohesion: 0.06
Nodes (47): Simple 전략: Dense-only 검색 → RAG 응답, AnthropicProvider, create_provider(), _extract_search_query(), _extract_system(), _import_anthropic(), OpenAICompatibleProvider, RAGPipeline (+39 more)

### Community 2 - "Streamlit Web UI"
Cohesion: 0.05
Nodes (42): main(), show_admin_page(), show_chat_page(), show_search_page(), _trigger_indexing(), CacheClearResponse, CacheStatsResponse, clear_all_caches() (+34 more)

### Community 3 - "Audio Playback"
Cohesion: 0.06
Nodes (42): AudioPlayer, get_audio_id(), 오디오 스트리밍 API 라우트 - SPEC-RAG-002 REQ-004, audio_id(hex hash)로 실제 파일 경로 복원      audio_id는 audio_path의 MD5 hex digest.     d, 오디오 파일 재생          Args:             audio_path: 오디오 파일 경로 (None이면 무시), 오디오 파일 스트리밍      - **audio_id**: audio_path의 MD5 hex digest     - HTTP Range 요청, Windows 오디오 재생 - 시스템 네이티브 PowerShell 사용, audio_path 문자열로 audio_id(MD5) 조회      Streamlit에서 audio_path -> audio_id 변환에 사용. (+34 more)

### Community 4 - "Knowledge Graph (GraphRAG)"
Cohesion: 0.11
Nodes (39): Enum, build_from_documents(), graph_rag_fusion(), RelationType, WordKnowledgeGraph, WordNode, WordRelation, graph() (+31 more)

### Community 5 - "Adaptive API Layer"
Cohesion: 0.06
Nodes (45): adaptive_query(), AdaptiveRequest, AdaptiveResponse, AgentStepInfo, get_adaptive(), AdaptiveRAG 인스턴스 반환 (lazy initialization), Agent 스텝 요약 (Complex 전략 시), SourceInfo (+37 more)

### Community 6 - "ReAct Agent Internals"
Cohesion: 0.07
Nodes (19): _extract_action(), _extract_thought(), _is_low_score_result(), _is_search_tool(), ReAct 루프 실행          Args:             question: 사용자 질문          Returns:, Tool 실행 + Self-Correction + Corrective RAG (설계서 12.4, 13.3)          1. 검색 결과, Tool 호출 및 결과를 문자열로 반환, Self-RAG: 검색 필요 여부 판단 (설계서 13.3)          LLM이 질문을 보고 Anki DB 검색이 필요한지 자율 판단한다 (+11 more)

### Community 7 - "BGE-M3 Embedding"
Cohesion: 0.07
Nodes (33): BGEEmbedder, EmbeddingResult, model(), 임베딩 - BGE-M3 Dense+Sparse 벡터 생성, 검색 쿼리 임베딩 (BGE-M3 query instruction prefix 적용)          문서 임베딩(embed/embed_batch, 쿼리 임베딩 텍스트 구성 — BGE-M3 권장 instruction prefix 적용, Sparse 가중치를 dict로 변환          BGE-M3의 lexical_weights는 dict[int, float] 형태, Args:             model_name: HuggingFace 모델명             device: 실행 디바이스 (cuda/ (+25 more)

### Community 8 - "REST API Routes"
Cohesion: 0.05
Nodes (53): POST /api/adaptive, POST /api/index, POST /api/query, POST /api/search, CLI __main__.py entry, Adaptive RAG (complexity-based strategy), Advanced RAG, BGE-M3 embedding model (+45 more)

### Community 9 - "Data Sources & Citations"
Cohesion: 0.08
Nodes (26): 24,472 documents across 6 sources, GET /api/audio/{id}, .apkg ZIP format (SQLite + media), Citation: Microsoft GraphRAG (2024), collection.anki2 (fallback), collection.anki21 (preferred), GraphRAG (knowledge graph fusion), Knowledge Graph (Word/Meaning/Category nodes) (+18 more)

### Community 10 - "RAG Query API"
Cohesion: 0.18
Nodes (17): get_rag(), query(), QueryRequest, QueryResponse, RAG 질의 API 라우트 - RAGPipeline 연동, RAGPipeline 인스턴스 반환 (lazy initialization, retriever 공유), RAG 질의 수행      - **question**: 질문     - **top_k**: 검색할 문서 수 (1-20)     - **s, SourceInfo (+9 more)

### Community 11 - "MD-to-HWP Converter"
Cohesion: 0.18
Nodes (15): _apply_table_borders(), _find_hwp(), _find_pandoc(), main(), _make_tc_borders(), md_to_docx(), open_in_hwp(), _patch_document_xml() (+7 more)

### Community 12 - "Test Init Files"
Cohesion: 0.5
Nodes (1): Tests for Anki RAG 시스템.

### Community 13 - "FastAPI Root"
Cohesion: 0.67
Nodes (0): 

### Community 14 - "Project Definition"
Cohesion: 1.0
Nodes (2): Anki RAG System, Problem: Anki lacks semantic search

### Community 15 - "Audio Backward Compat"
Cohesion: 1.0
Nodes (1): 기존 audio_path(단수)를 audio_paths(복수)로 변환 — 하위 호환

### Community 16 - "Orphan __init__ (1)"
Cohesion: 1.0
Nodes (0): 

### Community 17 - "Orphan __init__ (2)"
Cohesion: 1.0
Nodes (0): 

### Community 18 - "Orphan __init__ (3)"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **84 isolated node(s):** `Moderate 전략: Hybrid RRF 검색 → RAG 응답 (기존 파이프라인)`, `LLM으로 쿼리 재작성 (Self-Correction 보조)`, `유의어·파생어 검색 (벡터 유사도 기반)`, `발음 오디오 재생 (API 환경에서는 경로 반환)`, `오디오 파일 재생          Args:             audio_path: 오디오 파일 경로 (None이면 무시)` (+79 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Project Definition`** (2 nodes): `Anki RAG System`, `Problem: Anki lacks semantic search`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Audio Backward Compat`** (1 nodes): `기존 audio_path(단수)를 audio_paths(복수)로 변환 — 하위 호환`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Orphan __init__ (1)`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Orphan __init__ (2)`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Orphan __init__ (3)`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Document` connect `Adaptive RAG Engine` to `LLM Providers & Strategy Mix`, `Streamlit Web UI`, `Audio Playback`, `Knowledge Graph (GraphRAG)`, `Adaptive API Layer`, `ReAct Agent Internals`, `BGE-M3 Embedding`, `RAG Query API`?**
  _High betweenness centrality (0.273) - this node is a cross-community bridge._
- **Why does `HybridRetriever` connect `Adaptive API Layer` to `Adaptive RAG Engine`, `LLM Providers & Strategy Mix`, `Streamlit Web UI`, `Audio Playback`, `Knowledge Graph (GraphRAG)`, `ReAct Agent Internals`, `BGE-M3 Embedding`?**
  _High betweenness centrality (0.133) - this node is a cross-community bridge._
- **Why does `SearchResult` connect `Adaptive RAG Engine` to `LLM Providers & Strategy Mix`, `Streamlit Web UI`, `Knowledge Graph (GraphRAG)`, `Adaptive API Layer`, `ReAct Agent Internals`, `BGE-M3 Embedding`, `RAG Query API`?**
  _High betweenness centrality (0.089) - this node is a cross-community bridge._
- **Are the 217 inferred relationships involving `Document` (e.g. with `AgentStep` and `AgentResult`) actually correct?**
  _`Document` has 217 INFERRED edges - model-reasoned connections that need verification._
- **Are the 161 inferred relationships involving `SearchResult` (e.g. with `AgentStep` and `AgentResult`) actually correct?**
  _`SearchResult` has 161 INFERRED edges - model-reasoned connections that need verification._
- **Are the 94 inferred relationships involving `AgentResult` (e.g. with `QueryComplexity` and `AdaptiveRAG`) actually correct?**
  _`AgentResult` has 94 INFERRED edges - model-reasoned connections that need verification._
- **Are the 80 inferred relationships involving `HybridRetriever` (e.g. with `QueryComplexity` and `AdaptiveRAG`) actually correct?**
  _`HybridRetriever` has 80 INFERRED edges - model-reasoned connections that need verification._