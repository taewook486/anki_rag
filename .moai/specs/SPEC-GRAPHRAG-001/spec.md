---
id: SPEC-GRAPHRAG-001
version: 1.0.0
status: Planned
created: 2026-04-19
updated: 2026-04-19
author: manager-spec
priority: High
issue_number: null
---

# SPEC-GRAPHRAG-001 — GraphRAG v2.0 Full-Stack 통합

## HISTORY

- v1.0.0 (2026-04-19): 초안 작성. 7개 Gap을 AdaptiveRAG Complex 전략에 한정하여 통합. Neo4j는 범위 제외.

## 1. 목표 (Goal)

`src/graph.py` 에 이미 구현된 `WordKnowledgeGraph` 및 `graph_rag_fusion` 을
**실제 인덱싱·검색 파이프라인, 영속 저장, API, Streamlit UI** 까지 연결하여
Anki RAG 시스템의 v2.0 GraphRAG 기능을 Production-ready 상태로 만든다.

핵심 원칙:

- 기존 `src/graph.py` 코드는 **보존·확장** 한다 (재작성 금지).
- GraphRAG Fusion 은 **AdaptiveRAG Complex 전략에서만** 동작한다.
- Simple (Dense-only), Moderate (Hybrid RRF) 전략은 **변경 없이** 유지한다.
- Neo4j 백엔드는 이 SPEC 의 범위가 아니다 (향후 SPEC 에서 다룬다).

## 2. 배경 (Context)

- 설계서 v2 섹션 6 (Adaptive RAG) 및 섹션 9 (GraphRAG) 참조: `doc/설계서_v2.md`.
- 현재 상태: `src/graph.py` 에 `RelationType`, `WordKnowledgeGraph`,
  `build_from_documents`, `graph_rag_fusion` 함수가 존재한다.
- 미완 영역:
  1. ANTONYM 추출이 `build_from_documents` 에서 채워지지 않는다.
  2. `src/indexer.py` 가 Qdrant 인덱싱 후 그래프를 빌드하지 않는다.
  3. `graph_rag_fusion` 이 어떤 검색 경로에서도 호출되지 않는다.
  4. 그래프 영속화 (save/load) 가 없다.
  5. API 엔드포인트가 없다.
  6. Streamlit UI 탭이 없다.
  7. CO_OCCURS 에 문서당 상한이 없어 엣지 폭발 위험이 있다.

## 3. 범위 (Scope)

### 3.1 IN SCOPE (포함)

- `src/graph.py` 확장: ANTONYM (WordNet), save/load, CO_OCCURS 상한.
- `src/indexer.py` 수정: 인덱싱 성공 후 그래프 빌드·저장 호출.
- `src/adaptive.py` 수정: Complex 전략 경로에만 GraphRAG Fusion 주입.
- `src/api/` 신규 라우트: `/api/graph/related/{word}`, `/api/graph/stats`,
  기존 `/api/adaptive` 에 `use_graph` 파라미터 추가.
- `web/app.py` 확장: "지식 그래프" 탭 추가 (pyvis 또는 graphviz_chart).
- `data/graph.pkl` (pickle) 및 `data/graph.graphml` (안전망) 영속화.
- `tests/test_graph.py` 확장 및 신규 `tests/test_adaptive_graph.py`.

### 3.2 EXCLUSIONS — What NOT to Build (제외)

- [EXCLUDED] Neo4j 백엔드 전환 (NetworkX 유지).
- [EXCLUDED] 다국어 그래프 (영어 전용).
- [EXCLUDED] 덱 간 교차 엣지 pruning 로직.
- [EXCLUDED] 그래프 시각화 스타일 커스터마이징 (기본 제공만).
- [EXCLUDED] Simple/Moderate 전략에 그래프 주입 (Complex 전용).
- [EXCLUDED] 실시간 그래프 업데이트 (인덱싱 시점에만 rebuild).
- [EXCLUDED] 그래프 편집 UI (읽기 전용 탐색만).

## 4. 요구사항 (Requirements — EARS Format)

### 4.1 Ubiquitous (항시 유효)

- **U1**: The system SHALL store the knowledge graph as a NetworkX `MultiDiGraph`
  in `src/graph.py` and SHALL NOT introduce Neo4j in this SPEC.
- **U2**: The system SHALL preserve existing public APIs of
  `WordKnowledgeGraph` (`add_word`, `add_relation`, `get_related`,
  `get_synonyms`, `get_antonyms`, `get_derived_words`, `node_count`,
  `edge_count`).
- **U3**: The system SHALL keep `graph_rag_fusion` signature backward
  compatible with its current callers.
- **U4**: The system SHALL apply graph-augmented retrieval ONLY on the
  AdaptiveRAG Complex strategy path. Simple and Moderate paths SHALL remain
  unchanged.
- **U5**: The system SHALL write documentation-facing outputs in Korean and
  keep all code identifiers in English.
- **U6**: The system SHALL cap per-document co-occurrence edges at
  `max_cooccurrence_per_doc = 10`.
- **U7**: The system SHALL log a WARNING when total graph edges exceed
  `100_000`.

### 4.2 Event-Driven (WHEN)

- **E1**: WHEN `src/indexer.py` finishes a successful Qdrant indexing run,
  the system SHALL invoke `build_from_documents` and persist the resulting
  graph to `data/graph.pkl` and `data/graph.graphml`.
- **E2**: WHEN `WordKnowledgeGraph.__init__` is called and
  `data/graph.pkl` exists, the system SHALL auto-load the graph from disk.
- **E3**: WHEN `AdaptiveRAG.answer()` is invoked with strategy = Complex and
  `use_graph = True`, the system SHALL call `graph_rag_fusion` before
  generation.
- **E4**: WHEN the client calls
  `GET /api/graph/related/{word}?relation_type=<TYPE>`, the system SHALL
  return related words filtered by the requested `RelationType`.
- **E5**: WHEN the client calls `GET /api/graph/stats`, the system SHALL
  return `node_count`, `edge_count`, and per-`RelationType` edge counts.
- **E6**: WHEN the Streamlit "지식 그래프" tab is opened, the system SHALL
  render a related-words explorer seeded by the latest user query.
- **E7**: WHEN `nltk` or the WordNet corpus is unavailable, the system
  SHALL skip ANTONYM extraction, log a WARNING, and continue graph build
  using the remaining relation types.
- **E8**: WHEN the graph build runs on the full dataset, the system SHALL
  emit a progress log every `1000` nodes processed.

### 4.3 State-Driven (WHILE)

- **S1**: WHILE the graph is being built, the system SHALL hold
  intermediate state in memory only and SHALL NOT serve
  `/api/graph/related/{word}` from partial state.
- **S2**: WHILE `use_graph` is `False` on the `/api/adaptive` request body,
  the system SHALL bypass `graph_rag_fusion` regardless of strategy.
- **S3**: WHILE the graph file on disk is older than the Qdrant collection
  timestamp, the system SHALL log an INFO-level staleness notice at startup.

### 4.4 Unwanted Behavior (IF … THEN)

- **UB1**: IF the pickle load fails (version mismatch or corruption),
  THEN the system SHALL fall back to `data/graph.graphml` and log an
  ERROR with the failing path.
- **UB2**: IF both pickle and GraphML load fail, THEN the system SHALL
  initialize an empty `WordKnowledgeGraph` and log a WARNING instructing
  the user to re-run indexing.
- **UB3**: IF the Complex path receives `use_graph = True` but the graph
  is empty (`node_count == 0`), THEN the system SHALL bypass fusion and
  log an INFO message; it SHALL NOT raise.
- **UB4**: IF `build_from_documents` would emit more than `100_000` total
  edges, THEN the system SHALL continue but log a WARNING indicating the
  edge budget has been exceeded.
- **UB5**: IF the DERIVED_FROM suffix matcher produces a stem shorter
  than `3` characters, THEN the system SHALL reject that edge.

### 4.5 Optional (WHERE)

- **O1**: WHERE `pyvis` is installed, the Streamlit tab SHALL render an
  interactive HTML graph; otherwise it SHALL fall back to
  `st.graphviz_chart`.
- **O2**: WHERE the configuration key `graph.path` is set, the system
  SHALL use that path for persistence; otherwise it SHALL default to
  `data/graph.pkl`.
- **O3**: WHERE `nltk` WordNet is installed, the system SHALL populate
  ANTONYM edges via `wordnet.lemma.antonyms()`.

## 5. 비기능 요구사항 (Non-Functional)

- **NFR1 (성능)**: Complex 질의 P95 응답시간 증가는 기존 대비 `+150 ms` 이하.
- **NFR2 (품질)**: 신규 코드 테스트 커버리지 `≥ 85 %`.
- **NFR3 (회귀)**: v1.2 (agent), v1.3 (adaptive) 테스트 전부 통과.
- **NFR4 (안정성)**: ANTONYM 재현율 ≥ `60 %` (20개 샘플 단어 기준).
- **NFR5 (유지보수)**: `ruff check` 및 `mypy src/graph.py src/adaptive.py`
  클린 상태 유지.

## 6. Traceability (설계서 연계)

- 설계서 v2 §6 (Adaptive RAG): Complex 전략 분기 지점에 Fusion 주입.
  → REQ U4, E3, S2, UB3.
- 설계서 v2 §9 (GraphRAG): 관계 타입 정의와 Fusion 알고리즘.
  → REQ U1, U2, U3, E1–E8.
- 설계서 v2 §9.3 (영속화): pickle + GraphML 이중화.
  → REQ E1, E2, UB1, UB2.

## 7. 용어 (Glossary)

- **Fusion**: 벡터 검색 결과와 그래프 기반 후보 단어의 가중 병합.
- **Complex 전략**: `src/adaptive.py` 의 `QueryClassifier` 가 질의를
  복합 추론 대상으로 분류한 경우 실행되는 ReAct Agent 경로.
- **WordNet**: Princeton 의 영어 의미 사전. `nltk.corpus.wordnet` 로 접근.
