# Changelog

모든 주요 변경 사항이 이 파일에 기록됩니다.

## [v2.0.0] - 2026-04-19

### Added

- **GraphRAG v2.0 전체 통합** (SPEC-GRAPHRAG-001)
  - WordNet 기반 단어 관계 추출 (SYNONYM, ANTONYM, DERIVATION)
  - 문서 동시성 관계(CO_OCCURS) 자동 빌드
  - 그래프 영속화: pickle + GraphML 이중 저장
  - CO_OCCURS 엣지 문서당 상한 설정 (기본값 10)
  - GraphRAG Fusion: Complex 쿼리 경로에서 자동 적용

- **FastAPI 그래프 라우트**
  - `GET /api/graph/related/{word}`: 관계 타입 필터 지원
  - `GET /api/graph/stats`: 그래프 통계 조회

- **Streamlit 지식 그래프 탭**
  - 인터랙티브 그래프 시각화 (Plotly)
  - 단어 관계 탐색 UI

- **POST /api/adaptive 확장**
  - `use_graph` 파라미터 추가 (GraphRAG 수동 활성화)

### Changed

- `AdaptiveRAG.query()`: Complex 경로에서 GraphRAG Fusion 자동 적용
- `QdrantIndexer.index()`: 인덱싱 완료 후 그래프 자동 빌드 및 저장
- `src/api/routes/`: adaptive.py, graph.py 신규 추가

### Dependencies

- `nltk>=3.9.4`: WordNet 코퍼스 기반 관계 추출

### Testing

- `tests/test_graph.py`: 78개 테스트 신규 추가
- `tests/test_adaptive.py`: GraphRAG Fusion 통합 테스트
- 기존 v1.2 agent 및 v1.3 adaptive 회귀 테스트 전체 통과

### Quality Metrics

- 전체 테스트: 78 → 160 패스 (+82 신규)
- 코드 커버리지:
  - `src/graph.py`: 86%
  - `src/adaptive.py`: 96%
  - `src/indexer.py`: 90%
  - `src/api/routes/graph.py`: 89%

---

## [v1.3] - 2026-04-15

### Added

- **Adaptive RAG 전략**
  - QueryClassifier: 휴리스틱(정규식) + LLM 2단계 분류
  - Simple: Dense-only 검색 (빠름, 정확도 낮음)
  - Moderate: Hybrid RRF 검색 (균형)
  - Complex: Agent ReAct 추론 (느림, 정확도 높음)

- **POST /api/adaptive**: 통합 엔드포인트

### Testing

- `tests/test_adaptive.py`: 27개 테스트

---

## [v1.2] - 2026-04-10

### Added

- **ReAct LearningAgent**
  - AgentStep, AgentResult 타입
  - Self-Correction: 점수 재시도 + 쿼리 재작성
  - Self-RAG: `_needs_retrieval()` - 검색 필요성 판단
  - Corrective RAG: `_is_relevant_result()` - 결과 관련성 평가

### Testing

- `tests/test_agent.py`: 30개 테스트

---

## [v1.1] - 2026-04-05

### Added

- **Hybrid RAG Pipeline**
  - HybridRetriever: Dense + Sparse + RRF Fusion
  - RAGPipeline: Multi-LLM 지원
  - FastAPI 서버 + Streamlit UI
  - Few-shot 프롬프팅

### Features

- Anki 파싱 (.apkg SQLite)
- BGE-M3 임베딩
- Qdrant 벡터 DB
- RRF 점수 융합 (k=60)
- Exact Match 부스팅
- 단어 중복 제거

---

## [v1.0] - 2026-03-20

### Initial Release

- Anki RAG 프로젝트 초기 구성
- CLI 인터페이스
- 기본 검색 기능
