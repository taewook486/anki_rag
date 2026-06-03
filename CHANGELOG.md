# Changelog

모든 주요 변경 사항이 이 파일에 기록됩니다.

## [v2.1.0] - 2026-05-31

기말 평가 제출 사이클. 사용자 시연 가능 패키지로의 안정화.

### Added

- **LLM 프로바이더 다중화**
  - `AnthropicProvider` 구현체 추가 (anthropic SDK 0.105.2)
  - `create_provider()` 환경변수 우선순위: `ANTHROPIC_API_KEY` → `LLM_API_KEY`
  - 기본 시연 모델: `claude-sonnet-4-6` (Claude API)
  - 기존 GLM/OpenAI 호환 폴백 보존

- **Streamlit 지식 그래프 탭 (4번째 페이지)**
  - 6개 메트릭 카드: 노드/엣지/관계별 분포
  - Plotly 방사형 네트워크 시각화 (관계 타입 색상 코드)
  - `_RELATION_STYLE` 매핑: SYNONYM/ANTONYM/DERIVED_FROM/CO_OCCURS/SAME_CATEGORY
  - 빠른 예시 버튼 6개 (abandon/terminate/accept/begin/quibble/rescind)
  - 좌측 차트 + 우측 관계 목록 (한국어 라벨)

- **Qdrant 공유 클라이언트 DI**
  - `src/api/deps.py`: 프로세스 전역 단일 `QdrantClient` (threading.Lock 더블 체크)
  - `reset_qdrant_client()` 테스트·셧다운 훅
  - `HybridRetriever`, `QdrantIndexer`에 키워드 인자 `client` 추가 (DI)
  - search/index/agent 라우트가 공유 클라이언트 사용

- **기말 제출 패키지**
  - `.env.example`: 환경변수 템플릿 (Anthropic + OpenAI 호환 + Qdrant)
  - `제출_안내.md`: 교수 측 실행 가이드 9장 구성
    - 한국어 Windows cp949 인코딩 함정 + 해결책 A/B
    - 빠른 실행 3단계, 시연 시나리오 5종, 문제 해결 표
  - 압축 패키지: `anki_rag_submit.zip` (397 MB, qdrant_data 사전 빌드 포함)

- **문서**
  - `doc/설계서_final.md`: 기말 최종판 (12장, [업데이트] 12개 + [신규] 3개, 약 22~24페이지)
  - `doc/기말_동작검증_보고서.md`: 종단간 검증 보고서 (그룹 1·2 자가 평가)
  - HWP 편집본 → 마크다운 역동기화 (pandoc + 자동 잡음 정리)
  - README.md 강화: mermaid 아키텍처 다이어그램 2개 + 데모 응답 샘플 3종 + 학술 출처 6편

- **테스트**
  - `tests/test_api_deps.py`: Qdrant DI 9건 (싱글톤/스레드안전/리셋/메모리모드)
  - `tests/test_indexer.py`, `tests/test_retriever.py`: 외부 client 주입 시나리오

### Fixed

- `/api/adaptive` 라우트가 별도 `HybridRetriever`를 생성하여 다른 라우트와 Qdrant 로컬 경로를 이중 점유, "Storage folder ... already accessed" 500 에러 → 공유 retriever 싱글톤 사용
- `QdrantIndexer` 인덱싱 중 검색 라우트가 점유한 락과 충돌하던 문제 → DI 패턴 도입
- Streamlit "지식 그래프" 탭 예시 버튼 클릭 시 `StreamlitAPIException: graph_word cannot be modified after widget instantiated` → 보조 키 `_graph_word_pending` + `st.rerun()` 패턴
- pandoc docx→md 변환 시 자동 헤딩 ID(`{#xxx-1}`), 마크다운 이스케이프(`\[신규\]`, `\|`, `\~`), em dash 변환 잡음 → 자동 정리 스크립트

### Dependencies

- `anthropic>=0.105.2`: Claude API 클라이언트 (선택)
- `plotly>=6.7.0`: Streamlit 지식 그래프 시각화

### Quality Metrics

- 신규 테스트: 9건 (test_api_deps 4 + 외부 client 주입 5)
- 누적 회귀: 175+ 통과
- 인덱스 규모: 73,410 포인트 (Qdrant 컬렉션 `anki_rag`)
- 그래프 규모: 22,285 노드 / 30,670 엣지 (ANTONYM 7,160)

---

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
