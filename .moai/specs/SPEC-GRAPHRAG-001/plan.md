---
id: SPEC-GRAPHRAG-001
version: 1.0.0
type: plan
created: 2026-04-19
updated: 2026-04-19
---

# SPEC-GRAPHRAG-001 — Implementation Plan

본 문서는 `spec.md` 에 정의된 요구사항을 구현하기 위한 작업 분해, 우선순위,
담당 Agent, 파일 변경 목록, 리스크를 정리한다.

**기본 원칙**: 시간 추정은 작성하지 않는다. 우선순위(High/Medium/Low)와
단계 순서(T1 → T8)만 사용한다.

## 1. 기술 접근 (Technical Approach)

### 1.1 아키텍처 결정

- **그래프 저장소**: NetworkX `MultiDiGraph` 유지. Neo4j 전환은 범위 제외.
- **영속화**: `pickle` (primary) + `GraphML` (fallback). 버전 민감성 완화.
- **ANTONYM 추출**: `nltk.corpus.wordnet` 사용. 미설치 시 graceful skip.
- **주입 지점**: `src/adaptive.py` 의 Complex 분기 한 지점에만 주입.
  Simple/Moderate 경로는 수정하지 않는다.
- **API 스타일**: 기존 FastAPI 라우터 규약을 따른다 (`src/api/` 하위).
- **UI**: Streamlit 기본 컴포넌트 + `pyvis` (optional).

### 1.2 호환성 원칙

- `src/graph.py` 의 기존 public 심볼은 그대로 유지한다.
- `graph_rag_fusion` 시그니처는 추가 키워드 인자(`max_graph_terms` 등)만
  허용한다 (기존 호출자 호환).
- `/api/adaptive` 는 `use_graph` 파라미터를 선택적 필드로 추가한다
  (기본값은 Complex 에서 `true`, Simple/Moderate 에서 `false`).

## 2. 작업 분해 (Task Decomposition)

### T1. ANTONYM 추출 추가 — `Priority: High`

- **담당**: `expert-backend`
- **의존**: 없음
- **파일**:
  - `src/graph.py` (수정): `_extract_antonyms_wordnet()` 헬퍼 추가 및
    `build_from_documents` 내부 호출.
  - `pyproject.toml` (수정): `nltk` 의존성 추가.
  - `scripts/download_wordnet.py` (신규, 선택): WordNet 코퍼스 다운로드 헬퍼.
- **내용**:
  - `nltk.corpus.wordnet.synsets(word)` → `lemma.antonyms()` 순회.
  - `try/except LookupError` 로 코퍼스 미설치 graceful skip.
  - 최소 단어 길이 `≥ 3`.
- **요구사항 매핑**: U1, E7, O3.

### T2. CO_OCCURS 상한 및 엣지 버짓 경고 — `Priority: High`

- **담당**: `expert-backend`
- **의존**: 없음 (T1 과 병렬 가능)
- **파일**: `src/graph.py` (수정).
- **내용**:
  - `build_from_documents` 에 `max_cooccurrence_per_doc: int = 10`
    파라미터 추가.
  - 문서별 엣지 추가 루프에서 카운터 적용.
  - 전체 edge_count 가 `100_000` 초과 시 `logger.warning`.
- **요구사항 매핑**: U6, U7, UB4.

### T3. 영속화 (save / load / auto-load) — `Priority: High`

- **담당**: `expert-backend`
- **의존**: T1, T2 완료 후 (데이터 안정화)
- **파일**:
  - `src/graph.py` (수정):
    - `WordKnowledgeGraph.save(path: str = "data/graph.pkl")`.
    - `WordKnowledgeGraph.load(path: str) -> WordKnowledgeGraph`.
    - `WordKnowledgeGraph.export_graphml(path: str)`.
    - `__init__` 에서 `auto_load: bool = True` 시 기존 파일 자동 로드.
  - `data/.gitkeep` (신규): 런타임 디렉터리.
  - `.gitignore` (수정): `data/graph.pkl`, `data/graph.graphml` 추가.
- **내용**:
  - pickle 로드 실패 시 GraphML fallback.
  - 둘 다 실패 시 빈 그래프 + WARNING (UB2).
- **요구사항 매핑**: E2, UB1, UB2, O2.

### T4. Indexer 통합 (Qdrant 이후 그래프 빌드) — `Priority: High`

- **담당**: `expert-backend`
- **의존**: T1, T2, T3 완료
- **파일**: `src/indexer.py` (수정).
- **내용**:
  - 기존 인덱싱 성공 분기 말미에 다음 추가:
    1. `WordKnowledgeGraph(auto_load=False)` 생성.
    2. `build_from_documents(graph, documents, max_cooccurrence_per_doc=10)`.
    3. `graph.save("data/graph.pkl")` 및
       `graph.export_graphml("data/graph.graphml")`.
  - 1000 노드마다 progress log (E8).
- **요구사항 매핑**: E1, E8.

### T5. AdaptiveRAG Complex 전략에 Fusion 주입 — `Priority: High`

- **담당**: `expert-backend`
- **의존**: T3 완료
- **파일**:
  - `src/adaptive.py` (수정):
    - `AdaptiveRAG.__init__` 에 `graph: WordKnowledgeGraph | None = None`
      수락.
    - `_answer_complex()` 내부 호출 직전에 `use_graph` 가 True 이고
      `graph.node_count > 0` 이면 `graph_rag_fusion` 호출.
    - Simple/Moderate 경로는 수정 금지.
  - `src/graph.py` (수정): `graph_rag_fusion` 호출 시그니처에
    `max_graph_terms: int = 5` 추가 (기본값).
- **요구사항 매핑**: U3, U4, E3, S2, UB3.

### T6. API 엔드포인트 — `Priority: Medium`

- **담당**: `expert-backend`
- **의존**: T3, T5 완료
- **파일**:
  - `src/api/graph.py` (신규 라우터).
  - `src/api/main.py` (수정): 라우터 등록.
  - `src/api/adaptive.py` (수정 추정, 파일 위치 확인 후 조정):
    요청 스키마에 `use_graph: bool = True` 추가.
- **내용**:
  - `GET /api/graph/related/{word}` 쿼리 파라미터
    `relation_type: Literal["SYNONYM", "ANTONYM", "DERIVED_FROM", "CO_OCCURS", "SAME_CATEGORY"]`.
  - `GET /api/graph/stats` 응답:
    `{ "node_count": int, "edge_count": int, "per_relation": {...} }`.
  - 그래프 빈 경우 `404`가 아닌 `200 {"related": []}` 응답 (UB3 정책과 일관).
- **요구사항 매핑**: E4, E5, UB3.

### T7. Streamlit "지식 그래프" 탭 — `Priority: Medium`

- **담당**: `expert-frontend`
- **의존**: T6 완료 (API 호출 또는 직접 그래프 객체 사용 가능)
- **파일**: `web/app.py` (수정).
- **내용**:
  - 최상단 tab list 에 `"지식 그래프"` 추가.
  - 검색 단어 입력 → `GET /api/graph/related/{word}` 호출.
  - `pyvis` 설치 시 HTML 임베드, 미설치 시 `st.graphviz_chart`.
  - 하단에 `GET /api/graph/stats` 기반 통계 패널 표시.
- **요구사항 매핑**: E6, O1.

### T8. 테스트 및 회귀 검증 — `Priority: High`

- **담당**: `expert-testing`
- **의존**: T1 ~ T7 완료
- **파일**:
  - `tests/test_graph.py` (확장):
    - `test_build_extracts_antonyms_via_wordnet`.
    - `test_build_skips_antonyms_when_wordnet_unavailable` (모킹).
    - `test_cooccurrence_cap_respected`.
    - `test_save_load_roundtrip`.
    - `test_graphml_fallback_on_pickle_corruption`.
  - `tests/test_adaptive_graph.py` (신규):
    - `test_complex_path_uses_graph_when_enabled`.
    - `test_simple_path_ignores_graph_flag`.
    - `test_moderate_path_ignores_graph_flag`.
    - `test_complex_path_skips_fusion_on_empty_graph`.
  - `tests/test_api_graph.py` (신규):
    - `test_get_related_returns_filtered_words`.
    - `test_get_stats_returns_counts`.
    - `test_adaptive_use_graph_false_bypasses_fusion`.
- **요구사항 매핑**: 모든 E*/S*/UB* 검증.

## 3. 파일 변경 요약

| Path | Action | Task |
|---|---|---|
| `src/graph.py` | Modify (보존·확장) | T1, T2, T3, T5 |
| `src/indexer.py` | Modify | T4 |
| `src/adaptive.py` | Modify | T5 |
| `src/api/graph.py` | Create | T6 |
| `src/api/main.py` | Modify | T6 |
| `src/api/adaptive.py` | Modify (조건부) | T6 |
| `web/app.py` | Modify | T7 |
| `tests/test_graph.py` | Modify | T8 |
| `tests/test_adaptive_graph.py` | Create | T8 |
| `tests/test_api_graph.py` | Create | T8 |
| `pyproject.toml` | Modify | T1 |
| `.gitignore` | Modify | T3 |
| `data/.gitkeep` | Create | T3 |
| `scripts/download_wordnet.py` | Create (optional) | T1 |

## 4. 설정 추가 (Configuration)

- `pyproject.toml` 의 `[tool.poetry.dependencies]` (또는 PEP 621 `project.dependencies`)
  에 다음 추가:
  - `nltk = "^3.9"` (또는 현재 호환 최신).
  - `pyvis = { version = "^0.3", optional = true }` (optional-dependencies 에
    `viz` extra 로 정의).
- 새 설정 키:
  - `graph.path`: 기본 `"data/graph.pkl"`.
  - `graph.enable_antonym`: 기본 `true`.
  - `graph.max_cooccurrence_per_doc`: 기본 `10`.
  - `graph.edge_budget_warn`: 기본 `100_000`.
- 현재 프로젝트 설정 위치(예: `.env` 또는 `src/models.py` Settings)에 맞춰
  실제 반영 위치는 T3 착수 시 확정한다.

## 5. 환경 셋업 단계 (Setup Steps)

1. `.venv/Scripts/python.exe -m pip install nltk pyvis`.
2. WordNet 코퍼스 다운로드 (약 10 MB, 1회):
   `.venv/Scripts/python.exe -c "import nltk; nltk.download('wordnet')"`.
3. (옵션) `scripts/download_wordnet.py` 로 자동화.
4. 기존 Qdrant 인덱스가 있는 경우 `python -m src index` 재실행하여
   그래프 캐시 생성.

## 6. 마일스톤 (Priority 순서)

- **Milestone A (High — 코어 그래프 완성도)**: T1, T2, T3.
  → 그래프 자체가 정상 동작하고 저장·로드가 가능해짐.
- **Milestone B (High — 파이프라인 연결)**: T4, T5.
  → 인덱싱 후 자동 빌드 + Complex 전략에서 실제 사용됨.
- **Milestone C (Medium — 외부 인터페이스)**: T6, T7.
  → API 및 UI 노출.
- **Milestone D (High — 품질 게이트)**: T8.
  → 커버리지 85 % 이상, 회귀 테스트 전부 통과.

## 7. 리스크 레지스터 (Risk Register)

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | DERIVED_FROM 접미사 매칭 오탐 (false positive) | Medium | Low | 최소 stem 길이 `≥ 3` 유지, 결과를 샘플 검수 후 추후 개선 (이 SPEC 범위 밖). |
| R2 | NetworkX pickle 버전 비호환 | Medium | High | GraphML 보조 포맷 병행 저장, pickle 실패 시 fallback (UB1). |
| R3 | WordNet 다운로드(~10 MB) 미설치 환경 | Medium | Medium | graceful skip + WARNING (E7). `scripts/download_wordnet.py` 제공. |
| R4 | 24 k 문서에서 그래프 빌드 시간 증가 | Medium | Medium | 1000 노드마다 progress log (E8). CO_OCCURS 상한 (U6) 으로 속도 확보. |
| R5 | CO_OCCURS 엣지 폭발 (O(n²)) | High | High | `max_cooccurrence_per_doc = 10` (U6) + 전체 `100_000` 경고 (U7, UB4). |
| R6 | Complex 경로 지연 증가 (NFR1 위반) | Medium | Medium | `max_graph_terms` 기본 5로 제한, `use_graph=false` 긴급 우회 (S2). |
| R7 | pyvis 미설치 환경에서 UI 실패 | Low | Low | `st.graphviz_chart` fallback (O1). |
| R8 | 기존 호출자의 `graph_rag_fusion` 시그니처 깨짐 | Low | High | 신규 파라미터는 키워드+기본값으로만 추가 (U3). |

## 8. 구현 후 검토 (Post-Implementation Review)

T1 ~ T8 완료 후 다음 항목을 PR 체크리스트에 포함한다.

- [ ] `ruff check src/graph.py src/adaptive.py src/indexer.py` 클린.
- [ ] `mypy src/graph.py src/adaptive.py` 클린.
- [ ] 커버리지 리포트: 신규 코드 ≥ 85 %.
- [ ] v1.2 `tests/test_agent.py` 전체 통과.
- [ ] v1.3 `tests/test_adaptive.py` 전체 통과.
- [ ] Complex 쿼리 P95 지연 측정 (NFR1).
- [ ] ANTONYM 샘플 재현율 리포트 (NFR4).
