---
id: SPEC-GRAPHRAG-001
version: 1.0.0
type: acceptance
created: 2026-04-19
updated: 2026-04-19
---

# SPEC-GRAPHRAG-001 — Acceptance Criteria

본 문서는 Given-When-Then 시나리오, 엣지 케이스, 품질 게이트, DoD 를 정의한다.
각 AC 는 `spec.md` 의 요구사항 ID 와 매핑된다.

## 1. 기능 AC (Functional Acceptance Criteria)

### AC-F1. 인덱싱 후 그래프가 빌드되고 저장된다

- **Given**: 사용자가 `data/*.apkg` 파일을 준비하고 Qdrant 가 비어 있다.
- **When**: `python -m src index` 를 실행한다.
- **Then**:
  - `data/graph.pkl` 이 생성된다.
  - `data/graph.graphml` 이 함께 생성된다.
  - 로그에 `Graph build complete: {node_count} nodes, {edge_count} edges` 가 출력된다.
  - 1000 노드마다 progress log 가 최소 1회 이상 발생한다.
- **매핑**: E1, E8.

### AC-F2. 앱 재시작 시 그래프가 자동 로드된다

- **Given**: `data/graph.pkl` 이 존재한다.
- **When**: FastAPI 서버를 재시작한다.
- **Then**:
  - `WordKnowledgeGraph` 인스턴스가 비어 있지 않다 (`node_count > 0`).
  - 시작 로그에 `Graph loaded from data/graph.pkl (N nodes)` 출력.
- **매핑**: E2.

### AC-F3. ANTONYM 관계가 WordNet 에서 추출된다

- **Given**: `nltk` 및 WordNet 코퍼스가 설치되어 있고, 입력 문서에 `"good"` 단어가 포함된다.
- **When**: `build_from_documents` 를 실행한다.
- **Then**:
  - 그래프에 `good -[ANTONYM]-> bad` (또는 WordNet 기준 반의어) 엣지가 존재한다.
  - `graph.get_antonyms("good")` 결과가 비어 있지 않다.
- **매핑**: O3.

### AC-F4. AdaptiveRAG Complex 경로에서 Fusion 이 동작한다

- **Given**: Complex 로 분류되는 질의, 그래프가 비어 있지 않다 (`node_count > 0`).
- **When**: `POST /api/adaptive { "query": "...", "use_graph": true }` 호출.
- **Then**:
  - 응답 payload 의 `strategy` 필드가 `"complex"`.
  - 응답 payload 의 `graph_terms` 필드 (또는 동등 필드) 가 비어 있지 않다.
  - `graph_rag_fusion` 호출이 최소 1회 발생한다 (로그 또는 메트릭으로 확인).
- **매핑**: E3, U4.

### AC-F5. Simple/Moderate 경로는 그래프를 사용하지 않는다

- **Given**: Simple 로 분류되는 질의와 Moderate 로 분류되는 질의 각 1건.
- **When**: 동일 엔드포인트 호출, `use_graph: true` 포함.
- **Then**:
  - 두 응답 모두 `graph_terms` 가 비어 있거나 부재하다.
  - 로그에 `graph_rag_fusion` 호출 기록이 없다.
- **매핑**: U4.

### AC-F6. `use_graph=false` 는 항상 Fusion 을 우회한다

- **Given**: Complex 질의, `use_graph=false`.
- **When**: `/api/adaptive` 호출.
- **Then**: 응답이 Fusion 없이 반환되며 로그에 `Graph fusion bypassed (use_graph=false)`.
- **매핑**: S2.

### AC-F7. `/api/graph/related/{word}` 가 관계별 필터링을 지원한다

- **Given**: 그래프에 `run`, `jog` (SYNONYM), `runner` (DERIVED_FROM) 존재.
- **When**: `GET /api/graph/related/run?relation_type=SYNONYM`.
- **Then**:
  - 응답 status `200`.
  - 결과 리스트에 `jog` 포함, `runner` 미포함.
- **매핑**: E4.

### AC-F8. `/api/graph/stats` 가 통계를 반환한다

- **Given**: 그래프에 `1_000` 노드 이상 존재.
- **When**: `GET /api/graph/stats`.
- **Then**: 응답에 다음 필드 존재:
  - `node_count: int`
  - `edge_count: int`
  - `per_relation: { "SYNONYM": int, "ANTONYM": int, "DERIVED_FROM": int, "CO_OCCURS": int, "SAME_CATEGORY": int }`
- **매핑**: E5.

### AC-F9. Streamlit "지식 그래프" 탭이 렌더링된다

- **Given**: Streamlit 앱이 실행 중이고 그래프가 비어 있지 않다.
- **When**: 사용자가 "지식 그래프" 탭을 열고 단어 `"run"` 을 입력.
- **Then**:
  - 인접 단어 리스트가 화면에 표시된다.
  - 통계 패널에 `node_count`, `edge_count` 값이 보인다.
  - `pyvis` 설치 환경에서는 인터랙티브 그래프가 렌더링된다.
- **매핑**: E6, O1.

## 2. 엣지 케이스 AC (Edge Cases)

### AC-E1. WordNet 미설치 환경

- **Given**: `nltk.corpus.wordnet` import 가 `LookupError` 를 낸다.
- **When**: `build_from_documents` 실행.
- **Then**:
  - ANTONYM 엣지 수 = 0.
  - WARNING 로그: `WordNet unavailable, skipping ANTONYM extraction`.
  - 나머지 관계 타입은 정상 생성.
- **매핑**: E7.

### AC-E2. pickle 로드 실패 → GraphML fallback

- **Given**: `data/graph.pkl` 손상, `data/graph.graphml` 정상.
- **When**: 서버 시작.
- **Then**:
  - ERROR 로그: pickle 실패 사유.
  - INFO 로그: `Falling back to GraphML`.
  - 그래프가 GraphML 에서 로드되어 `node_count > 0`.
- **매핑**: UB1.

### AC-E3. 두 파일 모두 실패

- **Given**: `data/graph.pkl`, `data/graph.graphml` 모두 손상.
- **When**: 서버 시작.
- **Then**:
  - WARNING 로그: `Graph files unreadable; start empty. Re-run indexing.`
  - `node_count == 0`, `edge_count == 0`.
  - `/api/graph/stats` 는 `200 { ..., node_count: 0, ... }` 반환.
- **매핑**: UB2.

### AC-E4. 빈 그래프에서 Complex 질의

- **Given**: `node_count == 0`, Complex 질의, `use_graph=true`.
- **When**: `/api/adaptive` 호출.
- **Then**:
  - 예외 발생하지 않음.
  - Fusion 우회, INFO 로그: `Graph empty, skipping fusion`.
  - 응답은 정상 (Complex Agent 결과만 포함).
- **매핑**: UB3.

### AC-E5. 엣지 버짓 초과

- **Given**: 문서 수가 많아 `edge_count > 100_000`.
- **When**: `build_from_documents` 실행.
- **Then**:
  - WARNING 로그: `Edge budget 100000 exceeded: current={N}`.
  - 빌드는 중단되지 않고 완료된다.
- **매핑**: U7, UB4.

### AC-E6. DERIVED_FROM 오탐 제거

- **Given**: 단어 `"ed"` (2글자), 후보 stem 길이 `< 3`.
- **When**: DERIVED_FROM 추출.
- **Then**: 해당 엣지는 생성되지 않는다.
- **매핑**: UB5.

### AC-E7. CO_OCCURS 상한 적용

- **Given**: 단일 문서의 `examples` 에 30개 단어 포함.
- **When**: `build_from_documents(max_cooccurrence_per_doc=10)`.
- **Then**: 해당 문서에서 파생되는 CO_OCCURS 엣지 수 ≤ 10.
- **매핑**: U6.

## 3. 품질 AC (Quality Gate Criteria)

### AC-Q1. 테스트 커버리지

- **Given**: 신규/수정 코드(`src/graph.py`, `src/adaptive.py`, `src/indexer.py`,
  `src/api/graph.py`).
- **When**: `.venv/Scripts/python.exe -m pytest --cov=src --cov-report=term`.
- **Then**: 신규 코드 커버리지 `≥ 85 %`.

### AC-Q2. Linter / Type checker

- **When**: `ruff check src/` 및 `mypy src/graph.py src/adaptive.py`.
- **Then**: 에러 0, 경고는 기존 baseline 이하.

### AC-Q3. 정적 명명 규약

- **When**: 신규 코드 검토.
- **Then**: 식별자·함수명·클래스명·API path 는 전부 영어, 문서화 문자열은 한국어 허용.

## 4. 측정 가능 AC (Measurable)

### AC-M1. 빌드 완주 (24 k 문서)

- **Given**: 24 k 문서 데이터셋.
- **When**: `python -m src index` 실행.
- **Then**:
  - 빌드 실패 없이 완료.
  - 로그에 최소 `24` 회 이상의 progress 메시지 (1000 노드 단위 가정).
- **매핑**: E8, NFR5.

### AC-M2. ANTONYM 재현율 샘플 테스트

- **Given**: 미리 정의된 20개 샘플 단어 리스트 (good, bad, big, small, happy,
  sad, fast, slow, hot, cold, rich, poor, open, close, accept, reject, begin,
  end, love, hate).
- **When**: 각 단어에 대해 `graph.get_antonyms(word)` 호출.
- **Then**: 최소 `12 / 20 = 60 %` 에서 최소 1개 이상의 반의어 반환.
- **매핑**: NFR4.

### AC-M3. Complex 경로 P95 지연

- **Given**: 100회 반복의 Complex 질의 벤치마크.
- **When**: `use_graph=true` 와 `use_graph=false` 를 비교 측정.
- **Then**: `P95(on) - P95(off) ≤ 150 ms`.
- **매핑**: NFR1.

## 5. 회귀 AC (Regression)

### AC-R1. v1.3 Adaptive 회귀

- **When**: `.venv/Scripts/python.exe -m pytest tests/test_adaptive.py`.
- **Then**: 기존 27개 테스트 전부 통과.

### AC-R2. v1.2 Agent 회귀

- **When**: `.venv/Scripts/python.exe -m pytest tests/test_agent.py`.
- **Then**: 기존 30개 테스트 전부 통과.

### AC-R3. 기존 `/api/search`, `/api/query` 동작 유지

- **When**: 기존 엔드포인트 시나리오 재실행.
- **Then**: 응답 스키마 변경 없음, status 200.

### AC-R4. 기존 `graph_rag_fusion` 호출자 호환

- **When**: 기존 호출자 시그니처로 `graph_rag_fusion(vector_results, graph,
  query_word, retriever, relation_type, top_k)` 호출.
- **Then**: 시그니처 에러 없이 동작 (U3).

## 6. Definition of Done (DoD)

SPEC-GRAPHRAG-001 은 다음 조건이 모두 참일 때 "Done" 으로 간주한다.

- [ ] `spec.md` 의 모든 U*/E*/S*/UB*/O* 요구사항이 구현 또는 테스트로 커버됨.
- [ ] §1 기능 AC (AC-F1 ~ AC-F9) 전부 통과.
- [ ] §2 엣지 케이스 AC (AC-E1 ~ AC-E7) 전부 통과.
- [ ] §3 품질 AC (AC-Q1 ~ AC-Q3) 전부 통과.
- [ ] §4 측정 AC (AC-M1 ~ AC-M3) 전부 목표 도달.
- [ ] §5 회귀 AC (AC-R1 ~ AC-R4) 전부 통과.
- [ ] PR 본문에 테스트 결과, 커버리지 리포트, ANTONYM 샘플 재현율 측정값 첨부.
- [ ] `doc/설계서_v2.md` §6, §9 와 구현 간 불일치 검토 완료 (필요 시 sync).
- [ ] Streamlit UI 수동 smoke test 완료 (AC-F9).
