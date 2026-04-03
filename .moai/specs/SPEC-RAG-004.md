# SPEC-RAG-004: Response Caching

**Status**: Completed
**Created**: 2026-04-03
**Author**: [I
**Parent SPEC**: SPEC-RAG-001
**Tech Stack**: Python 3.11+, threading, hashlib, collections.OrderedDict

---

## 1. 개요 (Overview)

Anki RAG 시스템의 RAG 질의 및 검색 응답을 캐싱하여 LLM API 호출 비용을 절감하고
반복 쿼리의 응답 지연 시간을 개선한다.

### 1.1 아키텍처 다이어그램

```
                        +-----------------------+
                        |   CLI / API / Web     |
                        +-----------+-----------+
                                    |
                    +---------------v---------------+
                    |       RAGPipeline.query()      |
                    |  +-------------------------+  |
                    |  | Level 2: Pipeline Cache  |  |
                    |  | (question + filters +    |  |
                    |  |  history -> answer)      |  |
                    |  +------------+------------+  |
                    |               | cache miss     |
                    +---------------v---------------+
                    |    HybridRetriever.search()    |
                    |  +-------------------------+  |
                    |  | Level 1: Search Cache    |  |
                    |  | (query + filters         |  |
                    |  |  -> SearchResult[])      |  |
                    |  +------------+------------+  |
                    |               | cache miss     |
                    +---------------v---------------+
                    |          Qdrant DB             |
                    +-------------------------------+

                    +-------------------------------+
                    |      Cache Invalidation       |
                    |  - indexer.create_collection() |
                    |  - indexer.upsert()            |
                    |  - TTL 만료 (기본 24시간)       |
                    |  - DELETE /api/cache           |
                    |  - Streamlit 관리 UI 버튼      |
                    +-------------------------------+
```

### 1.2 목표

- LLM API 호출 횟수 절감 (동일 쿼리 반복 시 캐시 적중)
- 검색 응답 지연 시간 개선 (임베딩 연산 + Qdrant 조회 생략)
- 외부 의존성 없이 인메모리 LRU 캐시로 구현
- 스레드 안전성 보장 (FastAPI 동시 요청 대응)

### 1.3 범위

- **포함**: 인메모리 LRU 캐시 모듈, 2단계 캐싱 통합, 무효화 전략, 통계 API, 환경변수 설정
- **제외**: 분산 캐시 (Redis 등), 디스크 기반 영속 캐시, 임베딩 벡터 캐시

---

## 2. 환경 및 가정 (Environment & Assumptions)

### 2.1 환경 (Environment)

| 항목 | 값 |
|------|------|
| 런타임 | Python >= 3.11 |
| 프레임워크 | FastAPI (비동기 핸들러, sync 내부 로직) |
| 벡터 DB | Qdrant (in-memory 또는 로컬 파일) |
| LLM | Claude API / OpenAI 호환 API |
| 동시성 모델 | threading (FastAPI 스레드풀) |
| 데이터 규모 | 24,472개 Document |

### 2.2 가정 (Assumptions)

| ID | 가정 | 근거 | 신뢰도 | 위반 시 영향 |
|----|------|------|--------|-------------|
| A-001 | 동일 질문에 대해 동일 컨텍스트가 반환된다 | 인덱스 변경 없으면 Qdrant 결과 동일 | 높음 | 캐시 불일치 |
| A-002 | 캐시 엔트리 1,000개 기준 메모리 사용량이 100MB 미만이다 | 응답 텍스트 평균 2KB 기준 추정 | 중간 | 메모리 초과 |
| A-003 | 단일 프로세스 환경에서 운영된다 | 개인 프로젝트, 단일 uvicorn 워커 | 높음 | 프로세스 간 캐시 불일치 |
| A-004 | 인덱싱은 빈번하지 않다 (일 1회 이하) | 학습 데이터 변경 빈도 낮음 | 높음 | 잦은 캐시 전체 초기화 |
| A-005 | history가 None인 경우만 Pipeline 캐시 대상이다 | history 포함 시 동일 질문이라도 맥락 달라짐 | 높음 | 잘못된 캐시 적중 |

---

## 3. 기능 요구사항 (EARS Format)

### REQ-001: 캐시 모듈 (QueryCache 클래스)

시스템은 **항상** `src/cache.py`에 `QueryCache` 클래스를 제공해야 한다.

- `QueryCache`는 인메모리 LRU 캐시를 구현한다
- `collections.OrderedDict`를 사용하여 LRU 순서를 관리한다
- `threading.Lock`으로 모든 캐시 연산의 스레드 안전성을 보장한다
- 생성자 파라미터: `max_entries: int = 1000`, `ttl_seconds: int = 86400`
- 공개 메서드: `get(key: str)`, `set(key: str, value: Any)`, `clear()`, `stats()`

### REQ-002: 캐시 키 생성

**WHEN** 캐시 조회/저장 요청이 발생하면 **THEN** 시스템은 SHA256 해시 기반 캐시 키를 생성해야 한다.

- 캐시 키 입력: `(question, top_k, source_filter, deck_filter)` 및 추가 직렬화 가능 파라미터
- `on_chunk` 콜백 등 직렬화 불가능한 파라미터는 제외한다
- Level 2 캐시에서는 `history`를 포함하여 키를 생성한다 (단, history=None인 경우만 캐싱)
- 키 생성 함수: `make_cache_key(**kwargs) -> str`을 별도로 정의한다

### REQ-003: Level 1 - 검색 결과 캐시

**WHEN** `HybridRetriever.search()` 호출 시 **THEN** 시스템은 캐시를 먼저 확인해야 한다.

- 캐시 적중 시: Qdrant 조회 및 임베딩 연산 없이 캐시된 `list[SearchResult]`를 반환한다
- 캐시 미스 시: 기존 로직대로 검색을 수행하고 결과를 캐시에 저장한다
- 캐시 키: `(query, top_k, source_filter, deck_filter, exclude_sources, deduplicate)` 조합

### REQ-004: Level 2 - 파이프라인 응답 캐시

**WHEN** `RAGPipeline.query()` 호출 시 **IF** `history`가 `None`이고 `stream`이 `False`이면 **THEN** 시스템은 파이프라인 캐시를 먼저 확인해야 한다.

- 캐시 적중 시: LLM API 호출 없이 캐시된 응답 문자열을 반환한다
- 캐시 미스 시: 기존 로직대로 LLM 호출 후 응답을 캐시에 저장한다
- 캐시 키: `(question, top_k, source_filter, deck_filter)` 조합
- `history`가 `None`이 아닌 경우 또는 `stream=True`인 경우: 캐시를 사용하지 않는다

### REQ-005: 캐시 무효화 - 인덱싱 연동

**WHEN** `QdrantIndexer.create_collection(recreate=True)` 또는 `QdrantIndexer.upsert()` 호출이 완료되면 **THEN** 시스템은 모든 캐시(Level 1 + Level 2)를 전체 초기화해야 한다.

- 인덱스 데이터가 변경되면 기존 캐시 결과가 무효하므로 전체 클리어한다
- 캐시 인스턴스 접근을 위한 모듈 수준 레지스트리 또는 콜백 메커니즘을 제공한다

### REQ-006: 캐시 무효화 - TTL 만료

시스템은 **항상** 캐시 엔트리의 TTL(Time-To-Live)을 검사해야 한다.

- `get()` 호출 시 엔트리의 저장 시각과 현재 시각을 비교한다
- TTL 초과 엔트리는 캐시 미스로 처리하고 해당 엔트리를 제거한다
- 기본 TTL: 86,400초 (24시간), `CACHE_TTL` 환경변수로 설정 가능

### REQ-007: 캐시 무효화 - 수동 API

**WHEN** `DELETE /api/cache` 요청을 수신하면 **THEN** 시스템은 모든 캐시를 즉시 초기화해야 한다.

- 응답: `{"message": "캐시가 초기화되었습니다", "cleared_entries": N}`
- Level 1 + Level 2 캐시 모두 클리어한다

### REQ-008: 캐시 무효화 - Streamlit 관리 UI

**WHEN** 관리 페이지(`show_admin_page`)에서 "캐시 초기화" 버튼을 클릭하면 **THEN** 시스템은 `DELETE /api/cache` API를 호출해야 한다.

- 관리 페이지에 "캐시 초기화" 섹션과 버튼을 추가한다
- 현재 캐시 통계(적중률, 엔트리 수)를 표시한다

### REQ-009: 캐시 통계

시스템은 **항상** 캐시 적중/미스 통계를 제공해야 한다.

- 추적 항목: `hit_count`, `miss_count`, `total_entries`, `hit_rate`
- `GET /api/cache/stats` 엔드포인트로 통계 조회 가능
- `QueryCache.stats()` 메서드가 딕셔너리로 통계를 반환한다
- 적중률 계산: `hit_count / (hit_count + miss_count)` (0 나눗셈 방지)

### REQ-010: 환경변수 설정

시스템은 **항상** 다음 환경변수를 통해 캐시 동작을 설정할 수 있어야 한다.

| 환경변수 | 기본값 | 설명 |
|----------|--------|------|
| `CACHE_ENABLED` | `true` | 캐시 활성화 여부 |
| `CACHE_TTL` | `86400` | TTL (초 단위) |
| `CACHE_MAX_ENTRIES` | `1000` | 최대 캐시 엔트리 수 |

- `CACHE_ENABLED=false`이면 캐시 로직을 완전히 우회한다 (get은 항상 None, set은 무시)
- 잘못된 값이 설정된 경우 기본값으로 폴백하고 경고 로그를 출력한다

---

## 4. 비기능 요구사항

### NFR-001: 성능

- 캐시 적중 시 응답 시간: Level 1 검색 결과 반환 < 1ms
- 캐시 적중 시 응답 시간: Level 2 파이프라인 응답 반환 < 1ms
- 캐시 키 생성 오버헤드: < 0.1ms
- LRU 퇴거 연산 오버헤드: O(1) (OrderedDict 기반)

### NFR-002: 스레드 안전성

- 모든 캐시 읽기/쓰기 연산은 `threading.Lock`으로 보호한다
- Lock 획득 대기 시간이 성능에 미치는 영향을 최소화한다 (lock 범위 최소화)
- 동시 요청 100건에서도 데이터 경합이 발생하지 않아야 한다

### NFR-003: 메모리

- 최대 1,000 엔트리 기준 메모리 사용량 100MB 미만
- LRU 정책으로 max_entries 초과 시 가장 오래된 엔트리 자동 퇴거

### NFR-004: 투명성

- 캐시 존재가 기존 API 계약(입출력 스키마)을 변경하지 않는다
- 캐시 비활성화(`CACHE_ENABLED=false`) 시 기존 동작과 완전히 동일하다
- 캐시 적중/미스 로그를 DEBUG 레벨로 출력한다

---

## 5. 금지 요구사항 (Unwanted)

### NREQ-001: 외부 의존성 금지

시스템은 캐시 구현을 위해 외부 라이브러리(Redis, memcached, diskcache 등)를 사용**하지 않아야 한다**.

### NREQ-002: 스트리밍 캐시 금지

시스템은 `stream=True`인 요청의 응답을 캐시**하지 않아야 한다**.

### NREQ-003: 대화 히스토리 캐시 금지

시스템은 `history`가 `None`이 아닌 요청의 Level 2 파이프라인 응답을 캐시**하지 않아야 한다**.

---

## 6. 파일 구조 및 영향 파일

### 6.1 신규 파일

| 파일 | 설명 |
|------|------|
| `src/cache.py` | QueryCache 클래스, make_cache_key 함수 |
| `src/api/routes/cache.py` | DELETE /api/cache, GET /api/cache/stats 라우트 |
| `tests/test_cache.py` | QueryCache 단위 테스트 |

### 6.2 수정 파일

| 파일 | 변경 내용 |
|------|-----------|
| `src/retriever.py` | Level 1 캐시 통합 (search 메서드 앞단에 캐시 조회/저장) |
| `src/rag.py` | Level 2 캐시 통합 (query 메서드 앞단에 캐시 조회/저장) |
| `src/api/main.py` | cache 라우터 등록 (`app.include_router`) |
| `src/api/routes/index.py` | 인덱싱 완료 후 캐시 무효화 호출 |
| `src/web/app.py` | 관리 페이지에 캐시 통계/초기화 UI 추가 |

---

## 7. 구현 순서 (Priority-Based Milestones)

### Primary Goal: 캐시 코어 모듈

1. `src/cache.py` - `QueryCache` 클래스 구현 (LRU, TTL, 스레드 안전성, 통계)
2. `src/cache.py` - `make_cache_key()` 유틸리티 함수
3. `tests/test_cache.py` - QueryCache 단위 테스트

### Secondary Goal: 캐시 통합

4. `src/retriever.py` - Level 1 캐시 통합 (HybridRetriever.search)
5. `src/rag.py` - Level 2 캐시 통합 (RAGPipeline.query)
6. `src/api/routes/index.py` - 인덱싱 후 캐시 무효화

### Tertiary Goal: API 및 UI

7. `src/api/routes/cache.py` - 캐시 관리 API 엔드포인트
8. `src/api/main.py` - cache 라우터 등록
9. `src/web/app.py` - Streamlit 관리 페이지 캐시 섹션

### Optional Goal: 고급 기능

10. 캐시 워밍업 메커니즘 (자주 쿼리되는 질문 사전 캐싱)
11. 캐시 엔트리별 메모리 사이즈 추적

---

## 8. 수락 기준 (Acceptance Criteria)

### AC-001: 캐시 모듈 기본 동작

```gherkin
Given QueryCache가 max_entries=3, ttl_seconds=60으로 생성되었을 때
When "key1"에 "value1"을 저장하고 "key1"을 조회하면
Then "value1"이 반환된다

Given 3개의 엔트리가 저장된 상태에서
When 4번째 엔트리를 저장하면
Then 가장 오래 사용되지 않은 엔트리가 퇴거되고 총 엔트리 수는 3이다
```

### AC-002: TTL 만료

```gherkin
Given TTL이 1초인 QueryCache에 엔트리가 저장되었을 때
When 2초 후에 해당 엔트리를 조회하면
Then None이 반환된다 (캐시 미스)
And miss_count가 1 증가한다
```

### AC-003: 스레드 안전성

```gherkin
Given 100개의 스레드가 동시에 캐시에 읽기/쓰기하는 상황에서
When 모든 스레드가 완료되면
Then 데이터 경합이나 예외 없이 정상 동작해야 한다
And 최종 엔트리 수가 max_entries를 초과하지 않아야 한다
```

### AC-004: Level 1 캐시 적중

```gherkin
Given 캐시가 활성화된 HybridRetriever가 준비되었을 때
When 동일한 (query, top_k, source_filter, deck_filter)로 search()를 2회 호출하면
Then 첫 번째 호출은 Qdrant를 조회하고 (캐시 미스)
And 두 번째 호출은 캐시에서 반환한다 (캐시 적중)
And 두 번째 호출의 결과가 첫 번째와 동일하다
```

### AC-005: Level 2 캐시 적중

```gherkin
Given 캐시가 활성화된 RAGPipeline이 준비되었을 때
When history=None, stream=False로 동일한 질문을 2회 query()하면
Then 첫 번째 호출은 LLM API를 호출하고 (캐시 미스)
And 두 번째 호출은 캐시에서 반환한다 (캐시 적중)
And LLM API 호출이 1회만 발생한다
```

### AC-006: 스트리밍 및 히스토리 캐시 제외

```gherkin
Given 캐시가 활성화된 RAGPipeline이 준비되었을 때
When stream=True로 query()를 호출하면
Then 캐시를 사용하지 않고 매번 LLM API를 호출한다

Given 캐시가 활성화된 RAGPipeline이 준비되었을 때
When history=[{"role": "user", "content": "..."}]로 query()를 호출하면
Then Level 2 캐시를 사용하지 않고 LLM API를 호출한다
```

### AC-007: 인덱싱 후 캐시 무효화

```gherkin
Given Level 1 및 Level 2 캐시에 엔트리가 존재할 때
When 인덱싱이 완료되면 (upsert 또는 create_collection(recreate=True))
Then 모든 캐시 엔트리가 클리어된다
And 이후 쿼리는 캐시 미스로 처리된다
```

### AC-008: 수동 캐시 초기화 API

```gherkin
Given 캐시에 5개의 엔트리가 존재할 때
When DELETE /api/cache를 호출하면
Then 응답에 cleared_entries=5가 포함된다
And 이후 캐시 통계의 total_entries가 0이다
```

### AC-009: 캐시 통계 API

```gherkin
Given 3회 캐시 적중, 2회 캐시 미스가 발생한 상태에서
When GET /api/cache/stats를 호출하면
Then hit_count=3, miss_count=2, hit_rate=0.6이 반환된다
```

### AC-010: 환경변수 비활성화

```gherkin
Given CACHE_ENABLED=false로 설정되었을 때
When query() 또는 search()를 동일한 파라미터로 2회 호출하면
Then 매번 실제 연산을 수행한다 (캐시를 사용하지 않는다)
And 캐시 통계의 hit_count, miss_count 모두 0이다
```

### AC-011: Streamlit 관리 UI

```gherkin
Given Streamlit 관리 페이지에 접근했을 때
When 페이지가 로드되면
Then "캐시 관리" 섹션이 표시된다
And 현재 캐시 통계 (적중률, 엔트리 수)가 표시된다
And "캐시 초기화" 버튼이 존재한다

When "캐시 초기화" 버튼을 클릭하면
Then DELETE /api/cache가 호출된다
And 성공 메시지가 표시된다
```

---

## 9. 추적성 (Traceability)

| 요구사항 | 관련 파일 | 수락 기준 |
|----------|-----------|-----------|
| REQ-001 | src/cache.py | AC-001 |
| REQ-002 | src/cache.py | AC-001, AC-004, AC-005 |
| REQ-003 | src/retriever.py | AC-004 |
| REQ-004 | src/rag.py | AC-005, AC-006 |
| REQ-005 | src/api/routes/index.py | AC-007 |
| REQ-006 | src/cache.py | AC-002 |
| REQ-007 | src/api/routes/cache.py | AC-008 |
| REQ-008 | src/web/app.py | AC-011 |
| REQ-009 | src/cache.py, src/api/routes/cache.py | AC-009 |
| REQ-010 | src/cache.py | AC-010 |
| NREQ-001 | src/cache.py | - |
| NREQ-002 | src/rag.py | AC-006 |
| NREQ-003 | src/rag.py | AC-006 |
