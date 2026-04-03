# Product Overview

## Project Name
Anki RAG - 영어 학습 특화 RAG 시스템

## Vision
Anki 플래시카드 데이터를 활용하여 영어 단어, 구동사, 예문을 검색하고 AI 기반 학습 도우미로 활용할 수 있는 시스템

## Problem Statement
- Anki에 축적된 24,472개의 영어 학습 데이터(단어, 구동사, 예문)가 플래시카드 형태로만 활용됨
- 자연어 질의로 관련 단어/예문을 검색하거나, 맥락 기반 학습 질문에 답변받을 수 없음

## Solution
- Anki .apkg 파일과 텍스트 파일을 파싱하여 구조화된 Document로 변환
- BGE-M3 임베딩(Dense + Sparse)으로 벡터화 후 Qdrant에 저장
- Hybrid Search(Dense + Sparse + RRF Fusion)로 정확한 검색 제공
- Claude API를 통한 RAG 기반 자연어 질의응답
- CLI, FastAPI, Streamlit 3가지 인터페이스 제공

## Data Inventory

| 파일 | 포맷 | 카드 수 | source | deck |
|------|------|---------|--------|------|
| toefl_voca_v1.apkg | .anki2 | 5,308 | toefl | TOEFL 영단어 |
| xfer_voca_2022.apkg | .anki2 | 3,262 | xfer | 편입 영단어 2022 |
| _-forvo_-youglish_link.apkg | .anki21 | 1,227 | hacker_toeic | 해커스 토익 |
| --forvo-youglish_link.apkg | .anki21 | 2,438 | hacker_green | 해커스-초록이 |
| _-forvo-youglish_.apkg | .anki21 | 2,237 | phrasal | 구동사 |
| 10000.txt | tab | 10,000 | sentences | 원서 1만 문장 |

**총 24,472개 Document**

## Target Users
- 영어 학습자 (본인 - 개인 프로젝트)

## Key Features

### 1. 데이터 파싱 (Implemented)
- Anki .apkg 파일 파싱 (anki21/anki2 SQLite)
- 탭 구분 텍스트 파일 파싱 (UTF-8 BOM)
- 오디오 파일 추출

### 2. 벡터 검색 (Implemented)
- BGE-M3 Dense(1024차원) + Sparse(SPLADE) 임베딩
- Qdrant Hybrid Search with RRF Fusion
- source/deck 필터링

### 3. RAG 질의응답 (Implemented - API Key Required)
- Claude API 기반 자연어 질의응답
- 검색 결과 컨텍스트 주입
- 스트리밍 응답 지원

### 4. 인터페이스 (Implemented)
- CLI: index, search, query, chat 명령어
- FastAPI: REST API (검색, 질의)
- Streamlit: 웹 UI (검색, 채팅, 관리)

### 5. 오디오 재생 (Implemented)
- 플랫폼별 네이티브 오디오 재생 (Windows/macOS/Linux)

## Current Status
- Core 모듈: 구현 완료 (models, parser, embedder, indexer, retriever, rag, audio)
- Web 레이어: 구현 완료 (FastAPI + Streamlit)
- 테스트: 대부분 stub 상태, 실질적 커버리지 낮음
- AnkiParser SQLite 파싱: stub 상태 (구조만 존재)
- Sparse Search + RRF Fusion: stub 상태

## Constraints
- Python >= 3.11
- BGE-M3 모델 다운로드 필요 (~2GB)
- Qdrant 현재 :memory: 모드 (프로세스 종료 시 데이터 소실)
- ANTHROPIC_API_KEY 필요 (RAG 질의 기능)
