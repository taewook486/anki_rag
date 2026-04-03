"""멀티링구얼 쿼리 성능 검증 — 한국어 뜻으로 영어 단어 검색

인덱싱된 데이터에서 무작위 100개 단어를 추출하고,
한국어 뜻을 쿼리로 → top 5에 원래 영어 단어가 포함되는지 검증한다.
"""

import json
import os
import random
import sys
import time

from dotenv import load_dotenv

load_dotenv()

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qdrant_client import QdrantClient

from src.retriever import HybridRetriever


def extract_test_pairs(qdrant_location: str, collection_name: str, sample_size: int = 100):
    """Qdrant에서 무작위 단어-뜻 쌍을 추출한다."""
    if qdrant_location.startswith("http"):
        client = QdrantClient(location=qdrant_location)
    else:
        client = QdrantClient(path=qdrant_location)

    # 컬렉션 전체 문서 수 확인
    info = client.get_collection(collection_name)
    total = info.points_count
    print(f"컬렉션 '{collection_name}' 총 문서 수: {total}")

    # scroll로 전체 문서의 payload 가져오기
    all_points = []
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_points.extend(points)
        if next_offset is None:
            break
        offset = next_offset

    # meaning이 있는 문서만 필터링
    valid_points = []
    for p in all_points:
        payload = p.payload or {}
        word = payload.get("word", "").strip()
        meaning = payload.get("meaning", "").strip()
        if word and meaning:
            valid_points.append({"word": word, "meaning": meaning, "source": payload.get("source", "")})

    print(f"유효한 단어-뜻 쌍: {len(valid_points)}개")

    # 무작위 추출 (중복 단어 제거 후)
    seen_words = set()
    unique_points = []
    for p in valid_points:
        word_lower = p["word"].lower().strip()
        if word_lower not in seen_words:
            seen_words.add(word_lower)
            unique_points.append(p)

    print(f"고유 단어 수: {len(unique_points)}개")

    if len(unique_points) < sample_size:
        print(f"경고: 고유 단어가 {sample_size}개 미만이므로 {len(unique_points)}개로 테스트합니다.")
        sample_size = len(unique_points)

    random.seed(42)  # 재현 가능한 무작위 추출
    samples = random.sample(unique_points, sample_size)
    return samples


def run_evaluation(samples: list[dict], retriever: HybridRetriever, top_k: int = 5):
    """한국어 뜻으로 검색하여 top_k 안에 원래 단어가 포함되는지 검증한다."""
    results = []
    hit_count = 0
    total = len(samples)

    for i, sample in enumerate(samples, 1):
        word = sample["word"]
        meaning = sample["meaning"]
        source = sample["source"]

        # 한국어 뜻으로 검색
        search_results = retriever.search(query=meaning, top_k=top_k)
        found_words = [r.document.word.lower().strip() for r in search_results]
        target = word.lower().strip()
        hit = target in found_words

        if hit:
            hit_count += 1
            rank = found_words.index(target) + 1
        else:
            rank = -1

        result = {
            "word": word,
            "meaning": meaning,
            "source": source,
            "hit": hit,
            "rank": rank,
            "top_results": found_words[:top_k],
        }
        results.append(result)

        # 진행 상황 출력
        status = "O" if hit else "X"
        if i % 10 == 0 or i == total:
            print(f"[{i:3d}/{total}] 적중률: {hit_count}/{i} ({hit_count/i*100:.1f}%) | {status} {word} <- \"{meaning[:30]}\"")

    return results, hit_count


def print_report(results: list[dict], hit_count: int, top_k: int):
    """결과 리포트를 출력한다."""
    total = len(results)
    hit_rate = hit_count / total * 100 if total > 0 else 0

    print("\n" + "=" * 70)
    print(f"멀티링구얼 쿼리 성능 검증 결과")
    print("=" * 70)
    print(f"테스트 쿼리 수: {total}")
    print(f"성공 기준: top {top_k} 안에 기대 단어 포함")
    print(f"적중 수: {hit_count} / {total}")
    print(f"적중률: {hit_rate:.1f}%")

    # 순위별 분포
    rank_dist = {}
    for r in results:
        if r["hit"]:
            rank_dist[r["rank"]] = rank_dist.get(r["rank"], 0) + 1

    if rank_dist:
        print(f"\n순위별 분포:")
        for rank in sorted(rank_dist.keys()):
            bar = "#" * rank_dist[rank]
            print(f"  {rank}위: {rank_dist[rank]:3d}건 {bar}")

    # 실패 목록
    misses = [r for r in results if not r["hit"]]
    if misses:
        print(f"\n실패 목록 ({len(misses)}건):")
        print(f"{'단어':<20} {'뜻 (쿼리)':<40} {'실제 top 결과'}")
        print("-" * 100)
        for r in misses:
            top_str = ", ".join(r["top_results"][:3])
            meaning_short = r["meaning"][:38]
            print(f"{r['word']:<20} {meaning_short:<40} {top_str}")

    # source별 적중률
    source_stats = {}
    for r in results:
        src = r["source"] or "unknown"
        if src not in source_stats:
            source_stats[src] = {"total": 0, "hit": 0}
        source_stats[src]["total"] += 1
        if r["hit"]:
            source_stats[src]["hit"] += 1

    if len(source_stats) > 1:
        print(f"\n출처별 적중률:")
        for src, stats in sorted(source_stats.items()):
            rate = stats["hit"] / stats["total"] * 100 if stats["total"] > 0 else 0
            print(f"  {src:<20} {stats['hit']:3d}/{stats['total']:3d} ({rate:.1f}%)")

    print("=" * 70)
    return hit_rate


def main():
    qdrant_location = os.getenv("QDRANT_LOCATION", "./qdrant_data")
    collection_name = "anki_rag"
    top_k = 5
    sample_size = 100

    print(f"Qdrant 위치: {qdrant_location}")
    print(f"테스트 설정: 무작위 {sample_size}개, top {top_k} 기준\n")

    # 1단계: 테스트 쌍 추출
    print("1단계: 테스트 데이터 추출...")
    samples = extract_test_pairs(qdrant_location, collection_name, sample_size)

    # 2단계: retriever 초기화
    print("\n2단계: HybridRetriever 초기화...")
    start = time.time()
    retriever = HybridRetriever(location=qdrant_location, collection_name=collection_name)
    print(f"초기화 완료 ({time.time() - start:.1f}초)\n")

    # 3단계: 검증 실행
    print("3단계: 검증 실행...")
    start = time.time()
    results, hit_count = run_evaluation(samples, retriever, top_k=top_k)
    elapsed = time.time() - start
    print(f"검증 완료 ({elapsed:.1f}초, 쿼리당 {elapsed/len(samples):.2f}초)\n")

    # 4단계: 리포트
    hit_rate = print_report(results, hit_count, top_k)

    # 결과 JSON 저장
    output_path = os.path.join(os.path.dirname(__file__), "eval_multilingual_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {"sample_size": len(samples), "top_k": top_k, "seed": 42},
                "summary": {"total": len(samples), "hit_count": hit_count, "hit_rate": round(hit_rate, 2)},
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n상세 결과 저장: {output_path}")


if __name__ == "__main__":
    main()
