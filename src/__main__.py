"""CLI - 명령줄 인터페이스"""

import os

import click
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from src.parser import AnkiParser, TextParser
from src.embedder import BGEEmbedder
from src.indexer import QdrantIndexer
from src.retriever import HybridRetriever
from src.audio import AudioPlayer
from src.rag import RAGPipeline


@click.group()
def cli():
    """Anki RAG CLI - 영어 학습 특화 RAG 시스템"""
    pass


@cli.command()
@click.option("--data-dir", default="./data", help="데이터 디렉토리 경로")
@click.option("--source", help="특정 source만 인덱싱")
def index(data_dir: str, source: str):
    """데이터 인덱싱"""
    click.echo(f"인덱싱 시작: {data_dir}")

    data_path = Path(data_dir)

    # 1. 파싱
    all_documents = []

    # .apkg 파일 파싱
    for apkg_file in data_path.glob("*.apkg"):
        source_name = apkg_file.stem  # 파일명에서 source 추출
        if source and source_name != source:
            continue

        click.echo(f"파싱 중: {apkg_file.name}")
        parser = AnkiParser()
        try:
            docs = parser.parse_file(str(apkg_file), source=source_name)
            all_documents.extend(docs)
            click.echo(f"  -> {len(docs)}개 문서 파싱 완료")
        except Exception as e:
            click.echo(f"  -> 에러: {e}")

    # .txt 파일 파싱
    for txt_file in data_path.glob("*.txt"):
        click.echo(f"파싱 중: {txt_file.name}")
        parser = TextParser()
        docs = parser.parse_file(str(txt_file), source="sentences", deck="원서 1만 문장")
        all_documents.extend(docs)
        click.echo(f"  -> {len(docs)}개 문서 파싱 완료")

    click.echo(f"총 {len(all_documents)}개 문서 파싱 완료")

    # 2. 임베딩 (BGE-M3 배치 처리)
    click.echo(f"임베딩 시작 ({len(all_documents)}개)...")
    embedder = BGEEmbedder()
    embeddings = embedder.embed_batch(all_documents)
    click.echo(f"  -> {len(embeddings)}개 임베딩 완료")

    # 3. 인덱싱 (배치 단위 진행률 표시)
    click.echo("Qdrant 인덱싱 시작...")
    qdrant_location = os.getenv("QDRANT_LOCATION", "./qdrant_data")
    indexer = QdrantIndexer(location=qdrant_location)
    indexer.create_collection(recreate=True)

    _BATCH = 500
    with tqdm(total=len(all_documents), desc="인덱싱", unit="doc") as pbar:
        for i in range(0, len(all_documents), _BATCH):
            batch_docs = all_documents[i : i + _BATCH]
            batch_embs = embeddings[i : i + _BATCH]
            indexer.upsert(batch_docs, batch_embs)
            pbar.update(len(batch_docs))

    click.echo("인덱싱 완료!")


@cli.command()
@click.argument("query")
@click.option("--source", help="source 필터")
@click.option("--deck", help="deck 필터")
@click.option("--top-k", default=10, help="반환할 결과 수")
@click.option("--play-audio", is_flag=True, help="오디오 재생")
def search(query: str, source: str, deck: str, top_k: int, play_audio: bool):
    """단순 검색"""
    retriever = HybridRetriever(location=os.getenv("QDRANT_LOCATION", "./qdrant_data"))
    results = retriever.search(query, top_k=top_k, source_filter=source, deck_filter=deck)

    for result in results:
        doc = result.document
        click.echo(f"\n[{result.rank}.] {doc.word} - {doc.meaning}")
        if doc.pronunciation:
            click.echo(f"발음: {doc.pronunciation}")
        if doc.example:
            click.echo(f"예문: {doc.example}")
        click.echo(f"점수: {result.score:.3f} | 출처: {doc.source}")

        if play_audio and doc.audio_paths:
            player = AudioPlayer()
            player.play(doc.audio_paths[0])


@cli.command()
@click.argument("question")
@click.option("--source", help="source 필터")
@click.option("--deck", help="deck 필터")
@click.option("--top-k", default=5, help="검색할 문서 수")
@click.option("--play-audio", is_flag=True, help="오디오 재생")
@click.option("--stream", "use_stream", is_flag=True, help="실시간 스트리밍 출력")
def query(question: str, source: str, deck: str, top_k: int, play_audio: bool, use_stream: bool):
    """RAG 질의"""
    retriever = HybridRetriever(location=os.getenv("QDRANT_LOCATION", "./qdrant_data"))
    rag = RAGPipeline(retriever)

    if use_stream:
        click.echo("\n답변:")
        rag.query(
            question,
            top_k=top_k,
            source_filter=source,
            deck_filter=deck,
            stream=True,
            on_chunk=lambda c: click.echo(c, nl=False),
        )
        click.echo()
    else:
        answer = rag.query(question, top_k=top_k, source_filter=source, deck_filter=deck)
        click.echo(f"\n답변:\n{answer}")

    if play_audio and rag.last_results and rag.last_results[0].document.audio_paths:
        player = AudioPlayer()
        player.play(rag.last_results[0].document.audio_paths[0])


_CHAT_MAX_HISTORY_TURNS = 10  # 유지할 최대 대화 쌍 수


@cli.command()
@click.option("--stream", "use_stream", is_flag=True, help="실시간 스트리밍 출력")
def chat(use_stream: bool):
    """대화형 모드"""
    retriever = HybridRetriever(location=os.getenv("QDRANT_LOCATION", "./qdrant_data"))
    rag = RAGPipeline(retriever)
    player = AudioPlayer()

    click.echo("대화형 모드 (종료: Ctrl+C)")
    click.echo("-" * 40)

    history: list[dict] = []

    try:
        while True:
            question = click.prompt("\n질문")

            if use_stream:
                click.echo("\n답변:")
                answer = rag.query(
                    question,
                    history=history,
                    stream=True,
                    on_chunk=lambda c: click.echo(c, nl=False),
                )
                click.echo()
            else:
                answer = rag.query(question, history=history)
                click.echo(f"\n답변:\n{answer}")

            # 대화 히스토리 업데이트 (최근 N쌍 유지)
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})
            if len(history) > _CHAT_MAX_HISTORY_TURNS * 2:
                history = history[-(_CHAT_MAX_HISTORY_TURNS * 2):]

            # 오디오 재생 (last_results 재사용 — 중복 검색 없음)
            if rag.last_results and rag.last_results[0].document.audio_paths:
                if click.confirm("발음 듣기?"):
                    player.play(rag.last_results[0].document.audio_paths[0])

    except KeyboardInterrupt:
        click.echo("\n종료합니다.")


if __name__ == "__main__":
    cli()
