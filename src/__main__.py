"""CLI - 명령줄 인터페이스"""

import os

import click
from pathlib import Path
from dotenv import load_dotenv

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

    # 2. 임베딩
    click.echo("임베딩 시작...")
    embedder = BGEEmbedder()
    embeddings = embedder.embed_batch(all_documents)
    click.echo(f"  -> {len(embeddings)}개 문서 임베딩 완료")

    # 3. 인덱싱
    click.echo("Qdrant 인덱싱 시작...")
    qdrant_location = os.getenv("QDRANT_LOCATION", "./qdrant_data")
    indexer = QdrantIndexer(location=qdrant_location)
    indexer.create_collection(recreate=True)
    indexer.upsert(all_documents, embeddings)
    click.echo("인덱싱 완료!")


@cli.command()
@click.argument("query")
@click.option("--source", help="source 필터")
@click.option("--top-k", default=10, help="반환할 결과 수")
@click.option("--play-audio", is_flag=True, help="오디오 재생")
def search(query: str, source: str, top_k: int, play_audio: bool):
    """단순 검색"""
    retriever = HybridRetriever(location=os.getenv("QDRANT_LOCATION", "./qdrant_data"))
    results = retriever.search(query, top_k=top_k, source_filter=source)

    for result in results:
        doc = result.document
        click.echo(f"\n[{result.rank}.] {doc.word} - {doc.meaning}")
        if doc.pronunciation:
            click.echo(f"발음: {doc.pronunciation}")
        if doc.example:
            click.echo(f"예문: {doc.example}")
        click.echo(f"점수: {result.score:.3f} | 출처: {doc.source}")

        if play_audio and doc.audio_path:
            player = AudioPlayer()
            player.play(doc.audio_path)


@cli.command()
@click.argument("question")
@click.option("--source", help="source 필터")
@click.option("--top-k", default=5, help="검색할 문서 수")
@click.option("--play-audio", is_flag=True, help="오디오 재생")
def query(question: str, source: str, top_k: int, play_audio: bool):
    """RAG 질의"""
    retriever = HybridRetriever(location=os.getenv("QDRANT_LOCATION", "./qdrant_data"))
    rag = RAGPipeline(retriever)

    answer = rag.query(question, top_k=top_k, source_filter=source)
    click.echo(f"\n답변:\n{answer}")

    if play_audio:
        # 첫 번째 검색 결과의 오디오 재생
        results = retriever.search(question, top_k=1, source_filter=source)
        if results and results[0].document.audio_path:
            player = AudioPlayer()
            player.play(results[0].document.audio_path)


@cli.command()
def chat():
    """대화형 모드"""
    retriever = HybridRetriever(location=os.getenv("QDRANT_LOCATION", "./qdrant_data"))
    rag = RAGPipeline(retriever)
    player = AudioPlayer()

    click.echo("대화형 모드 (종료: Ctrl+C)")
    click.echo("-" * 40)

    try:
        while True:
            question = click.prompt("\n질문")

            answer = rag.query(question)
            click.echo(f"\n답변:\n{answer}")

            # 오디오 재생 여부 확인
            results = retriever.search(question, top_k=1)
            if results and results[0].document.audio_path:
                if click.confirm("발음 듣기?"):
                    player.play(results[0].document.audio_path)

    except KeyboardInterrupt:
        click.echo("\n종료합니다.")


if __name__ == "__main__":
    cli()
