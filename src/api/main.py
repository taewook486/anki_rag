"""FastAPI 메인 애플리케이션"""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import search, query, audio, index, cache, agent, adaptive
from src.api.routes import graph as graph_route

app = FastAPI(
    title="Anki RAG API",
    description="영어 학습 특화 RAG 시스템 API",
    version="1.3.0",
)

# CORS 미들웨어 설정 (Streamlit 통신)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 환경용, 프로덕션에서는 제한 필요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(search.router, prefix="/api", tags=["search"])
app.include_router(query.router, prefix="/api", tags=["query"])
app.include_router(audio.router, prefix="/api", tags=["audio"])
app.include_router(index.router, prefix="/api", tags=["index"])
app.include_router(cache.router, prefix="/api", tags=["cache"])
app.include_router(agent.router, prefix="/api", tags=["agent"])
app.include_router(adaptive.router, prefix="/api", tags=["adaptive"])
app.include_router(graph_route.router, prefix="/api", tags=["graph"])


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {"message": "Anki RAG API", "version": "0.1.0", "docs": "/docs"}


@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
