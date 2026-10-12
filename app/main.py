from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.ingest import router as ingest_router
from app.routes.chat import router as chat_router

app = FastAPI(
    title = "RAG system",
    description = "simple RAG system using FastAPI and python and Ollama",
    version= "1.0.0" 
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(ingest_router, prefix="/ingest", tags=["Ingestion"])
app.include_router(chat_router, prefix="/chat", tags=["Chat"])


@app.get("/health")
async def health():
    return {"status": "ok"}