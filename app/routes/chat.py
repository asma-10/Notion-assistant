from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.query import build_query_engine
from app.schemas import QueryRequest, QueryResponse

router = APIRouter()


async def token_stream(question: str, top_k: int):
    """
    Async generator that yields tokens as SSE events.
    LlamaIndex's streaming response exposes .async_response_gen() —
    an async iterator that yields tokens as the LLM produces them.
    """
    engine = build_query_engine(similarity_top_k=top_k)

    # aquery() is the async version — always use this inside async FastAPI routes
    streaming_response = await engine.aquery(question)

    async for token in streaming_response.async_response_gen():
        yield {"data": token}

    yield {"data": "[DONE]"}  # Tells the client the stream is finished

@router.post("/stream")
async def chat_stream(request: QueryRequest):
    """
    Query your documents with a streaming response.
    Returns tokens one by one via Server-Sent Events (SSE).

    Client-side usage example:
        const es = new EventSource('/chat/stream');
        es.onmessage = (e) => {
            if (e.data !== '[DONE]') appendToken(e.data)
        }
    """
    return EventSourceResponse(token_stream(request.question, request.top_k))

@router.post("/", response_model=QueryResponse)
async def chat(request: QueryRequest):
    try:
        engine = build_query_engine(similarity_top_k=request.top_k)
        response = engine.query(request.question)

        sources = list({
            node.metadata.get("file_name", "unknown")
            for node in response.source_nodes
        })

        return QueryResponse(answer=str(response), source_nodes=sources)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))