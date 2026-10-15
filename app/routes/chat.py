import json

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from sse_starlette.sse import EventSourceResponse

from app.query import build_query_engine
from app.schemas import QueryRequest, QueryResponse, RetrievedChunk

router = APIRouter()


def to_chunks(source_nodes) -> list[RetrievedChunk]:
    chunks = []
    for i, n in enumerate(source_nodes, start=1):
        md = n.node.metadata
        chunks.append(RetrievedChunk(
            rank=i,
            score=round(n.score, 3) if n.score is not None else None,
            source=md.get("file_name") or md.get("source") or md.get("page_id") or "unknown",
            text=n.node.get_content(),
        ))
    return chunks


async def token_stream(question: str, top_k: int):
    """Yields tokens as SSE events, then the retrieved chunks, then [DONE]."""
    engine = build_query_engine(similarity_top_k=top_k)

    streaming_response = await engine.aquery(question)

    async for token in streaming_response.async_response_gen():
        yield {"data": token}

    # Retrieved chunks, sent as a separate event once the answer is complete
    chunks = [c.model_dump() for c in to_chunks(streaming_response.source_nodes)]
    yield {"event": "sources", "data": json.dumps(chunks)}

    yield {"data": "[DONE]"}


@router.post("/stream")
async def chat_stream(request: QueryRequest):
    """
    Query your documents with a streaming response (Server-Sent Events).
    Default events carry tokens; the "sources" event carries the retrieved chunks.
    """
    return EventSourceResponse(token_stream(request.question, request.top_k))


@router.post("/", response_model=QueryResponse)
async def chat(request: QueryRequest):
    try:
        engine = build_query_engine(similarity_top_k=request.top_k)

        # engine.query is blocking, so run it off the event loop
        response = await run_in_threadpool(engine.query, request.question)

        # With a streaming synthesizer, str() consumes the stream into the full answer
        answer = str(response)

        return QueryResponse(answer=answer, source_nodes=to_chunks(response.source_nodes))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))