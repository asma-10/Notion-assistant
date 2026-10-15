from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question : str = Field(..., min_length=1, max_length=2000)
    top_k : int = Field(default=2, ge=1, le=10)

""" class QueryResponse(BaseModel):
    answer : str
    source_nodes : list[str] = [] """

class IngestResponse(BaseModel):
    message : str
    node_ingested : int
    files : list[str]

class RetrievedChunk(BaseModel):
    rank: int
    score: float | None = None
    source: str
    text: str


class QueryResponse(BaseModel):
    answer: str
    source_nodes: list[RetrievedChunk]   # was: list[str]