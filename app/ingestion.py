import os
from pathlib import Path

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.config import settings


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=settings.QDRANT_URL)


def ensure_collection_exists(client: QdrantClient) -> None:

    vector_size = 768  # nomic-embed-text dimension

    existing = [c.name for c in client.get_collections().collections]
    if settings.QDRANT_COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

def ingest_documents(file_paths: list[str]) -> dict:
    """ Raw file  →  Load (parse text)  →  Chunk (split into nodes)  →  Embed with Ollama  →  Store in Qdrant """
    documents = SimpleDirectoryReader(input_files=file_paths).load_data()

    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)

    embeder = OllamaEmbedding(
        base_url=settings.OLLAMA_BASE_URL,
        model_name=settings.EMBED_MODEL
    )

    client = QdrantClient(url=settings.QDRANT_URL)
    ensure_collection_exists(client)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.QDRANT_COLLECTION_NAME
    )

    storage = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex(
        embed_model=embeder,
        storage_context=storage,
        nodes=nodes
    )

    return {"nodes_ingested": len(nodes), "files": [Path(f).name for f in file_paths]}

