from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.config import settings

""" User question  →  Embed with Ollama  →  Similarity search in Qdrant  →  Top-K chunks  →  Ollama LLM synthesizes answer """

def build_query_engine(similarity_top_k: int = 4) -> RetrieverQueryEngine:
    """
    Build a query engine that:
    - Connects to the existing Qdrant collection (no re-indexing)
    - Embeds the user question with Ollama (nomic-embed-text)
    - Retrieves the top-K most similar chunks
    - Passes them to the Ollama LLM (llama3.2) to synthesize a final answer

    similarity_top_k=4 retrieves 4 chunks per query.
    Increase for broader context, decrease for speed.
    """

    client = QdrantClient(url=settings.QDRANT_URL)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.QDRANT_COLLECTION_NAME
    )

    embeder = OllamaEmbedding(
        model_name=settings.EMBED_MODEL,
        base_url=settings.OLLAMA_BASE_URL
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model= embeder
    )

    retriever = VectorIndexRetriever(
        similarity_top_k=similarity_top_k,
        index=index
    )

    llm = Ollama(
        model=settings.LLM_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        request_timeout=120.0
    )

    synthesizer = get_response_synthesizer(llm=llm, streaming=True)

    return RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=synthesizer
    )


