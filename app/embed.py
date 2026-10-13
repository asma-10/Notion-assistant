from llama_index.embeddings.ollama import OllamaEmbedding

from app.config import settings


def get_embed_model() -> OllamaEmbedding:
    """Used by both ingestion and query so the prefixes always match."""
    return OllamaEmbedding(
        model_name=settings.EMBED_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        query_instruction="search_query: ",
        text_instruction="search_document: ",
    )