from functools import lru_cache

from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.supabase import SupabaseVectorStore

from config.config import settings
from app.embed import get_embed_model

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.prompts import ChatPromptTemplate

from config.prompt_loader import load_system_prompt  # adjust path


""" User question  →  Embed with Ollama  →  Similarity search in Supabase (pgvector)  →  Top-K chunks  →  Ollama LLM synthesizes answer """


def build_qa_prompt(version: str | None = None) -> ChatPromptTemplate:
    return ChatPromptTemplate(
        message_templates=[
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=load_system_prompt(version),
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=(
                    "Context information:\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Question: {query_str}\n"
                    "Answer:"
                ),
            ),
        ]
    )

@lru_cache(maxsize=4)
def build_query_engine(similarity_top_k: int = 4) -> RetrieverQueryEngine:
    """
    Build a query engine that:
    - Connects to the existing Supabase collection (no re-indexing)
    - Embeds the user question with Ollama (nomic-embed-text)
    - Retrieves the top-K most similar chunks
    - Passes them to the Ollama LLM to synthesize a final answer

    Cached, so the Postgres connection is created once, not on every request.
    """

    vector_store = SupabaseVectorStore(
        postgres_connection_string=settings.SUPABASE_URL,
        collection_name=settings.VECTOR_COLLECTION_NAME,
        dimension=settings.EMBED_DIM,
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=get_embed_model(),
    )

    retriever = VectorIndexRetriever(
        similarity_top_k=similarity_top_k,
        index=index,
    )

    llm = Ollama(
        model=settings.LLM_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        request_timeout=120.0,
        context_window=2048,
    )

    synthesizer = get_response_synthesizer(llm=llm, streaming=True, text_qa_template=build_qa_prompt()
)

    return RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=synthesizer,
    )