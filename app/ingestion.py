from pathlib import Path

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.supabase import SupabaseVectorStore

from config.config import settings
from app.embed import get_embed_model


def get_vector_store() -> SupabaseVectorStore:
    # Creates the vecs.<collection_name> table on first use if it doesn't exist
    return SupabaseVectorStore(
        postgres_connection_string=settings.SUPABASE_URL,
        collection_name=settings.VECTOR_COLLECTION_NAME,
        dimension=settings.EMBED_DIM,
    )


def ingest_documents(file_paths: list[str]) -> dict:
    """ Raw file  →  Load (parse text)  →  Chunk (split into nodes)  →  Embed with Ollama  →  Store in Supabase (pgvector) """
    documents = SimpleDirectoryReader(input_files=file_paths).load_data()

    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)

    storage = StorageContext.from_defaults(vector_store=get_vector_store())

    # Embeds every node with Ollama and inserts them into Supabase
    VectorStoreIndex(
        nodes=nodes,
        storage_context=storage,
        embed_model=get_embed_model(),
    )

    return {"nodes_ingested": len(nodes), "files": [Path(f).name for f in file_paths]}