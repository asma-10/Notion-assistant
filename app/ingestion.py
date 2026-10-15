from pathlib import Path

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.supabase import SupabaseVectorStore

from config.config import settings
from app.embed import get_embed_model

from app.notion_loader import load_notion_documents



def get_vector_store() -> SupabaseVectorStore:
    # Creates the vecs.<collection_name> table on first use if it doesn't exist
    return SupabaseVectorStore(
        postgres_connection_string=settings.SUPABASE_URL,
        collection_name=settings.VECTOR_COLLECTION_NAME,
        dimension=settings.EMBED_DIM,
    )


""" def ingest_documents(file_paths: list[str]) -> dict:
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
 """

def ingest_documents_from_docs(documents) -> int:
    """Chunk, embed and store a list of Documents. Returns the node count."""
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)

    vector_store = get_vector_store()

    # Remove old chunks of these documents first, so re-syncing doesn't duplicate
    for doc in documents:
        vector_store.delete(ref_doc_id=doc.doc_id)

    storage = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex(nodes=nodes, storage_context=storage, embed_model=get_embed_model())
    return len(nodes)


def ingest_documents(file_paths: list[str]) -> dict:
    documents = SimpleDirectoryReader(input_files=file_paths).load_data()
    n = ingest_documents_from_docs(documents)
    return {"nodes_ingested": n, "files": [Path(f).name for f in file_paths]}


def ingest_notion(page_ids: list[str] | None = None) -> dict:
    documents = load_notion_documents(page_ids=page_ids)
    if not documents:
        return {"nodes_ingested": 0, "files": []}
    n = ingest_documents_from_docs(documents)
    return {"nodes_ingested": n, "files": [d.doc_id for d in documents]}