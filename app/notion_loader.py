from llama_index.core import Document
from llama_index.readers.notion import NotionPageReader

from config.config import settings


def load_notion_documents(page_ids: list[str] | None = None,
                          database_id: str | None = None) -> list[Document]:
    reader = NotionPageReader(integration_token=settings.NOTION_TOKEN)

    page_ids = page_ids or [p.strip() for p in settings.NOTION_PAGE_IDS.split(",") if p.strip()]
    database_id = database_id or settings.NOTION_DATABASE_ID or None

    docs: list[Document] = []
    if page_ids:
        docs += reader.load_data(page_ids=page_ids)
    if database_id:
        docs += reader.load_data(database_ids=[database_id])   # must be a list

    # Stable ids so a page can be replaced on re-sync instead of duplicated
    for d in docs:
        page_id = d.metadata.get("page_id") or d.doc_id
        d.doc_id = f"notion-{page_id}"
        d.metadata["source"] = "notion"
        d.metadata["page_id"] = page_id
        d.excluded_llm_metadata_keys = ["page_id", "source"]
        d.excluded_embed_metadata_keys = ["page_id", "source"]

    # "Week's plan" is a database row, so it can be loaded twice; keep one copy
    unique = {}
    for d in docs:
        unique[d.doc_id] = d
    return list(unique.values())