from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

class Settings(BaseSettings):
    LLM_MODEL: str = "qwen2:0.5b"
    EMBED_MODEL: str = "nomic-embed-text"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    SUPABASE_URL: str 
    VECTOR_COLLECTION_NAME: str = "documents"
    EMBED_DIM: int = 768 
    NOTION_TOKEN: str
    NOTION_DATABASE_ID: str
    NOTION_PAGE_IDS: str  # Comma-separated list of Notion page IDs to ingest

    model_config = SettingsConfigDict(env_file=ENV_PATH, extra="ignore")

settings = Settings()