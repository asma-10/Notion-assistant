from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

class Settings(BaseSettings):
    LLM_MODEL: str = "qwen2:0.5b"
    EMBED_MODEL: str = "nomic-embed-text"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    SUPABASE_URL: str = "postgresql://postgres.nckaviqeghaqjjegsbqp:notion2026_2027@aws-0-eu-west-1.pooler.supabase.com:5432/postgres"   # Postgres connection string from Supabase
    VECTOR_COLLECTION_NAME: str = "documents"
    EMBED_DIM: int = 768 

    model_config = SettingsConfigDict(env_file=ENV_PATH, extra="ignore")

settings = Settings()