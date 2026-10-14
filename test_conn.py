# test_conn.py
from config.config import settings
print("Scheme:", settings.SUPABASE_URL.split("://")[0])   # must print: postgresql

from app.ingestion import get_vector_store
from app.embed import get_embed_model

print("Vector store:", get_vector_store())
vec = get_embed_model().get_query_embedding("hello")
print("Embedding dimension:", len(vec))   # must print 768