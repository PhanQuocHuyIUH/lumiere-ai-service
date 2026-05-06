from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ai_service_key: str = Field(default="change-me", alias="AI_SERVICE_KEY")

    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")

    backend_base_url: str = Field(default="http://localhost:8080", alias="BACKEND_BASE_URL")

    # LLM (chatbot)
    llm_provider: str = Field(default="openai_compatible", alias="LLM_PROVIDER")
    llm_base_url: str = Field(default="", alias="LLM_BASE_URL")
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_model: str = Field(default="", alias="LLM_MODEL")

    # Vector DB (Qdrant)
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="menu_items", alias="QDRANT_COLLECTION")

    # Embedding model (Gemini API — reuses llm_api_key)
    embed_model: str = Field(default="text-embedding-004", alias="EMBED_MODEL")

    # Redis (CTR store + job status)
    redis_url: str = Field(default="redis://localhost:6379", alias="REDIS_URL")

    # Trained model directory
    model_dir: str = Field(default="models", alias="MODEL_DIR")


settings = Settings()
