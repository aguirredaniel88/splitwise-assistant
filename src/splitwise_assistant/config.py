from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM API keys (optional - users can provide via UI)
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None  # Free Llama models

    # Splitwise auth (optional - for WhatsApp endpoint only, web users provide via UI)
    splitwise_oauth_access_token: Optional[str] = None
    splitwise_api_key: Optional[str] = None

    # Default LLM provider and models
    llm_provider: str = "anthropic"  # "anthropic" | "openai"
    anthropic_model: str = "claude-sonnet-4-6"
    openai_model: str = "gpt-4o"

    # Twilio WhatsApp
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_whatsapp_number: str = "whatsapp:+14155238886"

    session_ttl_minutes: int = 60
    port: int = 8000


settings = Settings()
