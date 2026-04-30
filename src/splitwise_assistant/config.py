from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str

    # Splitwise auth (passed through to MCP server via env)
    splitwise_oauth_access_token: Optional[str] = None
    splitwise_api_key: Optional[str] = None

    # OpenAI
    openai_api_key: Optional[str] = None

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
