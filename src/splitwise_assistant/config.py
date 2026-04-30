from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str
    splitwise_mcp_path: str = "/app/splitwise-mcp/app.py"

    # Splitwise auth (passed through to MCP server via env)
    splitwise_oauth_access_token: Optional[str] = None
    splitwise_api_key: Optional[str] = None

    # Twilio WhatsApp
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_whatsapp_number: str = "whatsapp:+14155238886"

    session_ttl_minutes: int = 60
    port: int = 8000


settings = Settings()
