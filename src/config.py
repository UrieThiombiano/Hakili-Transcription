from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Google Gemini — provider principal de transcription
    google_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # OpenAI — filet de secours (GPT-5) si Gemini échoue
    openai_api_key: str = ""
    openai_model: str = "gpt-5"

    # Stockage des sessions (images ingérées, exports générés)
    runs_dir: str = "./runs"


settings = Settings()
