import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """API settings."""

    # API settings
    API_TITLE: str = "CSM Voice Cloning API"
    API_DESCRIPTION: str = "OpenAI-compatible API for CSM voice cloning"
    API_VERSION: str = "0.1.0"

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Model settings
    DEVICE: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") is not None else "cpu"
    HF_TOKEN: Optional[str] = os.environ.get("HF_TOKEN")

    # File storage
    UPLOAD_DIR: Path = Path("uploads")
    OUTPUT_DIR: Path = Path("outputs")

    # CORS settings
    CORS_ORIGINS: list[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list[str] = ["*"]
    CORS_ALLOW_HEADERS: list[str] = ["*"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Create directories if they don't exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
