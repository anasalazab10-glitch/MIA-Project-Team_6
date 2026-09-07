"""
Configuration module for the LEDGER Orchestrator Service.
Reads service endpoints and operational settings from environment variables.
"""

import os
from pathlib import Path
from pydantic import BaseModel


class Settings(BaseModel):
    # Downstream service URLs
    # Defaults use localhost ports for standalone development,
    # or container names when running inside Docker network.
    doc_processor_url: str = os.getenv("DOC_PROCESSOR_URL", "http://localhost:8002")
    retrieval_url: str = os.getenv("RETRIEVAL_URL", "http://localhost:8000")
    reasoning_url: str = os.getenv("REASONING_URL", "http://localhost:8001")
    validator_url: str = os.getenv("VALIDATOR_URL", "http://localhost:8004")

    # Orchestrator server config
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8003"))

    # Timeouts in seconds
    doc_processor_timeout: float = float(os.getenv("DOC_PROCESSOR_TIMEOUT", "120.0"))
    retrieval_timeout: float = float(os.getenv("RETRIEVAL_TIMEOUT", "30.0"))
    reasoning_timeout: float = float(os.getenv("REASONING_TIMEOUT", "60.0"))
    validator_timeout: float = float(os.getenv("VALIDATOR_TIMEOUT", "15.0"))

    # Data persistence directory for dashboard & document catalog
    data_dir: str = os.getenv("ORCHESTRATOR_DATA_DIR", str(Path(__file__).parent / "data"))

    # Langfuse Observability (optional)
    langfuse_host: str | None = os.getenv("LANGFUSE_HOST", None)
    langfuse_public_key: str | None = os.getenv("LANGFUSE_PUBLIC_KEY", None)
    langfuse_secret_key: str | None = os.getenv("LANGFUSE_SECRET_KEY", None)


settings = Settings()
