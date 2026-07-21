from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class IngestionResult(BaseModel):
    doc_id: str
    total_pages: int
    pages: list[Path]  # Chemins des images extraites, une par page
    output_dir: Path


class Table(BaseModel):
    """Tableau détecté sur une page (ex : liste, tableau de données, grille)."""
    title: str = ""
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)


class PageTranscription(BaseModel):
    page_number: int
    content: str
    tables: list[Table] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class TranscriptionResult(BaseModel):
    doc_id: str
    global_quality: Literal["good", "medium", "poor"]
    pages: list[PageTranscription]


class AIResponse(BaseModel):
    """Enveloppe générique pour une réponse de client IA (Gemini ou GPT-5)."""
    success: bool
    data: Any = None  # TranscriptionResult si success=True
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    raw_response: str = ""
    error: str | None = None
