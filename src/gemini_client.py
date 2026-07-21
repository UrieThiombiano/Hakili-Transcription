"""
GeminiTranscriptionClient — transcription de manuscrit via Google Gemini Flash.
Provider principal. En cas d'échec, le pipeline bascule sur OpenAIClient (GPT-5).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from PIL import Image
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.config import settings
from src.json_utils import parse_transcription_json
from src.models import AIResponse

logger = logging.getLogger(__name__)

_MAX_OUTPUT_TOKENS = 16384

_GEMINI_FALLBACKS = [
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.0-flash",
    "gemini-1.5-flash-002",
    "gemini-1.5-flash",
]


def _is_retryable_gemini(exc: BaseException) -> bool:
    msg = str(exc)
    if "PerDay" in msg or "PerDayPer" in msg:
        return False
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower()


_retry_gemini = retry(
    retry=retry_if_exception(_is_retryable_gemini),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=6, max=60),
    reraise=True,
)


class GeminiTranscriptionClient:
    """Transcription multimodale de manuscrit via Gemini Flash."""

    def __init__(self) -> None:
        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY manquante. Ajoutez-la dans .env.")
        self._client = genai.Client(api_key=settings.google_api_key)
        self._model = self._probe_model(settings.gemini_model)
        self._transcription_prompt = self._load_prompt("transcription_prompt.md")
        logger.info("GeminiTranscriptionClient initialisé (modèle=%s)", self._model)

    def _probe_model(self, configured: str) -> str:
        candidates = [configured] + [m for m in _GEMINI_FALLBACKS if m != configured]
        for model in candidates:
            try:
                self._client.models.generate_content(
                    model=model,
                    contents="test",
                    config=types.GenerateContentConfig(max_output_tokens=1),
                )
                if model != configured:
                    logger.warning("Gemini : modèle '%s' indisponible — fallback → '%s'", configured, model)
                return model
            except Exception as e:
                if "404" in str(e) or "NOT_FOUND" in str(e):
                    continue
                logger.warning("Gemini probe '%s' erreur non-404 : %s — modèle conservé.", model, e)
                return configured
        logger.error("Gemini : aucun modèle disponible parmi %s.", candidates)
        return configured

    def _load_prompt(self, filename: str) -> str:
        prompt_path = Path(__file__).parent.parent / "prompts" / filename
        return prompt_path.read_text(encoding="utf-8")

    def transcribe(self, doc_id: str, image_paths: list[Path], page_offset: int = 0) -> AIResponse:
        try:
            return self._transcribe_batch(doc_id, image_paths, page_offset)
        except Exception as e:
            logger.error("Gemini transcription abandonnée après retries (doc_id=%s) : %s", doc_id, e)
            return AIResponse(success=False, data=None, confidence=0.0, raw_response="", error=str(e))

    @_retry_gemini
    def _transcribe_batch(self, doc_id: str, image_paths: list[Path], page_offset: int = 0) -> AIResponse:
        first_page = page_offset + 1
        page_hint = (
            f"\nCes images sont les pages {first_page} à {page_offset + len(image_paths)} du document."
            if page_offset > 0 else ""
        )
        schema_example = (
            '{\n'
            f'  "doc_id": "{doc_id}",\n'
            '  "global_quality": "good",\n'
            '  "pages": [\n'
            f'    {{"page_number": {first_page}, "content": "...", "tables": [], "uncertainties": [], "confidence": 0.9}}'
            + (f',\n    {{"page_number": {first_page + 1}, "content": "...", "tables": [], "uncertainties": [], "confidence": 0.9}}'
               if len(image_paths) > 1 else "")
            + '\n  ]\n}'
        )
        prompt = (
            f"{self._transcription_prompt}\n\ndoc_id : {doc_id}{page_hint}\n\n"
            "Retourne UNIQUEMENT un objet JSON valide sans aucune balise markdown, "
            "avec cette structure exacte (une entrée dans `pages` par image fournie, dans l'ordre) :\n"
            f"{schema_example}"
        )

        contents: list[Any] = [prompt]
        for path in image_paths:
            contents.append(Image.open(path))

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    max_output_tokens=_MAX_OUTPUT_TOKENS,
                    temperature=0.1,
                ),
            )
            raw = response.text.strip()
            result = parse_transcription_json(raw, page_offset, "Gemini")
            if result.success:
                logger.info("Gemini transcription OK (doc_id=%s, pages=%d, offset=%d)",
                            doc_id, len(image_paths), page_offset)
            return result

        except Exception as e:
            if _is_retryable_gemini(e):
                logger.warning("Gemini 429 (doc_id=%s) — tenacity va réessayer : %s", doc_id, e)
                raise
            logger.error("Erreur Gemini (doc_id=%s) : %s", doc_id, e)
            return AIResponse(success=False, data=None, confidence=0.0, raw_response="", error=str(e))
