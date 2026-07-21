"""
OpenAIClient — filet de secours (GPT-5) pour la transcription, utilisé
uniquement quand GeminiTranscriptionClient échoue.
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

from openai import OpenAI
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.config import settings
from src.json_utils import parse_transcription_json
from src.models import AIResponse

logger = logging.getLogger(__name__)

_MEDIA_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def _is_retryable_openai(exc: BaseException) -> bool:
    try:
        from openai import APIStatusError, APIConnectionError, APITimeoutError
        if isinstance(exc, APITimeoutError):
            return False
        if isinstance(exc, APIStatusError):
            return exc.status_code in (429, 500, 503)
        return isinstance(exc, APIConnectionError)
    except ImportError:
        return False


_retry = retry(
    retry=retry_if_exception(_is_retryable_openai),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    reraise=True,
)


class OpenAIClient:
    """Filet de secours GPT-5 pour la transcription de manuscrit."""

    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY manquante. Ajoutez-la dans .env pour le fallback GPT-5.")
        # Extraction de tableaux structurés sur plusieurs pages = réponse volumineuse
        # + tokens de raisonnement GPT-5 invisibles → nettement plus long que 90s.
        self._client = OpenAI(api_key=settings.openai_api_key, timeout=240.0)
        self._transcription_prompt = self._load_prompt("transcription_prompt.md")
        logger.info("OpenAIClient initialisé (modèle=%s)", settings.openai_model)

    def _load_prompt(self, filename: str) -> str:
        prompt_path = Path(__file__).parent.parent / "prompts" / filename
        return prompt_path.read_text(encoding="utf-8")

    def _media_type(self, path: Path) -> str:
        return _MEDIA_TYPES.get(path.suffix.lower(), "image/jpeg")

    def _encode_image(self, path: Path) -> str:
        return base64.b64encode(path.read_bytes()).decode("utf-8")

    def _image_content(self, path: Path) -> dict[str, Any]:
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{self._media_type(path)};base64,{self._encode_image(path)}"},
        }

    @_retry
    def transcribe(self, doc_id: str, image_paths: list[Path], page_offset: int = 0) -> AIResponse:
        logger.info("[%s] GPT-5 transcription (fallback) — modèle : %s | pages : %d (offset=%d)",
                    doc_id, settings.openai_model, len(image_paths), page_offset)
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
            f'    {{"page_number": {first_page}, "content": "...", "tables": [], "uncertainties": [], "confidence": 0.9}}\n'
            '  ]\n}'
        )
        prompt = (
            f"{self._transcription_prompt}\n\ndoc_id : {doc_id}{page_hint}\n\n"
            "Retourne UNIQUEMENT un objet JSON valide sans aucune balise markdown, "
            "avec cette structure exacte (une entrée dans `pages` par image fournie, dans l'ordre) :\n"
            f"{schema_example}"
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend(self._image_content(p) for p in image_paths)

        try:
            response = self._client.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
                # GPT-5 est un modèle de raisonnement : les tokens de raisonnement (invisibles)
                # sont décomptés de max_completion_tokens. Avec les tableaux structurés, une
                # limite trop basse peut consommer tout le budget en raisonnement et renvoyer
                # un contenu vide — d'où une marge large.
                max_completion_tokens=32768,
            )
            raw = response.choices[0].message.content or ""
            logger.info("GPT-5 transcription OK — tokens: %d in / %d out",
                        response.usage.prompt_tokens, response.usage.completion_tokens)
            if not raw:
                logger.error(
                    "GPT-5 a renvoyé un contenu vide (finish_reason=%s, completion_tokens=%d) — "
                    "probablement tout le budget consommé par le raisonnement interne.",
                    response.choices[0].finish_reason, response.usage.completion_tokens,
                )
            return parse_transcription_json(raw, page_offset, "GPT-5")
        except Exception as e:
            logger.error("GPT-5 transcribe erreur (doc_id=%s) : %s", doc_id, e)
            return AIResponse(success=False, data=None, confidence=0.0, raw_response="", error=str(e))
