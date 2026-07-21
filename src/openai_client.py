"""
OpenAIClient — filet de secours (GPT-5) pour la transcription, utilisé
uniquement quand GeminiTranscriptionClient échoue.
"""
from __future__ import annotations

import base64
import json
import logging
import re
from pathlib import Path
from typing import Any

from openai import OpenAI
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.config import settings
from src.models import AIResponse, TranscriptionResult

logger = logging.getLogger(__name__)

_MEDIA_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


def _repair_json(text: str) -> str:
    stack: list[str] = []
    in_string = False
    i = 0
    while i < len(text):
        c = text[i]
        if c == "\\" and in_string:
            i += 2
            continue
        if c == '"':
            in_string = not in_string
        elif not in_string:
            if c in ("{", "["):
                stack.append(c)
            elif c in ("}", "]") and stack:
                stack.pop()
        i += 1
    close_map = {"[": "]", "{": "}"}
    return text + "".join(close_map[c] for c in reversed(stack))


def _parse_json_response(raw: str) -> AIResponse:
    text = raw.strip()
    m = _JSON_FENCE.search(text)
    if m:
        text = m.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    text = _TRAILING_COMMA.sub(r"\1", text)

    for candidate in (text, _repair_json(text)):
        candidate = _TRAILING_COMMA.sub(r"\1", candidate)
        try:
            data = json.loads(candidate)
            validated = TranscriptionResult(**data)
            return AIResponse(success=True, data=validated, confidence=0.85, raw_response=raw[:2000], error=None)
        except (json.JSONDecodeError, Exception):
            continue

    logger.error("GPT-5 — JSON irréparable. Début : %.300s", raw)
    return AIResponse(
        success=False, data=None, confidence=0.0,
        raw_response=raw[:500], error=f"JSON invalide retourné par GPT-5. Début : {raw[:200]}",
    )


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
        self._client = OpenAI(api_key=settings.openai_api_key, timeout=90.0)
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
    def transcribe(self, doc_id: str, image_paths: list[Path]) -> AIResponse:
        logger.info("[%s] GPT-5 transcription (fallback) — modèle : %s | pages : %d",
                    doc_id, settings.openai_model, len(image_paths))
        schema_example = (
            '{\n'
            f'  "doc_id": "{doc_id}",\n'
            '  "global_quality": "good",\n'
            '  "pages": [\n'
            '    {"page_number": 1, "content": "...", "uncertainties": [], "confidence": 0.9}\n'
            '  ]\n}'
        )
        prompt = (
            f"{self._transcription_prompt}\n\ndoc_id : {doc_id}\n\n"
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
                max_completion_tokens=8192,
            )
            raw = response.choices[0].message.content or ""
            logger.info("GPT-5 transcription OK — tokens: %d in / %d out",
                        response.usage.prompt_tokens, response.usage.completion_tokens)
            return _parse_json_response(raw)
        except Exception as e:
            logger.error("GPT-5 transcribe erreur (doc_id=%s) : %s", doc_id, e)
            return AIResponse(success=False, data=None, confidence=0.0, raw_response="", error=str(e))
