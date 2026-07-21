"""
GeminiTranscriptionClient — transcription de manuscrit via Google Gemini Flash.
Provider principal. En cas d'échec, le pipeline bascule sur OpenAIClient (GPT-5).
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from PIL import Image
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.config import settings
from src.models import AIResponse, PageTranscription, TranscriptionResult

logger = logging.getLogger(__name__)

_MAX_OUTPUT_TOKENS = 8192

_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")

_GEMINI_FALLBACKS = [
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.0-flash",
    "gemini-1.5-flash-002",
    "gemini-1.5-flash",
]


def _item_to_str(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("zone", "area", "location", "description", "desc", "content"):
            if key in item:
                return str(item[key])
        for v in item.values():
            if v:
                return str(v)
    return str(item)


def _normalize_transcription(data: Any) -> dict:
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return {"doc_id": "unknown", "global_quality": "poor", "pages": []}
    if not isinstance(data, dict):
        return {"doc_id": "unknown", "global_quality": "poor", "pages": []}

    if data.get("global_quality") not in ("good", "medium", "poor"):
        data["global_quality"] = "medium"
    data["doc_id"] = str(data["doc_id"]) if "doc_id" in data else "unknown"

    raw_pages = data.get("pages", [])
    if not isinstance(raw_pages, list):
        raw_pages = []

    normalized: list[dict] = []
    for i, page in enumerate(raw_pages):
        if isinstance(page, str):
            normalized.append({
                "page_number": i + 1,
                "content": page,
                "uncertainties": [],
                "confidence": 0.5,
            })
        elif isinstance(page, dict):
            if not isinstance(page.get("content"), str):
                page["content"] = str(page.get("content") or "")
            val = page.get("uncertainties", [])
            if not isinstance(val, list):
                val = [val] if val else []
            page["uncertainties"] = [_item_to_str(x) for x in val if x is not None]
            page.setdefault("page_number", i + 1)
            page.setdefault("confidence", 0.5)
            normalized.append({
                "page_number": page["page_number"],
                "content": page["content"],
                "uncertainties": page["uncertainties"],
                "confidence": page["confidence"],
            })

    data["pages"] = normalized
    return {"doc_id": data["doc_id"], "global_quality": data["global_quality"], "pages": data["pages"]}


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

    def transcribe(self, doc_id: str, image_paths: list[Path]) -> AIResponse:
        try:
            return self._transcribe_batch(doc_id, image_paths)
        except Exception as e:
            logger.error("Gemini transcription abandonnée après retries (doc_id=%s) : %s", doc_id, e)
            return AIResponse(success=False, data=None, confidence=0.0, raw_response="", error=str(e))

    @_retry_gemini
    def _transcribe_batch(self, doc_id: str, image_paths: list[Path]) -> AIResponse:
        schema_example = (
            '{\n'
            f'  "doc_id": "{doc_id}",\n'
            '  "global_quality": "good",\n'
            '  "pages": [\n'
            '    {"page_number": 1, "content": "...", "uncertainties": [], "confidence": 0.9}'
            + (',\n    {"page_number": 2, "content": "...", "uncertainties": [], "confidence": 0.9}'
               if len(image_paths) > 1 else "")
            + '\n  ]\n}'
        )
        prompt = (
            f"{self._transcription_prompt}\n\ndoc_id : {doc_id}\n\n"
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

            m = _JSON_FENCE.search(raw)
            if m:
                raw = m.group(1).strip()

            start_idx = raw.find("{")
            end_idx = raw.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                raw = raw[start_idx: end_idx + 1]

            raw = _TRAILING_COMMA.sub(r"\1", raw)

            data_dict: dict | None = None
            for candidate in (raw, _repair_json(raw)):
                candidate = _TRAILING_COMMA.sub(r"\1", candidate)
                try:
                    data_dict = json.loads(candidate)
                    break
                except json.JSONDecodeError:
                    continue

            if data_dict is None:
                logger.error("Gemini — JSON irréparable (doc_id=%s). Début : %.300s", doc_id, raw)
                return AIResponse(
                    success=False, data=None, confidence=0.0,
                    raw_response=raw[:500], error="JSON invalide retourné par Gemini.",
                )

            data_dict = _normalize_transcription(data_dict)
            validated = TranscriptionResult(**data_dict)

            logger.info("Gemini transcription OK (doc_id=%s, pages=%d, qualité=%s)",
                        doc_id, len(image_paths), validated.global_quality)
            avg_conf = sum(p.confidence for p in validated.pages) / len(validated.pages) if validated.pages else 0.5
            return AIResponse(success=True, data=validated, confidence=avg_conf, raw_response=raw[:2000], error=None)

        except Exception as e:
            if _is_retryable_gemini(e):
                logger.warning("Gemini 429 (doc_id=%s) — tenacity va réessayer : %s", doc_id, e)
                raise
            logger.error("Erreur Gemini (doc_id=%s) : %s", doc_id, e)
            return AIResponse(success=False, data=None, confidence=0.0, raw_response="", error=str(e))
