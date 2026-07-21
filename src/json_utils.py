"""Parsing et normalisation du JSON de transcription — partagé entre
GeminiTranscriptionClient et OpenAIClient (même schéma de sortie attendu)."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.models import AIResponse, TranscriptionResult

logger = logging.getLogger(__name__)

_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


def repair_json(text: str) -> str:
    """Ferme les accolades/crochets non fermés (JSON tronqué par une limite de tokens)."""
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


def _normalize_table(item: Any) -> dict:
    if not isinstance(item, dict):
        return {"title": "", "headers": [], "rows": []}
    title = str(item.get("title") or "")
    headers = item.get("headers", [])
    headers = [str(h) for h in headers] if isinstance(headers, list) else []
    raw_rows = item.get("rows", [])
    rows: list[list[str]] = []
    if isinstance(raw_rows, list):
        for row in raw_rows:
            if isinstance(row, list):
                rows.append([str(cell) for cell in row])
    return {"title": title, "headers": headers, "rows": rows}


def normalize_transcription(data: Any, page_offset: int = 0) -> dict:
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
                "page_number": page_offset + i + 1,
                "content": page,
                "tables": [],
                "uncertainties": [],
                "confidence": 0.5,
            })
        elif isinstance(page, dict):
            content = page.get("content")
            if not isinstance(content, str):
                content = str(content or "")
            uncertainties = page.get("uncertainties", [])
            if not isinstance(uncertainties, list):
                uncertainties = [uncertainties] if uncertainties else []
            tables_val = page.get("tables", [])
            if not isinstance(tables_val, list):
                tables_val = []
            normalized.append({
                # Renumérotation séquentielle propre — on ne fait pas confiance au
                # page_number renvoyé par le modèle pour l'assemblage final des lots.
                "page_number": page_offset + i + 1,
                "content": content,
                "tables": [_normalize_table(t) for t in tables_val],
                "uncertainties": [_item_to_str(x) for x in uncertainties if x is not None],
                "confidence": page.get("confidence", 0.5),
            })

    data["pages"] = normalized
    return {"doc_id": data["doc_id"], "global_quality": data["global_quality"], "pages": data["pages"]}


def parse_transcription_json(raw: str, page_offset: int, provider_name: str) -> AIResponse:
    """Extrait, répare et valide un JSON de TranscriptionResult depuis une réponse brute de modèle."""
    text = raw.strip()

    m = _JSON_FENCE.search(text)
    if m:
        text = m.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    text = _TRAILING_COMMA.sub(r"\1", text)

    data_dict: dict | None = None
    for candidate in (text, repair_json(text)):
        candidate = _TRAILING_COMMA.sub(r"\1", candidate)
        try:
            data_dict = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue

    if data_dict is None:
        logger.error("%s — JSON irréparable. Début : %.300s", provider_name, raw)
        return AIResponse(
            success=False, data=None, confidence=0.0,
            raw_response=raw[:500], error=f"JSON invalide retourné par {provider_name}.",
        )

    try:
        normalized = normalize_transcription(data_dict, page_offset)
        validated = TranscriptionResult(**normalized)
    except Exception as e:
        logger.error("%s — validation du schéma échouée : %s", provider_name, e)
        return AIResponse(
            success=False, data=None, confidence=0.0,
            raw_response=raw[:500], error=f"Schéma invalide retourné par {provider_name} : {e}",
        )

    avg_conf = (
        sum(p.confidence for p in validated.pages) / len(validated.pages)
        if validated.pages else 0.5
    )
    return AIResponse(success=True, data=validated, confidence=avg_conf, raw_response=raw[:2000], error=None)
