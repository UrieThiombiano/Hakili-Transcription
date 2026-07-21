"""Orchestration de la transcription d'un document complet : découpage en
lots de pages (pour éviter de dépasser la limite de tokens de sortie sur les
documents longs), bascule Gemini → GPT-5 par lot, et résilience partielle
(un lot en échec ne fait pas échouer tout le document)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from src.gemini_client import GeminiTranscriptionClient
from src.models import PageTranscription, TranscriptionResult
from src.openai_client import OpenAIClient

logger = logging.getLogger(__name__)

_MAX_PAGES_PER_BATCH = 4


class TranscriptionFailed(Exception):
    """Levée uniquement si AUCUN lot n'a pu être transcrit (Gemini et GPT-5 tous deux en échec partout)."""


def transcribe_document(
    doc_id: str, image_paths: list[Path], on_progress: Callable[[int, int], None] | None = None,
) -> tuple[TranscriptionResult, list[int]]:
    """
    Transcrit un document complet, par lots de _MAX_PAGES_PER_BATCH pages.
    Pour chaque lot : tente Gemini, puis GPT-5 en cas d'échec.
    Si les deux échouent pour un lot, insère des pages "placeholder" en échec
    plutôt que d'abandonner tout le document — sauf si AUCUN lot n'a réussi,
    auquel cas TranscriptionFailed est levée.

    Retourne (résultat_fusionné, liste_des_numéros_de_page_en_échec).
    """
    gemini = GeminiTranscriptionClient()
    openai_client: OpenAIClient | None = None

    batches = [
        (start, image_paths[start:start + _MAX_PAGES_PER_BATCH])
        for start in range(0, len(image_paths), _MAX_PAGES_PER_BATCH)
    ]

    all_pages: list[PageTranscription] = []
    qualities: list[str] = []
    failed_pages: list[int] = []
    any_batch_succeeded = False

    for batch_index, (offset, batch) in enumerate(batches):
        if on_progress:
            on_progress(batch_index + 1, len(batches))

        response = gemini.transcribe(doc_id, batch, page_offset=offset)

        if not response.success or response.data is None:
            logger.warning("Lot %d (offset=%d) : échec Gemini — bascule GPT-5. Erreur : %s",
                           batch_index + 1, offset, response.error)
            if openai_client is None:
                openai_client = OpenAIClient()
            response = openai_client.transcribe(doc_id, batch, page_offset=offset)

        if response.success and response.data is not None:
            any_batch_succeeded = True
            all_pages.extend(response.data.pages)
            qualities.append(response.data.global_quality)
        else:
            logger.error("Lot %d (offset=%d) : échec Gemini ET GPT-5 — pages marquées en échec. Erreur : %s",
                         batch_index + 1, offset, response.error)
            for i in range(len(batch)):
                page_number = offset + i + 1
                failed_pages.append(page_number)
                all_pages.append(PageTranscription(
                    page_number=page_number,
                    content="[ÉCHEC DE TRANSCRIPTION — réessayez cette page individuellement]",
                    tables=[],
                    uncertainties=["Transcription IA indisponible pour cette page (Gemini et GPT-5 ont échoué)."],
                    confidence=0.0,
                ))

    if not any_batch_succeeded:
        raise TranscriptionFailed(
            "Échec de la transcription sur tous les lots (Gemini et GPT-5 tous deux indisponibles)."
        )

    global_quality = (
        "poor" if "poor" in qualities or failed_pages
        else "medium" if "medium" in qualities
        else "good"
    )
    all_pages.sort(key=lambda p: p.page_number)
    merged = TranscriptionResult(doc_id=doc_id, global_quality=global_quality, pages=all_pages)
    return merged, failed_pages


def transcribe_single_page(doc_id: str, image_path: Path, page_number: int) -> PageTranscription:
    """Retranscrit une seule page (ex : après rotation) — Gemini puis GPT-5 en secours."""
    gemini = GeminiTranscriptionClient()
    offset = page_number - 1
    response = gemini.transcribe(doc_id, [image_path], page_offset=offset)

    if not response.success or response.data is None:
        logger.warning("Retranscription page %d : échec Gemini — bascule GPT-5. Erreur : %s",
                       page_number, response.error)
        openai_client = OpenAIClient()
        response = openai_client.transcribe(doc_id, [image_path], page_offset=offset)

    if not response.success or response.data is None or not response.data.pages:
        raise TranscriptionFailed(f"Échec de la retranscription de la page {page_number} : {response.error}")

    return response.data.pages[0]
