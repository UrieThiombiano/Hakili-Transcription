import logging
import uuid
from pathlib import Path

import streamlit as st

from src.config import settings
from src.export import export_to_excel, export_to_pdf
from src.gemini_client import GeminiTranscriptionClient
from src.ingestion import ingest_images, ingest_pdf
from src.models import TranscriptionResult
from src.openai_client import OpenAIClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Transcription de manuscrit", layout="wide")

RUNS_DIR = Path(settings.runs_dir)
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def _new_doc_id() -> str:
    return uuid.uuid4().hex[:8]


def _run_transcription(doc_id: str, image_paths: list[Path]) -> TranscriptionResult:
    gemini = GeminiTranscriptionClient()
    response = gemini.transcribe(doc_id, image_paths)

    if not response.success or response.data is None:
        st.warning("Gemini a échoué — bascule sur GPT-5 (filet de secours).")
        logger.warning("Gemini échec (doc_id=%s) : %s — fallback GPT-5", doc_id, response.error)
        openai_client = OpenAIClient()
        response = openai_client.transcribe(doc_id, image_paths)

    if not response.success or response.data is None:
        raise RuntimeError(f"Échec de la transcription (Gemini et GPT-5) : {response.error}")

    return response.data


def _text_area_height_for_image(img_path: Path | None) -> int:
    if not img_path:
        return 500
    try:
        from PIL import Image
        with Image.open(img_path) as im:
            w, h = im.size
        return max(300, min(1200, int(600 * h / w))) if w else 500
    except Exception:
        return 500


def render_upload_step() -> None:
    st.title("Transcription de manuscrit")
    st.caption("Déposez un PDF ou des images (une par page) du document manuscrit à transcrire.")

    uploaded = st.file_uploader(
        "Document manuscrit",
        type=["pdf", "jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if not uploaded:
        return

    if st.button("Transcrire", type="primary"):
        doc_id = _new_doc_id()
        tmp_dir = RUNS_DIR / "_uploads" / doc_id
        tmp_dir.mkdir(parents=True, exist_ok=True)

        is_pdf = len(uploaded) == 1 and uploaded[0].name.lower().endswith(".pdf")

        with st.spinner("Ingestion du document…"):
            if is_pdf:
                pdf_path = tmp_dir / uploaded[0].name
                pdf_path.write_bytes(uploaded[0].getvalue())
                ingestion = ingest_pdf(pdf_path, doc_id, RUNS_DIR)
            else:
                image_paths = []
                for f in uploaded:
                    p = tmp_dir / f.name
                    p.write_bytes(f.getvalue())
                    image_paths.append(p)
                ingestion = ingest_images(image_paths, doc_id, RUNS_DIR)

        with st.spinner("Transcription en cours (Gemini)…"):
            try:
                transcription = _run_transcription(doc_id, ingestion.pages)
            except Exception as e:
                st.error(f"Échec de la transcription : {e}")
                return

        st.session_state["doc_id"] = doc_id
        st.session_state["ingestion"] = ingestion
        st.session_state["transcription"] = transcription
        st.session_state["edits"] = {p.page_number: p.content for p in transcription.pages}
        st.rerun()


def render_review_step() -> None:
    transcription: TranscriptionResult = st.session_state["transcription"]
    ingestion = st.session_state["ingestion"]

    st.title("Relecture de la transcription")
    st.caption(
        "⟦texte⟧ = lecture incertaine  ·  [ILLISIBLE] = passage illisible — "
        "corrigez directement dans le texte ci-dessous."
    )

    if st.button("← Recommencer avec un autre document"):
        for key in ("doc_id", "ingestion", "transcription", "edits"):
            st.session_state.pop(key, None)
        st.rerun()

    edits = st.session_state["edits"]
    page_images = {i + 1: p for i, p in enumerate(ingestion.pages)}

    for page in sorted(transcription.pages, key=lambda p: p.page_number):
        idx = page.page_number
        st.markdown(f"##### Page {idx}")
        col_img, col_txt = st.columns([1, 1], gap="large")

        img_path = page_images.get(idx)
        img_exists = bool(img_path) and Path(img_path).exists()

        with col_img:
            if img_exists:
                st.image(str(img_path), width="stretch")
            else:
                st.caption("Image indisponible")

        with col_txt:
            height = _text_area_height_for_image(img_path) if img_exists else 500
            edited = st.text_area(
                f"Transcription page {idx}",
                value=edits.get(idx, page.content),
                height=height,
                key=f"trans_edit_{idx}",
                label_visibility="collapsed",
                help="Corrigez directement le texte si la transcription IA s'est trompée.",
            )
            edits[idx] = edited

        if page.uncertainties:
            with st.expander(f"⚠ {len(page.uncertainties)} zone(s) signalée(s) — page {idx}"):
                for u in page.uncertainties:
                    st.markdown(f"- {u}")

        st.divider()

    st.session_state["edits"] = edits

    st.subheader("Export")
    col1, col2 = st.columns(2)

    validated = TranscriptionResult(
        doc_id=transcription.doc_id,
        global_quality=transcription.global_quality,
        pages=[
            page.model_copy(update={"content": edits.get(page.page_number, page.content)})
            for page in transcription.pages
        ],
    )

    with col1:
        pdf_bytes = export_to_pdf(validated)
        st.download_button(
            "Exporter en PDF",
            data=pdf_bytes,
            file_name=f"transcription_{validated.doc_id}.pdf",
            mime="application/pdf",
        )

    with col2:
        excel_bytes = export_to_excel(validated)
        st.download_button(
            "Exporter en Excel",
            data=excel_bytes,
            file_name=f"transcription_{validated.doc_id}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


def main() -> None:
    if "transcription" in st.session_state:
        render_review_step()
    else:
        render_upload_step()


if __name__ == "__main__":
    main()
