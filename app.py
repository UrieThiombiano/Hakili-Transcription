import datetime
import logging
import re
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from src.config import settings
from src.export import export_to_excel, export_to_pdf
from src.ingestion import ingest_images, ingest_pdf
from src.models import Table, TranscriptionResult
from src.transcribe import TranscriptionFailed, transcribe_document, transcribe_single_page

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Transcription de manuscrit", layout="wide")

RUNS_DIR = Path(settings.runs_dir)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

_SLUG_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def _new_doc_id() -> str:
    return uuid.uuid4().hex[:8]


def _export_basename(source_name: str, doc_id: str) -> str:
    stem = Path(source_name).stem if source_name else doc_id
    slug = _SLUG_RE.sub("_", stem).strip("_") or doc_id
    date_str = datetime.date.today().isoformat()
    return f"{slug}_{date_str}"


def _load_rotated_image(img_path: Path, rotation: int) -> Image.Image:
    """Charge l'image et lui applique une rotation (0/90/180/270°, sens horaire)."""
    img = Image.open(img_path)
    if rotation:
        img = img.rotate(-rotation, expand=True)
    return img


def _apply_rotation_to_all_pages(ingestion, rotation: int) -> None:
    """Réécrit physiquement chaque image de page avec la rotation appliquée,
    AVANT le premier appel IA — un scan de travers dégrade fortement la
    transcription (cellules marquées [ILLISIBLE] alors qu'un humain les lit
    sans peine en redressant l'image)."""
    if not rotation:
        return
    for img_path in ingestion.pages:
        rotated = _load_rotated_image(img_path, rotation)
        rotated.convert("RGB").save(img_path)


def _text_area_height_for_image(width: int, height: int) -> int:
    return max(300, min(1200, int(600 * height / width))) if width else 500


def _attempt_transcription() -> None:
    """Lance (ou relance) la transcription à partir de l'ingestion déjà en session."""
    doc_id = st.session_state["doc_id"]
    ingestion = st.session_state["ingestion"]

    st.session_state.pop("transcription_error", None)
    progress_placeholder = st.empty()

    def progress(batch_num: int, total_batches: int) -> None:
        if total_batches > 1:
            progress_placeholder.info(f"Transcription du lot {batch_num}/{total_batches}…")

    with st.spinner("Transcription en cours (Gemini, secours GPT-5 si nécessaire)…"):
        try:
            transcription, failed_pages = transcribe_document(doc_id, ingestion.pages, on_progress=progress)
        except TranscriptionFailed as e:
            st.session_state["transcription_error"] = str(e)
            return
        finally:
            progress_placeholder.empty()

    st.session_state["transcription"] = transcription
    st.session_state["failed_pages"] = failed_pages
    st.session_state["edits"] = {p.page_number: p.content for p in transcription.pages}
    st.session_state["table_edits"] = {}


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

        st.session_state["doc_id"] = doc_id
        st.session_state["ingestion"] = ingestion
        st.session_state["source_name"] = uploaded[0].name
        st.rerun()


def render_orientation_step() -> None:
    """Étape avant la transcription : corriger l'orientation du scan si besoin.
    Un document scanné de travers dégrade fortement la qualité de la transcription
    IA (pas seulement la lisibilité humaine) — mieux vaut le redresser avant le
    premier appel plutôt qu'après coup."""
    ingestion = st.session_state["ingestion"]
    st.title("Orientation du document")
    st.caption(
        "Si le scan est de travers, redressez-le avant de lancer la transcription — "
        "l'IA lit beaucoup moins bien un document mal orienté. La rotation s'applique à toutes les pages."
    )

    rotation = st.session_state.get("global_rotation", 0)
    preview_path = ingestion.pages[0]

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⟲ Pivoter à gauche", width="stretch"):
            st.session_state["global_rotation"] = (rotation - 90) % 360
            st.rerun()
    with col2:
        if st.button("⟳ Pivoter à droite", width="stretch"):
            st.session_state["global_rotation"] = (rotation + 90) % 360
            st.rerun()

    st.image(_load_rotated_image(preview_path, rotation), caption="Aperçu — page 1", width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Confirmer l'orientation et transcrire", type="primary"):
            _apply_rotation_to_all_pages(ingestion, rotation)
            st.session_state["oriented"] = True
            _attempt_transcription()
            st.rerun()
    with col2:
        if st.button("← Recommencer avec un autre document"):
            for key in ("doc_id", "ingestion", "source_name", "global_rotation", "oriented",
                        "transcription_error", "transcription", "edits", "rotations",
                        "table_edits", "failed_pages"):
                st.session_state.pop(key, None)
            st.rerun()


def render_pending_step() -> None:
    """Ingestion faite mais transcription pas encore réussie — permet de réessayer sans réuploader."""
    ingestion = st.session_state["ingestion"]
    st.title("Transcription de manuscrit")
    st.info(f"Document ingéré : {ingestion.total_pages} page(s) prête(s) pour la transcription.")

    if st.session_state.get("transcription_error"):
        st.error(f"Échec de la transcription : {st.session_state['transcription_error']}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Réessayer la transcription", type="primary"):
            _attempt_transcription()
            st.rerun()
    with col2:
        if st.button("← Recommencer avec un autre document"):
            for key in ("doc_id", "ingestion", "source_name", "global_rotation", "oriented",
                        "transcription_error", "transcription", "edits", "rotations",
                        "table_edits", "failed_pages"):
                st.session_state.pop(key, None)
            st.rerun()


def _rebuild_table(page_number: int, table_index: int, original: Table) -> Table:
    edited_df = st.session_state["table_edits"].get((page_number, table_index))
    if edited_df is None:
        return original
    headers = [str(c) for c in edited_df.columns]
    rows = [[("" if pd.isna(v) else str(v)) for v in row] for row in edited_df.itertuples(index=False)]
    return Table(title=original.title, headers=headers, rows=rows)


def render_review_step() -> None:
    transcription: TranscriptionResult = st.session_state["transcription"]
    ingestion = st.session_state["ingestion"]

    st.title("Relecture de la transcription")
    st.caption(
        "⟦texte⟧ = lecture incertaine  ·  [ILLISIBLE] = passage illisible — "
        "corrigez directement dans le texte ou les tableaux ci-dessous."
    )

    failed_pages = st.session_state.get("failed_pages") or []
    if failed_pages:
        st.warning(
            "Certaines pages n'ont pas pu être transcrites (Gemini et GPT-5 ont échoué) : "
            f"page(s) {', '.join(str(p) for p in failed_pages)}. "
            "Utilisez « Retranscrire cette page » ci-dessous après avoir vérifié l'orientation."
        )

    if st.button("← Recommencer avec un autre document"):
        for key in ("doc_id", "ingestion", "source_name", "global_rotation", "oriented",
                    "transcription_error", "transcription", "edits", "rotations",
                    "table_edits", "failed_pages"):
            st.session_state.pop(key, None)
        st.rerun()

    if "rotations" not in st.session_state:
        st.session_state["rotations"] = {}
    if "table_edits" not in st.session_state:
        st.session_state["table_edits"] = {}
    rotations = st.session_state["rotations"]
    table_edits = st.session_state["table_edits"]

    edits = st.session_state["edits"]
    page_images = {i + 1: p for i, p in enumerate(ingestion.pages)}

    for page in sorted(transcription.pages, key=lambda p: p.page_number):
        idx = page.page_number
        st.markdown(f"##### Page {idx}")
        col_img, col_txt = st.columns([1, 1], gap="large")

        img_path = page_images.get(idx)
        img_exists = bool(img_path) and Path(img_path).exists()
        rotated_img = None

        with col_img:
            if img_exists:
                rot_left, rot_right = st.columns(2)
                with rot_left:
                    if st.button("⟲ Pivoter à gauche", key=f"rot_left_{idx}", width="stretch"):
                        rotations[idx] = (rotations.get(idx, 0) - 90) % 360
                with rot_right:
                    if st.button("⟳ Pivoter à droite", key=f"rot_right_{idx}", width="stretch"):
                        rotations[idx] = (rotations.get(idx, 0) + 90) % 360

                rotated_img = _load_rotated_image(img_path, rotations.get(idx, 0))
                st.image(rotated_img, width="stretch")

                if st.button("🔄 Retranscrire cette page (avec la rotation appliquée)", key=f"retrans_{idx}"):
                    with st.spinner(f"Retranscription de la page {idx}…"):
                        rot_path = img_path.parent / f"_rot_{idx}.jpg"
                        rotated_img.convert("RGB").save(rot_path)
                        try:
                            new_page = transcribe_single_page(st.session_state["doc_id"], rot_path, idx)
                        except TranscriptionFailed as e:
                            st.error(str(e))
                        else:
                            transcription.pages = [
                                new_page if p.page_number == idx else p for p in transcription.pages
                            ]
                            edits[idx] = new_page.content
                            for k in [k for k in table_edits if k[0] == idx]:
                                table_edits.pop(k, None)
                            if idx in st.session_state.get("failed_pages", []):
                                st.session_state["failed_pages"].remove(idx)
                            st.session_state["transcription"] = transcription
                            st.rerun()
            else:
                st.caption("Image indisponible")

        with col_txt:
            height = _text_area_height_for_image(*rotated_img.size) if img_exists and rotated_img else 500
            edited = st.text_area(
                f"Transcription page {idx}",
                value=edits.get(idx, page.content),
                height=height,
                key=f"trans_edit_{idx}",
                label_visibility="collapsed",
                help="Corrigez directement le texte si la transcription IA s'est trompée.",
            )
            edits[idx] = edited

        for tbl_idx, table in enumerate(page.tables):
            label = table.title or f"Tableau {tbl_idx + 1}"
            st.markdown(f"**{label}**")
            width = max([len(table.headers)] + [len(r) for r in table.rows] + [1])
            headers = list(table.headers) + [f"Colonne {j + 1}" for j in range(len(table.headers), width)]
            rows = [row + [""] * (width - len(row)) for row in table.rows]
            df = pd.DataFrame(rows, columns=headers)
            edited_df = st.data_editor(
                df, key=f"table_edit_{idx}_{tbl_idx}", num_rows="dynamic", width="stretch",
            )
            table_edits[(idx, tbl_idx)] = edited_df

        if page.uncertainties:
            with st.expander(f"⚠ {len(page.uncertainties)} zone(s) signalée(s) — page {idx}"):
                for u in page.uncertainties:
                    st.markdown(f"- {u}")

        st.divider()

    st.session_state["edits"] = edits
    st.session_state["table_edits"] = table_edits

    st.subheader("Export")
    col1, col2 = st.columns(2)

    validated = TranscriptionResult(
        doc_id=transcription.doc_id,
        global_quality=transcription.global_quality,
        pages=[
            page.model_copy(update={
                "content": edits.get(page.page_number, page.content),
                "tables": [
                    _rebuild_table(page.page_number, tbl_idx, table)
                    for tbl_idx, table in enumerate(page.tables)
                ],
            })
            for page in transcription.pages
        ],
    )

    basename = _export_basename(st.session_state.get("source_name", ""), validated.doc_id)

    with col1:
        pdf_bytes = export_to_pdf(validated)
        st.download_button(
            "Exporter en PDF",
            data=pdf_bytes,
            file_name=f"{basename}.pdf",
            mime="application/pdf",
        )

    with col2:
        excel_bytes = export_to_excel(validated)
        st.download_button(
            "Exporter en Excel",
            data=excel_bytes,
            file_name=f"{basename}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


def main() -> None:
    if "transcription" in st.session_state:
        render_review_step()
    elif "ingestion" in st.session_state and not st.session_state.get("oriented"):
        render_orientation_step()
    elif "ingestion" in st.session_state:
        render_pending_step()
    else:
        render_upload_step()


if __name__ == "__main__":
    main()
