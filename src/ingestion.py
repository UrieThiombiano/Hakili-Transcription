import shutil
from pathlib import Path

import fitz  # PyMuPDF

from src.models import IngestionResult


def ingest_images(image_paths: list[Path], doc_id: str, output_dir: Path) -> IngestionResult:
    """Copie des images JPG/PNG dans le dossier de session, renommées page_01, page_02…"""
    dest_dir = output_dir / doc_id / "pages"
    dest_dir.mkdir(parents=True, exist_ok=True)

    pages: list[Path] = []
    for i, src in enumerate(image_paths):
        suffix = src.suffix.lower() if src.suffix.lower() in {".jpg", ".jpeg", ".png"} else ".jpg"
        dest = dest_dir / f"page_{i + 1:02d}{suffix}"
        shutil.copy2(src, dest)
        pages.append(dest)

    return IngestionResult(
        doc_id=doc_id,
        total_pages=len(pages),
        pages=pages,
        output_dir=dest_dir,
    )


def ingest_pdf(pdf_path: Path, doc_id: str, output_dir: Path) -> IngestionResult:
    """Convertit un PDF en images JPG, une par page (150 DPI — optimal pour la vision LLM)."""
    output_dir = output_dir / doc_id / "pages"
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    image_paths = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=150)

        image_path = output_dir / f"page_{page_num + 1:02d}.jpg"
        pix.save(str(image_path))
        image_paths.append(image_path)

    doc.close()

    return IngestionResult(
        doc_id=doc_id,
        total_pages=len(image_paths),
        pages=image_paths,
        output_dir=output_dir,
    )
