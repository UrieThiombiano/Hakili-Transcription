from io import BytesIO
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from src.models import TranscriptionResult


def _content_to_html(content: str) -> str:
    """Échappe le texte pour reportlab et convertit les retours à la ligne en <br/>."""
    return escape(content).replace("\n", "<br/>")


def export_to_pdf(transcription: TranscriptionResult) -> bytes:
    """Génère un PDF contenant uniquement le texte transcrit, une section par page."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    story = []

    for page in sorted(transcription.pages, key=lambda p: p.page_number):
        story.append(Paragraph(f"Page {page.page_number}", styles["Heading3"]))
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph(_content_to_html(page.content), styles["BodyText"]))
        story.append(Spacer(1, 0.8 * cm))

    doc.build(story)
    return buffer.getvalue()


def export_to_excel(transcription: TranscriptionResult) -> bytes:
    """Génère un classeur Excel avec une ligne par page (n° de page | texte transcrit)."""
    rows = [
        {"Page": page.page_number, "Texte transcrit": page.content}
        for page in sorted(transcription.pages, key=lambda p: p.page_number)
    ]
    df = pd.DataFrame(rows)

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Transcription")
        worksheet = writer.sheets["Transcription"]
        worksheet.column_dimensions["A"].width = 10
        worksheet.column_dimensions["B"].width = 100
    return buffer.getvalue()
