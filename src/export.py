import re
from io import BytesIO
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.models import TranscriptionResult

_INVALID_SHEET_CHARS = re.compile(r"[:\\/?*\[\]]")


def _uncertain_markers_to_ascii(text: str) -> str:
    """La police PDF standard (Helvetica) n'a pas les glyphes ⟦/⟧ (rendus en carré
    illisible) — on les remplace par un équivalent ASCII qui s'affiche partout."""
    return text.replace("⟦", "[[").replace("⟧", "]]")


def _content_to_html(content: str) -> str:
    """Échappe le texte pour reportlab et convertit les retours à la ligne en <br/>."""
    content = _uncertain_markers_to_ascii(content)
    return escape(content).replace("\n", "<br/>")


def _sanitize_sheet_name(name: str, fallback: str) -> str:
    name = _INVALID_SHEET_CHARS.sub("", name).strip()
    return (name or fallback)[:31]


def _table_signature(headers: list[str]) -> tuple[str, ...]:
    return tuple(h.strip().lower() for h in headers)


def _collect_table_groups(transcription: TranscriptionResult) -> list[dict]:
    """Regroupe les tableaux détectés par schéma de colonnes (même en-têtes =
    continuation du même tableau sur plusieurs pages) et concatène leurs lignes."""
    groups: dict[tuple, dict] = {}
    order: list[tuple] = []
    anon_counter = 0

    for page in sorted(transcription.pages, key=lambda p: p.page_number):
        for table in page.tables:
            if table.headers:
                sig = _table_signature(table.headers)
            else:
                anon_counter += 1
                sig = ("__anon__", anon_counter)

            if sig not in groups:
                groups[sig] = {"title": table.title, "headers": list(table.headers), "rows": []}
                order.append(sig)
            elif not groups[sig]["title"] and table.title:
                groups[sig]["title"] = table.title

            groups[sig]["rows"].extend(table.rows)

    return [groups[sig] for sig in order]


def _table_group_to_dataframe(group: dict) -> pd.DataFrame:
    headers = list(group["headers"])
    width = max([len(headers)] + [len(r) for r in group["rows"]] + [1])
    headers += [f"Colonne {i + 1}" for i in range(len(headers), width)]
    rows = [row + [""] * (width - len(row)) for row in group["rows"]]
    return pd.DataFrame(rows, columns=headers)


def export_to_pdf(transcription: TranscriptionResult) -> bytes:
    """Génère un PDF avec le texte transcrit et les tableaux détectés, une section par page source."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    cell_style = styles["BodyText"].clone("cell")
    cell_style.fontSize = 8
    cell_style.leading = 10
    header_style = cell_style.clone("header_cell")
    header_style.textColor = colors.white
    header_style.fontName = "Helvetica-Bold"
    available_width = doc.width

    pages = sorted(transcription.pages, key=lambda p: p.page_number)
    story = []

    for i, page in enumerate(pages):
        story.append(Paragraph(f"Page {page.page_number}", styles["Heading3"]))
        story.append(Spacer(1, 0.2 * cm))

        if page.content.strip():
            story.append(Paragraph(_content_to_html(page.content), styles["BodyText"]))
            story.append(Spacer(1, 0.4 * cm))

        for table in page.tables:
            if table.title:
                story.append(Paragraph(escape(_uncertain_markers_to_ascii(table.title)), styles["Heading4"]))

            width = max([len(table.headers)] + [len(r) for r in table.rows] + [1])
            headers = list(table.headers) + [f"Colonne {j + 1}" for j in range(len(table.headers), width)]
            col_width = available_width / width

            data = [[Paragraph(escape(_uncertain_markers_to_ascii(h)), header_style) for h in headers]]
            for row in table.rows:
                padded = row + [""] * (width - len(row))
                data.append([Paragraph(escape(_uncertain_markers_to_ascii(c)), cell_style) for c in padded])

            t = Table(data, colWidths=[col_width] * width, repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#001e4a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            story.append(t)
            story.append(Spacer(1, 0.4 * cm))

        if i < len(pages) - 1:
            story.append(PageBreak())

    doc.build(story)
    return buffer.getvalue()


def export_to_excel(transcription: TranscriptionResult) -> bytes:
    """
    Génère un classeur Excel :
    - une feuille par tableau détecté (colonnes réelles, lignes fusionnées entre
      pages quand le même en-tête se poursuit sur plusieurs pages) ;
    - une feuille "Texte libre" pour le texte hors tableau, si non vide ;
    - si aucun tableau n'est détecté sur le document, une feuille unique
      "Transcription" (Page | Texte) — comportement d'origine, pour les
      manuscrits purement narratifs.
    """
    table_groups = _collect_table_groups(transcription)

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        if table_groups:
            used_names: set[str] = set()
            for idx, group in enumerate(table_groups, start=1):
                df = _table_group_to_dataframe(group)
                sheet_name = _sanitize_sheet_name(group["title"], f"Tableau {idx}")
                base_name, suffix = sheet_name, 2
                while sheet_name in used_names:
                    sheet_name = _sanitize_sheet_name(f"{base_name} ({suffix})", f"Tableau {idx} ({suffix})")
                    suffix += 1
                used_names.add(sheet_name)

                df.to_excel(writer, index=False, sheet_name=sheet_name)
                worksheet = writer.sheets[sheet_name]
                for col_idx, col in enumerate(df.columns, start=1):
                    max_len = max([len(str(col))] + [len(str(v)) for v in df[col]]) if len(df) else len(str(col))
                    worksheet.column_dimensions[worksheet.cell(row=1, column=col_idx).column_letter].width = (
                        min(60, max(10, max_len + 2))
                    )

            free_text_rows = [
                {"Page": page.page_number, "Texte": page.content}
                for page in sorted(transcription.pages, key=lambda p: p.page_number)
                if page.content.strip()
            ]
            if free_text_rows:
                df_text = pd.DataFrame(free_text_rows)
                df_text.to_excel(writer, index=False, sheet_name="Texte libre")
                ws = writer.sheets["Texte libre"]
                ws.column_dimensions["A"].width = 10
                ws.column_dimensions["B"].width = 100
        else:
            rows = [
                {"Page": page.page_number, "Texte transcrit": page.content}
                for page in sorted(transcription.pages, key=lambda p: p.page_number)
            ]
            df = pd.DataFrame(rows)
            df.to_excel(writer, index=False, sheet_name="Transcription")
            worksheet = writer.sheets["Transcription"]
            worksheet.column_dimensions["A"].width = 10
            worksheet.column_dimensions["B"].width = 100

    return buffer.getvalue()
