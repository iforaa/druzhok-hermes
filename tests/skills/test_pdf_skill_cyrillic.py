"""The pdf skill's reportlab scripts write Russian text that reads back as Russian.

Runs the real scripts (pdf_create, pdf_make_form, pdf_stamp) as the agent does
and extracts the text with pdfplumber. reportlab's built-in Helvetica has no
Cyrillic glyphs, so this fails unless the scripts pick a Unicode TTF.
"""
import json
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("reportlab")
pytest.importorskip("pypdf")
pdfplumber = pytest.importorskip("pdfplumber")
pytest.importorskip("matplotlib")  # bundled DejaVu Sans, when the system has none

SCRIPTS = Path(__file__).resolve().parents[2] / "skills/productivity/pdf/scripts"

HEADING = "Отчёт за ноябрь"
PARAGRAPH = "Съешь же ещё этих мягких французских булок, да выпей чаю."
CELLS = ["Имя", "Сумма", "Иван", "1 200 ₽"]
LABEL = "Фамилия"
OPTION = "Да"
STAMP = "ЧЕРНОВИК"
BOLD_STAMP = "КОПИЯ"  # an explicit --font Helvetica-Bold


def _run(script: str, *args) -> None:
    subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, args)],
                   check=True, capture_output=True)


def _text(pdf: Path) -> str:
    with pdfplumber.open(pdf) as doc:
        return "\n".join(page.extract_text() or "" for page in doc.pages)


def _font_names(pdf: Path) -> set[str]:
    with pdfplumber.open(pdf) as doc:
        return {ch["fontname"] for page in doc.pages for ch in page.chars}


def test_created_pdf_heading_paragraph_table_round_trip(tmp_path):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"title": HEADING, "elements": [
        {"type": "heading", "text": HEADING, "level": 1},
        {"type": "paragraph", "text": f"<b>Важно:</b> {PARAGRAPH}"},
        {"type": "table", "rows": [CELLS[:2], CELLS[2:]], "header": True},
    ]}, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "report.pdf"
    _run("pdf_create.py", spec, "-o", out)

    text = _text(out)
    for expected in (HEADING, "Важно:", PARAGRAPH, *CELLS):
        assert expected in text
    assert all("DejaVuSans" in name for name in _font_names(out))


def test_form_labels_and_stamp_round_trip(tmp_path):
    spec = tmp_path / "form.json"
    spec.write_text(json.dumps({"fields": [
        {"name": "surname", "type": "text", "label": LABEL,
         "label_box": [72, 700, 150, 714], "entry_box": [160, 696, 400, 716]},
        {"name": "agree", "type": "radio", "label": "Согласие",
         "label_box": [72, 660, 150, 674], "entry_box": [160, 656, 400, 676],
         "options": [OPTION, "Нет"], "value": OPTION},
    ]}, ensure_ascii=False), encoding="utf-8")
    form = tmp_path / "form.pdf"
    _run("pdf_make_form.py", spec, "-o", form)
    stamped = tmp_path / "stamped.pdf"
    _run("pdf_stamp.py", form, "-o", stamped, "--text", STAMP, "--x", 72, "--y", 400)
    bold = tmp_path / "bold.pdf"
    _run("pdf_stamp.py", stamped, "-o", bold, "--text", BOLD_STAMP, "--x", 72, "--y", 300,
         "--font", "Helvetica-Bold")

    text = _text(bold)
    for expected in (LABEL, "Согласие", OPTION, "Нет", STAMP, BOLD_STAMP):
        assert expected in text
