"""Shared reportlab font choice: DejaVu Sans when found, else Helvetica.

reportlab's built-in Helvetica covers Latin-1 only, so Cyrillic (and most
other non-Latin text) comes out as missing-glyph boxes. DejaVu Sans is looked
up in the system font dir (Debian/Ubuntu `fonts-dejavu-core`), then in
matplotlib's bundled copy. Not a CLI.
"""
from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

_SYSTEM_DIR = Path("/usr/share/fonts/truetype/dejavu")
_FILES = {
    "regular": "DejaVuSans.ttf",
    "bold": "DejaVuSans-Bold.ttf",
    "italic": "DejaVuSans-Oblique.ttf",
    "boldItalic": "DejaVuSans-BoldOblique.ttf",
}
_HELVETICA = {
    "regular": "Helvetica",
    "bold": "Helvetica-Bold",
    "italic": "Helvetica-Oblique",
    "boldItalic": "Helvetica-BoldOblique",
}


def _font_dirs() -> list[Path]:
    dirs = [_SYSTEM_DIR]
    spec = importlib.util.find_spec("matplotlib")
    for root in (spec.submodule_search_locations or []) if spec else []:
        dirs.append(Path(root) / "mpl-data" / "fonts" / "ttf")
    return dirs


@lru_cache(maxsize=1)
def fonts() -> dict[str, str]:
    """reportlab font names for regular/bold/italic/boldItalic.

    Registers DejaVu Sans (and its family, so <b>/<i> markup works) from the
    first dir holding both regular and bold; a missing oblique falls back to
    the upright face. Returns the Helvetica names when DejaVu isn't found.
    """
    from reportlab.lib.fonts import addMapping
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    for d in _font_dirs():
        if not all((d / _FILES[k]).is_file() for k in ("regular", "bold")):
            continue
        names = {}
        for style, upright in (("regular", "regular"), ("bold", "bold"),
                               ("italic", "regular"), ("boldItalic", "bold")):
            path = d / _FILES[style]
            if not path.is_file():
                names[style] = names[upright]
                continue
            name = path.stem
            pdfmetrics.registerFont(TTFont(name, str(path)))
            names[style] = name
        for (bold, italic), style in {(0, 0): "regular", (1, 0): "bold",
                                      (0, 1): "italic", (1, 1): "boldItalic"}.items():
            addMapping(names["regular"], bold, italic, names[style])
        return names
    return dict(_HELVETICA)


def face(name: str | None) -> str:
    """The font to use for `name`: a Helvetica face (or None, regular) maps to
    the matching fonts() face; any other name is kept as given."""
    f = fonts()
    by_helvetica = {_HELVETICA[k]: f[k] for k in _HELVETICA}
    return f["regular"] if name is None else by_helvetica.get(name, name)


def use_in_styles(styles) -> None:
    """Point a getSampleStyleSheet() at fonts(), replacing Helvetica faces."""
    for style in styles.byName.values():
        if getattr(style, "fontName", None):
            style.fontName = face(style.fontName)
