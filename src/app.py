"""app.py - Gradio interface: import a scan, review the transcription and the JSON record.

    python src/app.py                      # http://127.0.0.1:7860
    python src/app.py --share --device cuda:0

Layout
  top bar     Files (import a scan: it is processed at once), Save (zip of everything), Options, Help
  left        Image / Segmentation tabs: the scan, and the scan with the polygon of every text line coloured by
              the kraken confidence; + / - zoom, a download button in each tab
  right       Text / JSON tabs. Text: the transcription, undo / redo, [S] the spelling review, [E] editing
              (leaving the editor parses the text again). JSON: the record as a table (or raw), [E] editing
              the record by hand, [文] the French fields translated again from the Arabic ones, download
  bottom      console: validation errors and warnings, low confidence lines, spelling and parser notes

Hovering a line anywhere (a polygon of the segmentation, a line of the text, a row of the JSON table) highlights
the same line in the other panels (CSS only: every element of line n has the class "ln-n").

The spelling review: every proposed change highlighted in the text (green = applied, red = not applied, the
badge is the proposed spelling): click a change to accept / refuse it, then "Apply". A misread job close to
several jobs of jobs.json cycles through them at every click. A child's name whose Arabic and Latin readings
disagree gets a proposal that is never applied automatically, or an amber "≠ ..." flag.

Languages: English, French and Arabic, chosen in gradio's Settings (bottom of the page; by default the browser's
language), see ui_text.py. The app is a PWA (installable); when a processing ends while the page is in the
background, the browser shows a notification (the permission is asked at the first click on Files).

The kraken device and models are chosen on the command line only (not in the page: a shared app must not let
visitors point kraken at other files). The scans are personal documents: every file of a run (the upload in
gradio's cache, the downloads) is deleted after KEEP_SECONDS. Every stage runs on this
computer (kraken, the NER models, the spelling and translation models through Ollama): nothing is sent online.
"""

import argparse
import base64
import html
import inspect
import io
import json
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile
from functools import lru_cache
from pathlib import Path

import gradio as gr
from PIL import Image, ImageOps

import translator
from common import TEMPLATE, normalize_text, resolve_device, write_json
from mapper import map_fields
from parser import find_zones, flatten
from pipeline import run_pipeline
from recognition import MODEL_RECOGNITION, recognize_lines
from segmentation import MODEL_SEGMENTATION, segment_image
from spelling_checker import MODEL as SPELL_MODEL, apply_changes, describe, line_units, propose
from validator import validate
from ui_text import CSS as I18N_CSS, PERMISSION_JS, gradio_i18n, notification_js, tr

DISPLAY_MAX_SIDE = 1800   # the scan is shown at most this wide (polygons keep the original coordinates)
KEEP_SECONDS = 3600       # uploads and downloads of a run are deleted after one hour
RUNS_DIR = Path(tempfile.gettempdir()) / "madoc_runs"
ZOOM_STEPS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
# anonymized example scans offered in the interface (Samples), next to the transcriptions of the tests
SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


# the logos (logo/): the one with the name (wide) in the top bar and on the login page, the square one as the icon
# of the page (browser tab, installed application). Found by their shape, whatever their file names; a file with
# "dark" in its name is the version for the dark theme (madoc_dark.png).
LOGO_DIR = Path(__file__).resolve().parents[1] / "logo"
LOGO_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".svg")


def image_ratio(path):
    """Width / height of an image (an SVG from its viewBox or its width / height), None when unreadable."""
    try:
        if path.suffix.lower() == ".svg":
            head = path.read_text(encoding="utf-8", errors="ignore")[:4000]
            box = re.search(r'viewBox="[\d.\-]+[\s,]+[\d.\-]+[\s,]+([\d.]+)[\s,]+([\d.]+)"', head)
            size = box or re.search(r'<svg[^>]*?width="([\d.]+)[^"]*"[^>]*?height="([\d.]+)', head)
            return float(size.group(1)) / float(size.group(2)) if size else None
        with Image.open(path) as image:
            return image.width / image.height
    except (OSError, ValueError, ZeroDivisionError):
        return None


def find_logos(folder=None):
    """{"wordmark": the logo with the name (wider than high), "wordmark_dark": its version for the dark theme,
    "icon": the square one, "icon_dark"} -> paths (only those found)."""
    folder = Path(folder or LOGO_DIR)
    found = {}
    if not folder.is_dir():
        return found
    for path in sorted(folder.iterdir()):
        ratio = image_ratio(path) if path.suffix.lower() in LOGO_SUFFIXES else None
        if ratio is None:
            continue
        kind = "wordmark" if ratio >= 1.6 else "icon"
        found.setdefault(kind + ("_dark" if "dark" in path.stem.lower() else ""), path)
    return found


@lru_cache(maxsize=8)
def logo_uri(path, width=800):
    """The logo as a data: URI (it is shown on the login page, before any file of the server can be read);
    a large raster image is made smaller first."""
    path = Path(path)
    if path.suffix.lower() == ".svg":
        return "data:image/svg+xml;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
    with Image.open(path) as image:
        image = image.convert("RGBA")
        if image.width > width:
            image = image.resize((width, round(image.height * width / image.width)), Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def logo_images(logos):
    """<img> of the logo with the name: the light theme's one and, when there is one, the dark theme's one (the
    CSS shows one of them; without a dark version the light one is shown on a white box in the dark theme)."""
    light, dark = logos.get("wordmark"), logos.get("wordmark_dark")
    if not light:
        light, dark = dark, None
    if not light:
        return ""
    images = f'<img class="md-logo{"" if dark else " md-boxed"}" src="{logo_uri(str(light))}" alt="MaDoc"/>'
    if dark:
        images += f'<img class="md-logo-dark" src="{logo_uri(str(dark))}" alt="MaDoc"/>'
    return images


# the logo of the theme shown (gradio puts the class "dark" on the page in the dark theme); also on the login page,
# where the CSS of the app is not loaded yet
LOGO_CSS = """
.md-logo-dark {display:none !important;}
.dark .md-logo-dark {display:inline-block !important;} .dark .md-logo:not(.md-boxed) {display:none !important;}
.dark .md-boxed {background:#fff; border-radius:6px; padding:3px 8px; box-sizing:border-box;}
.md-login img {display:block; width:80%; max-width:340px; margin:4px auto 14px;}
.dark .md-login .md-logo-dark {display:block !important;}
"""


def title_html(logos):
    """The top bar's title: the logo with the name, or the name in letters when there is no logo."""
    images = logo_images(logos)
    return f'<div class="md-title">{images}</div>' if images else '<div class="md-title">MaDoc</div>'


def login_message(logos):
    """The login page: the logo with the name above the message."""
    text = "MaDoc: log in with the name and password you were given."
    images = logo_images(logos)
    if not images:
        return text
    return (f'<style>{LOGO_CSS}</style><div class="md-login">{images}'
            f'<p style="text-align:center">{html.escape(text)}</p></div>')


def sample_images():
    """The example scans of samples/, sorted by name (1.jpg, 2.jpg ...)."""
    if not SAMPLES_DIR.is_dir():
        return []
    return sorted(str(p) for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)

CSS = """
.md-bar {align-items:center !important; gap:8px !important; flex-wrap:nowrap !important;}
.md-bar button {min-width:7.5em !important; flex:0 0 auto !important; white-space:nowrap;}
.md-bar .md-help button, .md-bar button.md-help {min-width:2.6em !important;}
.md-title {text-align:right; font-weight:700; font-size:1.25em; padding-right:6px;}
.md-title img {height:40px; width:auto; display:inline-block; vertical-align:middle;}
.md-tools {gap:6px !important; justify-content:flex-end; flex-wrap:nowrap !important;}
.md-files {gap:8px !important; justify-content:flex-start; flex-wrap:nowrap !important;}
.md-files button {min-width:11em !important; flex:0 0 auto !important; white-space:nowrap;}
.md-tools button {min-width:2.6em !important; max-width:9em; flex:0 0 auto !important;}
.md-scroll {height:62vh; overflow:auto; border:1px solid #8884; border-radius:6px; background:#fff;}
.md-scroll svg, .md-scroll img {display:block; height:auto;}
.md-poly {fill-opacity:.2; stroke-width:3; vector-effect:non-scaling-stroke; transition:fill-opacity .1s;}
.md-base {fill:none; stroke:#111; stroke-opacity:.5; stroke-width:1.5; vector-effect:non-scaling-stroke;}
.md-text {height:62vh; overflow:auto; border:1px solid #8884; border-radius:6px; padding:4px 0;}
.md-line {display:grid; grid-template-columns:2.4em 1fr 3.6em; gap:8px; padding:3px 10px; align-items:center;
          border-bottom:1px solid #8882;}
.md-line.low {box-shadow:inset 4px 0 #dc2626;}
.md-num {color:#888; font-size:.8em; text-align:right;}
.md-t {direction:rtl; text-align:right; font-size:1.12em; line-height:1.6; unicode-bidi:plaintext;}
.md-conf {font-size:.75em; text-align:center; border-radius:4px; padding:1px 0; color:#111;}
.md-table {height:62vh; overflow:auto; border:1px solid #8884; border-radius:6px;}
.md-table table {width:100%; border-collapse:collapse; font-size:.92em;}
.md-table th {position:sticky; top:0; background:var(--background-fill-secondary, #f3f4f6); text-align:left;
              padding:6px 8px; z-index:1;}
.md-table td {padding:5px 8px; border-bottom:1px solid #8882; vertical-align:top;}
.md-table td.ar {direction:rtl; text-align:right; unicode-bidi:plaintext;}
.md-table .key {color:#888; font-size:.72em; white-space:nowrap;}
.md-table td:first-child {width:30%;}
.md-table tr.group td {font-weight:700; background:#8881; padding-top:8px;}
.md-table tr.error td {background:#dc262618;} .md-table tr.warning td {background:#f59e0b1c;}
.md-console {max-height:20vh; min-height:5.5em; overflow:auto; font-family:ui-monospace, Menlo, Consolas, monospace; font-size:.85em;
             border:1px solid #8884; border-radius:6px; padding:6px 10px; background:var(--background-fill-secondary, #f9fafb);}
.md-console div {padding:1px 0; white-space:pre-wrap; unicode-bidi:plaintext;}
.md-console .sum {font-weight:700; padding-bottom:4px;} .md-console .ok {color:#16a34a;} .md-console .bad {color:#dc2626;}
.md-console .error {color:#dc2626;} .md-console .warning {color:#b45309;} .md-console .low {color:#b45309;}
.md-console .note {color:#6b7280;}
.md-empty {color:#888; padding:24px; text-align:center;}
"""


# texts of the gradio components in English, French and Arabic (gradio picks the page's language)
I18N = gradio_i18n(model=SPELL_MODEL)

# the record as a table: (group, label, Arabic field, French field); group and label are keys of ui_text.TEXT
RECORD_ROWS = [
    ("g_act", "f_act_number", "NUMACTE", None),
    ("g_child", "f_first_name", "PRENOMAR", "PRENOM"),
    ("g_child", "f_family_name", "NOMAR", "NOM"),
    ("g_child", "f_sex", "SEXE", None),
    ("g_child", "f_birth_date_both", "DATENAISSAR", "DATENAISSFR"),
    ("g_child", "f_hour", "HEURENAISS", None),
    ("g_child", "f_minute", "MINNAISS", None),
    ("g_child", "f_birthplace", "LIEUNAISS", "LIEUNAISSFR"),
    ("g_child", "f_nationality", "NATIONALITE", "NATIONALITEFR"),
    ("g_father", "f_name", "PRENOMPEREAR", "PRENOMPEREFR"),
    ("g_father", "f_lineage", "NOMPEREAR", "NOMPEREFR"),
    ("g_father", "f_birth_date", "DATENAISSPEREAR", "DATENAISSPEREFR"),
    ("g_father", "f_birthplace", "ADRESSENAISSPERE", None),
    ("g_father", "f_nationality", "NATIONALITEPERE", "NATIONALITEPEREFR"),
    ("g_father", "f_job", "PROFESSIONPERE", "PROFESSIONPEREFR"),
    ("g_mother", "f_name", "PRENOMMEREAR", "PRENOMMEREFR"),
    ("g_mother", "f_lineage", "NOMMEREAR", "NOMMEREFR"),
    ("g_mother", "f_birth_date", "DATENAISSMEREAR", "DATENAISSMEREFR"),
    ("g_mother", "f_birthplace", "ADRESSENAISSMERE", None),
    ("g_mother", "f_nationality", "NATIONALITEMERE", "NATIONALITEMEREFR"),
    ("g_mother", "f_job", "PROFESSIONMERE", "PROFESSIONMEREFR"),
    ("g_declaration", "f_address", "ADRESSEPARENT", None),
    ("g_declaration", "f_declaration", "TEXTEDECLA", None),
    ("g_declaration", "f_declaration_date", "DATEDECLAAR", "DATEDECLAFR"),
    ("g_declaration", "f_year", "ANNEEDECLA", None),
    ("g_declaration", "f_officer", "EMPLOYECREANT", "EMPLOYECREANTFR"),
    ("g_declaration", "f_delegation", "BECNAME", "BECNAMEFR"),
]
# the zone of the form where each field is read (to find the line it comes from)
FIELD_ZONES = {"child": ("NUMACTE", "DATENAISSAR", "DATENAISSFR", "HEURENAISS", "MINNAISS", "LIEUNAISS",
                         "PRENOMAR", "PRENOM", "SEXE", "NATIONALITE", "NOMAR", "NOM"),
               "father": ("PRENOMPEREAR", "NOMPEREAR", "NATIONALITEPERE", "ADRESSENAISSPERE", "DATENAISSPEREAR",
                          "DATENAISSPEREFR", "PROFESSIONPERE"),
               "mother": ("PRENOMMEREAR", "NOMMEREAR", "NATIONALITEMERE", "ADRESSENAISSMERE", "DATENAISSMEREAR",
                          "DATENAISSMEREFR", "PROFESSIONMERE"),
               "admin": ("ADRESSEPARENT", "TEXTEDECLA", "DATEDECLAAR", "DATEDECLAFR", "ANNEEDECLA",
                         "EMPLOYECREANT")}


# ---------------------------------------------------------------------------------------------------------
# files of a run
# ---------------------------------------------------------------------------------------------------------
def new_run_folder():
    """A new folder for the files of one run; the folders older than KEEP_SECONDS are deleted first."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    limit = time.time() - KEEP_SECONDS
    for old in RUNS_DIR.iterdir():
        try:
            if old.stat().st_mtime < limit:
                shutil.rmtree(old, ignore_errors=True)
        except OSError:
            pass
    return Path(tempfile.mkdtemp(prefix="run_", dir=RUNS_DIR))


def scan_to_rgb(image):
    """The scan in 8-bit RGB. A 16-bit or 32-bit scan (TIFF from a scanner) is scaled down to 0-255 first:
    Pillow's own conversion clips every value above 255, which turns such a scan white."""
    if image.mode.startswith("I;16") or image.mode in ("I", "F"):
        import numpy as np
        values = np.asarray(image, dtype=np.float64)
        low, high = float(values.min()), float(values.max())
        if image.mode.startswith("I;16") and high > 255:
            low, high = 0.0, 65535.0                 # the full range of the scanner: keep the contrast
        scaled = (values - low) * (255.0 / (high - low)) if high > low else np.zeros_like(values)
        image = Image.fromarray(scaled.clip(0, 255).astype(np.uint8), "L")
    return image.convert("RGB")


def prepare_image(image_path, folder):
    """The scan as kraken and the viewer must see it, in the folder of the run: turned upright (phone photos
    store the rotation in the EXIF data, which kraken ignores) and in a mode kraken reads; an unchanged image is
    copied as it is. The run then no longer depends on gradio's copy of the upload (deleted after an hour)."""
    try:
        image = Image.open(image_path)
    except (Image.UnidentifiedImageError, Image.DecompressionBombError) as e:
        raise gr.Error(f"This file cannot be read as a scan ({type(e).__name__}): upload a JPEG, PNG or TIFF image.")
    with image:
        rotated = image.getexif().get(0x0112, 1) not in (None, 1)   # 0x0112 = EXIF orientation
        if not rotated and image.mode in ("RGB", "L", "1"):
            copied = Path(folder) / ("scan" + Path(image_path).suffix.lower())
            shutil.copyfile(image_path, copied)
            return str(copied)
        prepared = Path(folder) / "scan.png"
        scan_to_rgb(ImageOps.exif_transpose(image)).save(prepared)
    return str(prepared)


def keep_alive(doc):
    """The run is in use: its folder is kept for KEEP_SECONDS more (new_run_folder deletes the folders not used
    for that long). A folder already deleted (the page was left open longer) cannot be rebuilt: the person is
    asked to import the scan again."""
    folder = Path(doc["folder"])
    if not folder.is_dir() or not Path(doc["image"]).is_file():
        raise gr.Error("This scan was deleted from the server (files are kept one hour after their last use): "
                       "import it again.")
    os.utime(folder)


def confidence_color(confidence, threshold):
    """green (sure) > amber > red (under the threshold); grey when kraken gave no score."""
    if confidence is None:
        return "#9ca3af"
    if confidence < threshold:
        return "#dc2626"
    if confidence < min(0.98, threshold + 0.1):
        return "#f59e0b"
    return "#16a34a"


def encode_image(path):
    """(data URI of the scan reduced for display, original width, original height)"""
    return _encode_image(str(path), Path(path).stat().st_mtime)


@lru_cache(maxsize=8)
def _encode_image(path, mtime):
    """Cached by (path, modification time): zooming or moving the threshold does not encode the scan again."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        shown = image.copy()
        shown.thumbnail((DISPLAY_MAX_SIDE, DISPLAY_MAX_SIDE))
        buffer = io.BytesIO()
        shown.save(buffer, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"), width, height


# ---------------------------------------------------------------------------------------------------------
# which lines every field of the record comes from
# ---------------------------------------------------------------------------------------------------------
def field_lines(text, parsed):
    """{field: [line indexes]} for the values the parser read (searched in the text, in their zone first)."""
    if not parsed or not text.strip():
        return {}
    flat, index = flatten(text)
    zones = dict(zip(("child", "father", "mother", "admin"), find_zones(flat)))
    zone_of = {field: zone for zone, fields in FIELD_ZONES.items() for field in fields}
    out = {}
    for field, value in parsed.items():
        if not isinstance(value, str) or not value.strip() or field in ("BECNAME", "BECNAMEFR"):
            continue
        value = normalize_text(value)
        candidates = [value] + ([" ".join(value.split()[:2])] if len(value.split()) > 2 else [])
        zone = zones.get(zone_of.get(field, ""))
        position, length = -1, 0
        for candidate in candidates:
            if zone is not None and zone.text:
                position = flat.find(candidate, zone.start, zone.end + len(candidate))
            if position < 0:
                position = flat.find(candidate)
            if position >= 0:
                length = len(value) if candidate == value else len(candidate)
                break
        if position < 0 or not index:
            continue
        first = index[position]
        last = index[min(position + max(length, 1) - 1, len(index) - 1)]
        out[field] = list(range(text.count("\n", 0, first), text.count("\n", 0, last) + 1))
    return out


def line_classes(lines):
    return " ".join(f"ln-{i + 1}" for i in sorted(set(lines)))


def link_css(count):
    """Hovering any element of line n highlights every element of line n (polygon, text line, record row)."""
    return "".join(
        f"body:has(.ln-{n}:hover) .ln-{n}{{background:#3b82f633 !important; fill-opacity:.6 !important;"
        f" stroke-width:6 !important;}}"
        for n in range(1, count + 1)
    )


# ---------------------------------------------------------------------------------------------------------
# the panels
# ---------------------------------------------------------------------------------------------------------
def text_lines(doc):
    return doc["history"][doc["position"]].rstrip("\n").split("\n") if doc else []


def segmentation_linked(doc):
    """The polygons are the text lines only while the text has as many lines as the segmentation."""
    return bool(doc) and len(text_lines(doc)) == len(doc["lines"])


def empty(message):
    return f'<div class="md-empty">{message}</div>'


def image_html(doc, zoom):
    if not doc:
        return empty(tr("empty_import", escape=False))
    if not Path(doc["image"]).is_file():
        return empty(tr("empty_deleted"))
    uri, width, _ = encode_image(doc["image"])
    return f'<div class="md-scroll"><img src="{uri}" style="width:{zoom * 100:.0f}%" alt="scan"/></div>'


def segmentation_html(doc, zoom, threshold):
    if not doc:
        return empty(tr("empty_import", escape=False))
    if not Path(doc["image"]).is_file():
        return empty(tr("empty_deleted"))
    uri, width, height = encode_image(doc["image"])
    linked = segmentation_linked(doc)
    shapes = []
    for number, line in enumerate(doc["lines"], 1):
        color = confidence_color(line.get("confidence"), threshold)
        score = line.get("confidence")
        label = "n/a" if score is None else f"{score:.0%}"
        css = f"md-poly ln-{number}" if linked else "md-poly"
        if line.get("polygon"):
            points = " ".join(f"{x},{y}" for x, y in line["polygon"])
            shapes.append(f'<polygon class="{css}" points="{points}" fill="{color}" stroke="{color}">'
                          f'<title>{number}: {html.escape(line["text"])} ({label})</title></polygon>')
        if line.get("baseline"):
            points = " ".join(f"{x},{y}" for x, y in line["baseline"])
            shapes.append(f'<polyline class="md-base" points="{points}"/>')
    return (f'<div class="md-scroll"><svg viewBox="0 0 {width} {height}" style="width:{zoom * 100:.0f}%" '
            f'xmlns="http://www.w3.org/2000/svg"><image href="{uri}" width="{width}" height="{height}"/>'
            f'{"".join(shapes)}</svg></div>')


def text_html(doc, threshold):
    """The transcription, one row per line (number, text, recognition confidence when it is still known)."""
    if not doc:
        return empty(tr("empty_text"))
    lines = text_lines(doc)
    linked = segmentation_linked(doc)
    rows = []
    for number, line in enumerate(lines, 1):
        score = doc["lines"][number - 1].get("confidence") if linked else None
        conf = ""
        if score is not None:
            color = confidence_color(score, threshold)
            conf = f'<span class="md-conf" style="background:{color}33; border:1px solid {color}">{score:.0%}</span>'
        low = " low" if score is not None and score < threshold else ""
        rows.append(f'<div class="md-line ln-{number}{low}"><span class="md-num">{number}</span>'
                    f'<span class="md-t">{html.escape(line)}</span><span>{conf}</span></div>')
    return f'<style>{link_css(len(lines))}</style><div class="md-text">{"".join(rows)}</div>'


def show(value):
    if value is None or value == "":
        return ""
    return html.escape(str(value))


def record_html(doc):
    """The record as a table: one row per field (Arabic and French), the rows with an issue coloured."""
    if not doc or not doc.get("record"):
        return empty(tr("empty_record"))
    record, mapping = doc["record"], doc.get("field_lines", {})
    level = {}
    messages = {}
    for issue in doc["report"]["issues"]:
        field = issue["field"]
        if level.get(field) != "error":
            level[field] = issue["level"]
        messages.setdefault(field, []).append(f"{issue['level']}: {issue['message']}")
    rows, group = [], None
    for row_group, label, arabic, french in RECORD_ROWS:
        if row_group != group:
            rows.append(f'<tr class="group"><td colspan="3">{tr(row_group)}</td></tr>')
            group = row_group
        fields = [f for f in (arabic, french) if f]
        lines = [n for f in fields for n in mapping.get(f, [])]
        if french:   # the French value comes from the Arabic one: same lines
            lines += mapping.get(arabic, [])
        css = " ".join(filter(None, [line_classes(lines),
                                     "error" if "error" in (level.get(f) for f in fields) else
                                     "warning" if "warning" in (level.get(f) for f in fields) else ""]))
        title = html.escape(" | ".join(m for f in fields for m in messages.get(f, [])))
        keys = " · ".join(fields)
        rows.append(f'<tr class="{css}" title="{title}"><td>{tr(label)}<div class="key">{keys}</div></td>'
                    f'<td class="ar">{show(record.get(arabic))}</td><td>{show(record.get(french)) if french else ""}</td></tr>')
    return (f'<div class="md-table"><table><thead><tr><th>{tr("th_field")}</th><th>{tr("th_value")}</th>'
            f'<th>{tr("th_french")}</th></tr>'
            f'</thead><tbody>{"".join(rows)}</tbody></table></div>')


def log_entry(key, css="note", **values):
    """A console line of the run's log (kept as a key, shown in the page's language)."""
    return (css, key, values)


def console_html(doc, threshold):
    """Validation summary, issues, lines under the confidence threshold, spelling and parser notes, run log."""
    if not doc:
        return f'<div class="md-console"><div class="note">{tr("c_ready")}</div></div>'
    entries = []   # (css, html)
    if doc.get("report"):
        report = doc["report"]
        state = "ok" if report["valid"] else "bad"
        label = TEXT_STATE["valid" if report["valid"] else "invalid"]
        entries.append(("sum " + state, tr_summary(label, report["errors"], report["warnings"])))
        for issue in report["issues"]:
            entries.append((issue["level"], tr("c_error" if issue["level"] == "error" else "c_warning",
                                              field=issue["field"], message=issue["message"])))
    if segmentation_linked(doc):
        current = text_lines(doc)
        for number, line in enumerate(doc["lines"], 1):
            score = line.get("confidence")
            if score is not None and score < threshold:
                entries.append((f"low ln-{number}", tr("c_low", number=number, score=f"{score:.0%}",
                                                       text=current[number - 1])))
    elif doc.get("lines"):
        entries.append(("note", tr("c_unlinked")))
    for note in doc.get("notes", []):
        entries.append(("note", html.escape(note)))
    for css, key, values in doc.get("log", []):
        entries.append((css, tr(key, **values)))
    body = "".join(f'<div class="{css}">{content}</div>' for css, content in entries)
    return f'<div class="md-console">{body}</div>'


TEXT_STATE = {"valid": "c_valid", "invalid": "c_invalid"}


def tr_summary(state_key, errors, warnings):
    """'VALID · 0 error(s) · 2 warning(s)' in the three languages (the state word translated too)."""
    from ui_text import LANGUAGES, TEXT
    parts = []
    for n, language in enumerate(LANGUAGES):
        line = TEXT["c_summary"][n].format(state=TEXT[state_key][n], errors=errors, warnings=warnings)
        parts.append(f'<span class="l-{language}">{html.escape(line)}</span>')
    return f'<span class="tr">{"".join(parts)}</span>'


# ---------------------------------------------------------------------------------------------------------
# spelling review (kept from the previous interface, now inside the Text panel)
# ---------------------------------------------------------------------------------------------------------
ACCEPTED_COLOR, REFUSED_COLOR, FLAG_COLOR = "#16a34a", "#dc2626", "#f59e0b"
NO_SPELLING = {"text": "", "changes": [], "accepted": [], "selected": {}, "segments": []}


def resolved(state):
    """The changes with, for a change that has several options (jobs), the option chosen in the tab."""
    selected = state.get("selected") or {}
    out = []
    for i, change in enumerate(state["changes"]):
        options = change.get("options")
        if options:
            change = {**change, "new": options[selected.get(i, 0) % len(options)]}
        out.append(change)
    return out


def isolate(text):
    """Arabic inside an English sentence, isolated (Unicode FSI ... PDI): without it the browser shows
    "صباخ → صباغ" in the wrong order around the arrow."""
    return "\u2068" + text + "\u2069"


def spelling_review(state):
    """Spelling state -> (HighlightedText with every proposed change, details in Markdown, state with the
    index of the change behind every highlighted piece)."""
    changes, accepted = resolved(state), set(state["accepted"])
    if not state["text"]:
        return gr.HighlightedText(value=[]), tr("spell_off"), state
    by_position = {(c["line"], c["start"]): i for i, c in enumerate(changes) if c["old"]}
    pieces, segments, color_map = [], [], {}
    for line_index, line in enumerate(state["text"].split("\n")):
        # the line is shown exactly as written; only the words of a change are highlighted
        units, written = line_units(line), 0
        for position, unit in enumerate(units):
            index = by_position.get((line_index, position))
            if index is None:
                continue
            change = changes[index]
            begin, finish = unit["begin"], units[change["end"] - 1]["finish"]
            if begin > written:
                pieces.append((line[written:begin], None))
                segments.append(None)
            if change.get("flag"):   # a mismatch without a correction: shown, nothing to accept
                label = change["context"]
                color_map[label] = FLAG_COLOR
            else:
                proposal = " ".join(change["new"]) or "(removed)"
                if change.get("check") and index not in accepted:
                    # a disagreement inside the certificate, with its correction: orange, applied by a click
                    label = "≠ " + proposal
                    color_map[label] = FLAG_COLOR
                else:
                    label = ("✓ " if index in accepted else "✗ ") + proposal
                    color_map[label] = ACCEPTED_COLOR if index in accepted else REFUSED_COLOR
            pieces.append((line[begin:finish], label))
            segments.append(index)
            written = finish
        pieces.append((line[written:] + "\n", None))
        segments.append(None)
    shown = [c for c in changes if c["old"] and not c.get("flag")]
    if not shown and not any(c.get("flag") for c in changes):
        details = tr("spell_none")
    else:
        rows = []
        for i, change in enumerate(changes):
            if not change["old"]:
                continue
            if change.get("flag"):
                rows.append(f"- ≠ line {change['line'] + 1}: **{isolate(' '.join(change['old']))}** "
                            f"{change['reason']}: " + ("no correction found, check the scan"
                                                        if change.get("source") == "names" else "check the scan"))
                continue
            mark = "✓" if i in accepted else ("≠" if change.get("check") else "✗")
            if change.get("check"):
                rule = f"{change['context']}; {change['reason']}" + (
                    " (click again for the next one)" if len(change.get("options") or []) > 1
                    and "click again" not in change["reason"] else "")
            elif change.get("options") and len(change["options"]) > 1:
                where = {"jobs.json": "jobs.json", "cities.json": "cities.json", "names.json": "names.json",
                         "dates": "the Hijri date"}.get(change.get("source"), "date_time.json")
                rule = (f"from {where}, several possible: " + " / ".join(isolate(" ".join(o)) for o in change["options"])
                        + " (click again for the next one)")
            elif change.get("source") == "names" and change["accepted"]:
                rule = f"the child's Arabic and Latin names did not match; {change['context']}"
            elif change.get("source") == "names":
                rule = (f"the child's Arabic and Latin names did not match; proposed {change['context']}"
                        " (names are never applied automatically: click to apply)")
            elif change.get("source") in ("cities.json", "names.json") and not change["accepted"]:
                rule = f"{change['context']}; {change['reason']} (click to apply)"
            elif change.get("source") == "jobs.json":
                rule = "from jobs.json" + (", " + change["context"] if change.get("context") else "") + (
                    "" if change["accepted"] else f"; refused by the rules: {change['reason']}")
            elif change["accepted"]:
                rule = change.get("context") or "accepted by the rules"
            else:
                rule = f"refused by the rules: {change['reason']}"
            rows.append(f"- {mark} line {change['line'] + 1}: **{isolate(' '.join(change['old']))}** → "
                        f"**{isolate(' '.join(change['new']) or '(removed)')}** ({rule})")
        details = tr("spell_count", done=len(accepted), total=len(shown)) + "\n\n" + "\n".join(rows)
    state = {**state, "segments": segments}
    return gr.HighlightedText(value=pieces, color_map=color_map), details, state


def spelling_notes(state):
    """The changes with the decision taken (by the rules, or by the person in the spelling review)."""
    accepted, notes = set(state["accepted"]), []
    for i, change in enumerate(resolved(state)):
        chosen = i in accepted
        if chosen == change["accepted"]:
            notes.append(describe(change))
        elif chosen:   # accepted by the person although the rules refused it
            notes.append(describe({**change, "accepted": True})
                         + f" (accepted in the spelling review; the rules refused it: {change['reason']})")
        else:
            notes.append(describe({**change, "accepted": False, "reason": "refused in the spelling review"}))
    return "\n".join(f"[spelling] {note}" for note in notes) + "\n\n" if notes else ""


def toggle_change(state, evt: gr.SelectData):
    """A click on a highlighted change accepts / refuses it. A change with several options (a misread job close
    to two jobs) goes: refused -> first option -> second option ... -> refused."""
    segments = state.get("segments") or []
    index = segments[evt.index] if isinstance(evt.index, int) and evt.index < len(segments) else None
    if index is not None and state["changes"][index].get("flag"):
        index = None   # a flagged name has nothing to accept
    if index is not None:
        accepted, selected = set(state["accepted"]), dict(state.get("selected") or {})
        options = state["changes"][index].get("options") or [state["changes"][index]["new"]]
        if index not in accepted:
            accepted.add(index)
            selected[index] = 0
        elif selected.get(index, 0) + 1 < len(options):
            selected[index] = selected.get(index, 0) + 1
        else:
            accepted.discard(index)
            selected[index] = 0
        state = {**state, "accepted": sorted(accepted), "selected": selected}
    return spelling_review(state)


def set_all(state, accept):
    """accept=True: back to the choice of the rules; False: refuse every change.

    There is no "accept all": it would also accept the changes the rules refused (removed words, names, Latin
    text invented by the model); those are accepted one by one, by clicking them.
    """
    accepted = [i for i, c in enumerate(state["changes"]) if c["accepted"] and not c.get("flag")] if accept else []
    return spelling_review({**state, "accepted": accepted, "selected": {}})


# ---------------------------------------------------------------------------------------------------------
# processing
# ---------------------------------------------------------------------------------------------------------
def transcription(lines):
    return "\n".join(line["text"] for line in lines) + "\n"


def parse_text(doc, text, options):
    """Run the parsing chain on `text` and store the record, the report, the notes and the field lines."""
    stages = {}
    record, report, notes = run_pipeline(
        text, use_ner=options["ner"], use_translation=options["translate"], use_french_ner=options["french_ner"],
        stages=stages,
    )
    doc["auto_record"] = record
    record, report, record_notes = keep_record_edits(doc, record, report, options)
    doc.update(record=record, report=report, parser_notes=[f"[parser] {n}" for n in notes],
               field_lines=field_lines(text, stages.get("parsed")), record_notes=record_notes)
    doc["notes"] = doc.get("spelling_notes", []) + doc["parser_notes"] + record_notes
    write_files(doc, text)
    return doc


# ---------------------------------------------------------------------------------------------------------
# the record edited by hand (JSON tab)
# ---------------------------------------------------------------------------------------------------------
# the French fields written from an Arabic one: by the dictionaries (mapper.py) or by the local model
FRENCH_OF = {**translator.NAME_FIELDS, **{arabic: french for arabic, (french, _) in translator.OTHER_FIELDS.items()},
             "NATIONALITE": "NATIONALITEFR", "NATIONALITEPERE": "NATIONALITEPEREFR",
             "NATIONALITEMERE": "NATIONALITEMEREFR"}
TRANSLATED = set(FRENCH_OF.values())


def record_from_json(text, template=TEMPLATE):
    """The JSON written in the editor -> the record (fields in the template's order); ValueError with what is
    wrong (the JSON syntax with its line, a field unknown or missing, a value that is a list or an object)."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON: line {e.lineno}, column {e.colno}: {e.msg}") from e
    if not isinstance(data, dict):
        raise ValueError("JSON: the record must be an object { ... }")
    unknown = [key for key in data if key not in template]
    missing = [key for key in template if key not in data]
    if unknown or missing:
        raise ValueError("; ".join(filter(None, [
            unknown and "unknown field(s): " + ", ".join(unknown),
            missing and "missing field(s): " + ", ".join(missing)])))
    wrong = [key for key, value in data.items()
             if not (value is None or isinstance(value, (str, int, float, bool)))]
    if wrong:
        raise ValueError("a value must be a text or a number: " + ", ".join(wrong))
    return {key: data[key] for key in template}


def report_of(record, options):
    """The validation report of a record (the French fields checked when the translation can run)."""
    return validate(record, check_translation=options["translate"] and translator.is_available())


def edit_record(doc, before, record, options):
    """The record as the person wrote it in the editor. `before` is the record the editor started from: only
    the fields changed since then are taken (the record may have been parsed again meanwhile).

    Every field changed by hand is remembered with the automatic value it replaced: it is kept when the text is
    parsed again, as long as the text gives the same automatic value (see keep_record_edits)."""
    doc = dict(doc)
    current = dict(doc["record"])
    auto = doc.get("auto_record") or current
    edits = dict(doc.get("record_edits") or {})
    changed = [field for field in record if record[field] != before.get(field)]
    for field in changed:
        current[field] = record[field]
        if record[field] == auto.get(field):
            edits.pop(field, None)           # back to the value read in the text: no longer an edit
        else:
            edits[field] = {"auto": auto.get(field), "value": record[field]}
    doc.update(record=current, record_edits=edits, report=report_of(current, options))
    if changed:
        doc["log"] = doc.get("log", []) + [log_entry("c_record_edit", fields=", ".join(changed))]
    keep_alive(doc)
    write_files(doc, doc["history"][doc["position"]])
    return doc, changed


def translate_record(record, keep=()):
    """The French fields made again from the Arabic ones (dictionaries, then the local model); the fields of
    `keep` (French values written by hand) are not touched."""
    out = dict(record)
    for french in TRANSLATED - set(keep):
        out[french] = ""
    out = translator.translate(map_fields(out))
    out.update({field: record[field] for field in keep})
    return out


def keep_record_edits(doc, record, report, options):
    """The hand edits of the record applied to a record just parsed -> (record, report, notes).

    An edit is kept while the text gives the value it replaced; when the text gives another one (the text was
    corrected), the text wins and the edit is dropped. An Arabic value kept from an edit gets its French again."""
    edits = doc.get("record_edits") or {}
    if not edits:
        return record, report, []
    kept = {field: edit for field, edit in edits.items() if record.get(field) == edit["auto"]}
    dropped = sorted(set(edits) - set(kept))
    doc["record_edits"] = kept
    notes = [f"[record] value read in the corrected text used instead of the hand edit: {field}" for field in dropped]
    if not kept:
        return record, report, notes
    record = {**record, **{field: edit["value"] for field, edit in kept.items()}}
    if options["translate"] and any(field in FRENCH_OF for field in kept) and translator.is_available():
        record = translate_record(record, keep=[field for field in kept if field in TRANSLATED])
    notes.append("[record] kept from the hand edits: " + ", ".join(sorted(kept)))
    return record, report_of(record, options), notes


def translate_document(doc, options):
    """文 of the JSON tab: the French fields translated again from the Arabic of the record (with its hand
    edits); the French values written by hand are kept."""
    doc = dict(doc)
    keep = [field for field in (doc.get("record_edits") or {}) if field in TRANSLATED]
    record = translate_record(doc["record"], keep)
    doc.update(record=record, report=validate(record, check_translation=True))
    doc["log"] = doc.get("log", []) + [log_entry("c_translated", model=translator.MODEL)]
    keep_alive(doc)
    write_files(doc, doc["history"][doc["position"]])
    return doc


MAX_NAME_LENGTH = 120     # characters of an uploaded file name kept (the file systems allow 255 bytes)


def file_name(name):
    """The uploaded scan's file name, without its folder and the characters a file name cannot have ("." and
    ".." are not names; a very long name is shortened, its extension kept)."""
    name = re.split(r"[\\/]", str(name or "scan.png"))[-1]
    name = re.sub(r'[<>:"|?*\x00-\x1f]', "_", name).strip()
    if not name.strip("."):
        return "scan.png"
    if len(name) > MAX_NAME_LENGTH:
        suffix = Path(name).suffix[:10]
        name = name[:MAX_NAME_LENGTH - len(suffix)].rstrip(". ") + suffix
    return name


def page_xml(xml, image_name):
    """The segmentation PageXML naming its image by the file name alone ("1.jpg", not the path on the server)."""
    return re.sub(r'imageFilename="[^"]*"', f'imageFilename="{html.escape(image_name, quote=True)}"', xml)


def write_files(doc, text):
    """The downloads of the run, all named after the uploaded scan ("1.jpg" -> 1.xml, 1.txt, 1.json), and the
    zip of the four (Save): the original scan, its segmentation (PageXML), the transcription and the record."""
    keep_alive(doc)
    name = file_name(doc.get("name"))
    stem = Path(name).stem
    out = Path(doc["folder"]) / "download"
    out.mkdir(exist_ok=True)
    original = Path(doc.get("original") or doc["image"])
    files = {"xml": out / f"{stem}.xml", "text": out / f"{stem}.txt", "json": out / f"{stem}.json"}
    files["xml"].write_text(page_xml(doc["xml"], name), encoding="utf-8")
    files["text"].write_text(text, encoding="utf-8")
    write_json(doc["record"], files["json"])
    with zipfile.ZipFile(out / f"{stem}.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(original, name)
        for path in files.values():
            archive.write(path, path.name)
    doc["files"] = {"image": str(original), "segmentation": str(files["xml"]), "text": str(files["text"]),
                    "json": str(files["json"]), "zip": str(out / f"{stem}.zip")}


def options_of(ner, french_ner, translate, spellcheck):
    return {"ner": ner, "french_ner": french_ner, "translate": translate, "spellcheck": spellcheck}


def edited_words(before, after):
    """{(line, unit)} of the words of `after` a person wrote or changed by hand (units of line_units)."""
    from difflib import SequenceMatcher
    old, new = before.split("\n"), after.split("\n")
    out = set()
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if tag == "insert" or (tag == "replace" and i2 - i1 != j2 - j1):
            out.update((j, k) for j in range(j1, j2) for k in range(len(line_units(new[j]))))
        elif tag == "replace":   # the same number of lines: word by word
            for i, j in zip(range(i1, i2), range(j1, j2)):
                a = [u["text"] for u in line_units(old[i])]
                b = [u["text"] for u in line_units(new[j])]
                for op, _, _, k1, k2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
                    if op in ("replace", "insert"):
                        out.update((j, k) for k in range(k1, k2))
    return out


def person_refused(spelling):
    """The corrections the person refused in the review: {(line text, start, end, new words)}."""
    lines = spelling.get("text", "").split("\n")
    chosen = set(spelling.get("accepted") or [])
    return {(lines[c["line"]], c["start"], c["end"], tuple(c["new"]))
            for i, c in enumerate(spelling.get("changes") or [])
            if c["accepted"] and i not in chosen and c["line"] < len(lines)}


def check_spelling(doc, options, progress=None, text=None, protect=(), refused=()):
    """The spelling proposals on a text (the HTR text by default; local model); the rules' choice is applied.

    protect: {(line, unit)} of words written by hand: a correction on them is proposed in orange, never applied
    by itself (the person's edit wins over the rules). refused: corrections the person refused before (see
    person_refused), still refused."""
    base = doc["htr"] if text is None else text
    if not options["spellcheck"]:
        doc["spelling_notes"] = []
        return dict(NO_SPELLING), base
    if progress:
        progress(0.6, desc=f"Spelling check (local model {SPELL_MODEL})")
    changes, skipped = propose(base)
    lines = base.split("\n")
    for i, change in enumerate(changes):
        if not change["accepted"]:
            continue
        span = {(change["line"], k) for k in range(change["start"], max(change["end"], change["start"] + 1))}
        if span & set(protect):
            changes[i] = {**change, "accepted": False, "check": True,
                          "reason": "written by hand: click to apply the correction"}
        elif (lines[change["line"]], change["start"], change["end"], tuple(change["new"])) in refused:
            changes[i] = {**change, "accepted": False, "reason": "refused in the spelling review"}
    spelling = {"text": base, "changes": changes, "selected": {},
                "accepted": [i for i, c in enumerate(changes) if c["accepted"]], "segments": []}
    doc["spelling_notes"] = [f"[spelling] {n}" for n in skipped] + spelling_notes(spelling).strip().split("\n")
    doc["spelling_notes"] = [n for n in doc["spelling_notes"] if n]
    return spelling, apply_changes(base, changes)


def edit_text(doc, text, options, progress=None, spelling=None):
    """A hand edit: kept in the history, the spelling checked again on the edited text (the lines written by hand
    are not corrected by themselves), then parsed. -> (doc, spelling state)"""
    doc = dict(doc)
    before = doc["history"][doc["position"]]
    history = doc["history"][:doc["position"] + 1]
    if text != history[-1]:
        history.append(text)
    doc.update(history=history, position=len(history) - 1)
    doc["log"] = doc.get("log", []) + [log_entry("c_edit")]
    keep_alive(doc)
    spelling, corrected = check_spelling(doc, options, progress, text=text, protect=edited_words(before, text),
                                         refused=person_refused(spelling or {}))
    doc["log"] = doc["log"] + [log_entry("c_respell")]
    if corrected != text:   # the corrections of the rules: one more version (↶ gives the edit as typed)
        doc["history"] = doc["history"] + [corrected]
        doc["position"] = len(doc["history"]) - 1
    return parse_text(doc, corrected, options), spelling


def start_document(image_path, name, options, threshold, device, segmentation_model, recognition_model,
                   progress=None):
    """Scan -> segmentation -> recognition -> spelling -> parsing: the document state of the interface."""
    folder = new_run_folder()
    (folder / "original").mkdir()
    original = folder / "original" / file_name(name)
    shutil.copyfile(image_path, original)
    image_path = prepare_image(image_path, folder)
    if progress:
        progress(0.05, desc="Segmentation (kraken blla)")
    xml = segment_image(image_path, segmentation_model, device)
    if progress:
        progress(0.35, desc="Recognition (kraken HTR)")
    lines = recognize_lines(xml, image_path, recognition_model, device)
    if not lines:
        raise gr.Error("No text line was recognized on this image.")
    htr = transcription(lines)
    doc = {"folder": str(folder), "image": image_path, "original": str(original), "name": name, "xml": xml,
           "lines": lines, "htr": htr,
           "threshold": threshold, "log": [log_entry("c_file", name=name, count=len(lines))]}
    spelling, text = check_spelling(doc, options, progress)
    doc.update(history=[text], position=0)
    if progress:
        progress(0.8, desc="Parsing, mapping, translation, validation")
    parse_text(doc, text, options)
    return doc, spelling


def set_text(doc, text, options, note=None):
    """A new version of the text (edit, applied spelling): kept in the history, parsed again.

    note: a key of ui_text.TEXT for the console log ("c_edit", "c_spelling")."""
    doc = dict(doc)
    history = doc["history"][:doc["position"] + 1]
    if text != history[-1]:
        history.append(text)
    doc.update(history=history, position=len(history) - 1)
    if note:
        doc["log"] = doc.get("log", []) + [log_entry(note)]
    return parse_text(doc, text, options)


def move_in_history(doc, step, options):
    """Undo (step -1) / redo (+1): the previous / next version of the text, parsed again."""
    position = doc["position"] + step
    if not 0 <= position < len(doc["history"]):
        return doc
    doc = dict(doc, position=position)
    return parse_text(doc, doc["history"][position], options)


# ---------------------------------------------------------------------------------------------------------
# interface
# ---------------------------------------------------------------------------------------------------------
def make(component, **kwargs):
    """Create a gradio component, dropping the options this gradio version does not have (they changed a lot)."""
    allowed = inspect.signature(component.__init__).parameters
    return component(**{k: v for k, v in kwargs.items() if k in allowed})


def file_or_none(doc, key):
    return (doc.get("files") or {}).get(key) if doc else None


def build_app(device="cpu", segmentation_model=MODEL_SEGMENTATION, recognition_model=MODEL_RECOGNITION):
    # delete_cache: gradio's copy of every upload / download is removed after KEEP_SECONDS
    with make(gr.Blocks, title="MaDoc", delete_cache=(KEEP_SECONDS, KEEP_SECONDS), fill_width=True) as app:
        gr.HTML(f"<style>{CSS}{LOGO_CSS}{I18N_CSS}</style>")
        doc_state, spelling_state = gr.State(None), gr.State(dict(NO_SPELLING))
        zoom_state, options_open, help_open = gr.State(1.0), gr.State(False), gr.State(False)
        samples_open, files_open = gr.State(False), gr.State(False)
        view_state = gr.State("text")   # what the Text tab shows: "text", "edit" or "spelling"
        json_raw = gr.State(False)      # the JSON tab shows the table (False) or the raw JSON (True)
        json_editing = gr.State(None)   # {"folder", "record"} the JSON editor started from (None: not editing)

        # top bar ------------------------------------------------------------------------------------------
        with gr.Row(elem_classes="md-bar", equal_height=True):
            files_button = make(gr.Button, value=I18N("files"), size="sm", scale=0, min_width=110)
            samples = sample_images()
            save_button = make(gr.DownloadButton, label=I18N("save"), size="sm", scale=0, min_width=110,
                               interactive=False)
            options_button = make(gr.Button, value=I18N("options"), size="sm", scale=0, min_width=110)
            gr.HTML(title_html(find_logos()))
            help_button = make(gr.Button, value="ⓘ", size="sm", scale=0, min_width=48, elem_classes="md-help")
        with gr.Column(visible=False) as options_panel:
            with gr.Row():
                use_spellcheck = gr.Checkbox(True, label=I18N("opt_spellcheck"))
                use_ner = gr.Checkbox(True, label=I18N("opt_ner"))
                use_french_ner = gr.Checkbox(True, label=I18N("opt_french_ner"))
                use_translate = gr.Checkbox(True, label=I18N("opt_translate"))
            with gr.Row():
                threshold = gr.Slider(0.3, 0.99, value=0.8, step=0.01, label=I18N("opt_threshold"))
                rerun = make(gr.Button, value=I18N("rerun"), size="sm")
        with gr.Column(visible=False) as help_panel:
            gr.Markdown(I18N("help"))
        with gr.Column(visible=False) as files_panel:
            with gr.Row(elem_classes="md-files", equal_height=True):
                import_button = make(gr.UploadButton, label=I18N("import"), file_types=["image"],
                                     file_count="single", size="sm", scale=0, min_width=150)
                samples_button = make(gr.Button, value=I18N("choose_samples"), size="sm", scale=0, min_width=150,
                                      visible=bool(samples))
        with gr.Column(visible=False) as samples_panel:
            gr.Markdown(I18N("samples_hint"))
            samples_view = make(gr.Gallery, value=[(path, Path(path).stem) for path in samples], show_label=False,
                                columns=max(len(samples), 1), rows=1, height=230, object_fit="contain",
                                allow_preview=False, show_download_button=False, show_share_button=False,
                                interactive=False)

        # left: image / segmentation, right: text / json ---------------------------------------------------
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=360):
                with gr.Tabs():
                    with gr.Tab(I18N("tab_image")):
                        with gr.Row(elem_classes="md-tools"):
                            zoom_out_1 = make(gr.Button, value="−", size="sm")
                            zoom_in_1 = make(gr.Button, value="+", size="sm")
                            download_image = make(gr.DownloadButton, label="🖫", size="sm", interactive=False)
                        image_view = gr.HTML(image_html(None, 1.0))
                    with gr.Tab(I18N("tab_segmentation")):
                        with gr.Row(elem_classes="md-tools"):
                            zoom_out_2 = make(gr.Button, value="−", size="sm")
                            zoom_in_2 = make(gr.Button, value="+", size="sm")
                            download_segmentation = make(gr.DownloadButton, label="🖫", size="sm",
                                                         interactive=False)
                        segmentation_view = gr.HTML(segmentation_html(None, 1.0, 0.8))
            with gr.Column(scale=1, min_width=360):
                with gr.Tabs():
                    with gr.Tab(I18N("tab_text")):
                        with gr.Row(elem_classes="md-tools"):
                            undo = make(gr.Button, value="↶", size="sm")
                            redo = make(gr.Button, value="↷", size="sm")
                            spelling_button = make(gr.Button, value="℁", size="sm")
                            edit_button = make(gr.Button, value="🖉", size="sm")
                            download_text = make(gr.DownloadButton, label="🖫", size="sm", interactive=False, tooltip="Download Text")
                        text_view = gr.HTML(text_html(None, 0.8))
                        text_edit = make(gr.Textbox, show_label=False, lines=24, max_lines=40, rtl=True,
                                         visible=False, interactive=True)
                        with gr.Column(visible=False) as spelling_panel:
                            spelling_view = make(gr.HighlightedText, show_label=False, rtl=True, show_legend=False,
                                                 show_inline_category=True, combine_adjacent=False,
                                                 interactive=False)
                            with gr.Row():
                                back_to_rules = make(gr.Button, value=I18N("back_to_rules"), size="sm")
                                refuse_all = make(gr.Button, value=I18N("refuse_all"), size="sm")
                                apply_spelling = make(gr.Button, value=I18N("apply"), size="sm", variant="primary")
                            spelling_details = gr.Markdown("", rtl=False)
                    with gr.Tab(I18N("tab_json")):
                        with gr.Row(elem_classes="md-tools"):
                            json_mode = make(gr.Button, value=I18N("raw"), size="sm")
                            translate_button = make(gr.Button, value="文", size="sm")
                            json_edit_button = make(gr.Button, value="🖉", size="sm")
                            download_json = make(gr.DownloadButton, label="🖫", size="sm", interactive=False)
                        table_view = gr.HTML(record_html(None))
                        json_view = gr.JSON(visible=False)
                        json_editor = make(gr.Code, language="json", show_label=False, lines=26, max_lines=26,
                                           interactive=True, visible=False)
        console = gr.HTML(console_html(None, 0.8))

        # helpers -------------------------------------------------------------------------------------------
        panels = [image_view, segmentation_view, text_view, table_view, json_view, console,
                  save_button, download_image, download_segmentation, download_text, download_json]

        def render(doc, zoom, limit):
            ready = bool(doc)

            def download(key):
                return gr.update(value=file_or_none(doc, key), interactive=ready)

            return [image_html(doc, zoom), segmentation_html(doc, zoom, limit), text_html(doc, limit),
                    record_html(doc), (doc or {}).get("record"), console_html(doc, limit),
                    download("zip"), download("image"), download("segmentation"), download("text"),
                    download("json")]

        modes = [view_state, text_view, text_edit, spelling_panel, edit_button, spelling_button]

        def mode(view="text"):
            """Which content the Text tab shows: the lines, the editor or the spelling review."""
            return [view, gr.update(visible=view == "text"), gr.update(visible=view == "edit"),
                    gr.update(visible=view == "spelling"),
                    gr.update(value=I18N("done") if view == "edit" else "🖉", variant="primary" if view == "edit" else "secondary"),
                    gr.update(variant="primary" if view == "spelling" else "secondary")]

        option_inputs = [use_ner, use_french_ner, use_translate, use_spellcheck]
        review_outputs = [spelling_view, spelling_details, spelling_state]

        # events -------------------------------------------------------------------------------------------
        def on_upload(file, ner, french_ner, translate, spellcheck, limit, zoom, progress=gr.Progress()):
            if not file:
                raise gr.Error("Choose a scan.")
            path = file if isinstance(file, str) else getattr(file, "name", str(file))
            options = options_of(ner, french_ner, translate, spellcheck)
            doc, spelling = start_document(path, Path(path).name, options, limit, device, segmentation_model,
                                           recognition_model, progress)
            review = spelling_review(spelling)
            return [doc, *review, *render(doc, zoom, limit), *mode("text"), *closed_files()]

        def on_sample(ner, french_ner, translate, spellcheck, limit, zoom, evt: gr.SelectData,
                      progress=gr.Progress()):
            """A click on an example scan: processed like an upload (a copy of it goes to the run's folder)."""
            index = evt.index if isinstance(evt.index, int) else (evt.index or [0])[0]
            if not 0 <= index < len(samples):
                raise gr.Error("Choose a sample.")
            path = samples[index]
            options = options_of(ner, french_ner, translate, spellcheck)
            doc, spelling = start_document(path, Path(path).name, options, limit, device, segmentation_model,
                                           recognition_model, progress)
            return [doc, *spelling_review(spelling), *render(doc, zoom, limit), *mode("text"), *closed_files()]

        def closed_files():
            """After a scan is chosen: the Files choices and the samples are closed."""
            return [False, gr.update(visible=False), False, gr.update(visible=False)]

        files_outputs = [files_open, files_panel, samples_open, samples_panel]
        # Files: two choices, Import (a scan of this computer) and Choose from samples (the example scans)
        files_button.click(lambda is_open, shown: [not is_open, gr.update(visible=not is_open),
                                                   shown and not is_open, gr.update(visible=shown and not is_open)],
                           [files_open, samples_open], files_outputs).then(None, None, None, js=PERMISSION_JS)
        samples_button.click(lambda is_open: [not is_open, gr.update(visible=not is_open)],
                             [samples_open], [samples_open, samples_panel])
        samples_view.select(on_sample, [*option_inputs, threshold, zoom_state],
                            [doc_state, *review_outputs, *panels, *modes, *files_outputs]
                            ).then(None, None, None, js=notification_js())
        import_button.upload(on_upload, [import_button, *option_inputs, threshold, zoom_state],
                             [doc_state, *review_outputs, *panels, *modes, *files_outputs]
                             ).then(None, None, None, js=notification_js())

        def on_rerun(doc, ner, french_ner, translate, spellcheck, limit, zoom, progress=gr.Progress()):
            if not doc:
                raise gr.Error("Import a scan first.")
            options = options_of(ner, french_ner, translate, spellcheck)
            doc = dict(doc)
            keep_alive(doc)
            spelling, text = check_spelling(doc, options, progress)
            # a new version in the history: the hand edits stay one ↶ away
            history = doc["history"][:doc["position"] + 1]
            if text != history[-1]:
                history.append(text)
            doc.update(history=history, position=len(history) - 1)
            parse_text(doc, text, options)
            return [doc, *spelling_review(spelling), *render(doc, zoom, limit), *mode("text")]

        rerun.click(on_rerun, [doc_state, *option_inputs, threshold, zoom_state],
                    [doc_state, *review_outputs, *panels, *modes]).then(None, None, None, js=notification_js())

        def on_zoom(doc, zoom, step, limit):
            index = min(range(len(ZOOM_STEPS)), key=lambda i: abs(ZOOM_STEPS[i] - zoom))
            zoom = ZOOM_STEPS[max(0, min(len(ZOOM_STEPS) - 1, index + step))]
            return zoom, image_html(doc, zoom), segmentation_html(doc, zoom, limit)

        for button, step in ((zoom_in_1, 1), (zoom_in_2, 1), (zoom_out_1, -1), (zoom_out_2, -1)):
            button.click(lambda doc, zoom, limit, s=step: on_zoom(doc, zoom, s, limit),
                         [doc_state, zoom_state, threshold], [zoom_state, image_view, segmentation_view])

        threshold.release(lambda doc, zoom, limit: [segmentation_html(doc, zoom, limit), text_html(doc, limit),
                                                    console_html(doc, limit)],
                          [doc_state, zoom_state, threshold], [segmentation_view, text_view, console])

        def on_edit(doc, editing_text, view, spelling, ner, french_ner, translate, spellcheck, limit, zoom,
                    progress=gr.Progress()):
            if not doc:
                raise gr.Error("Import a scan first.")
            if view != "edit":   # start editing
                return [doc, gr.update(value=doc["history"][doc["position"]]), *spelling_review(spelling),
                        *render(doc, zoom, limit), *mode("edit")]
            text = editing_text if editing_text.endswith("\n") else editing_text + "\n"
            if not text.strip():
                raise gr.Error("The transcription is empty.")
            if text == doc["history"][doc["position"]]:   # nothing changed: no new check
                return [doc, gr.update(), *spelling_review(spelling), *render(doc, zoom, limit), *mode("text")]
            # the spelling is checked again on the edited text (only the new lines go to the model)
            doc, spelling = edit_text(doc, text, options_of(ner, french_ner, translate, spellcheck), progress,
                                      spelling)
            return [doc, gr.update(), *spelling_review(spelling), *render(doc, zoom, limit), *mode("text")]

        edit_button.click(on_edit, [doc_state, text_edit, view_state, spelling_state, *option_inputs, threshold,
                                    zoom_state],
                          [doc_state, text_edit, *review_outputs, *panels, *modes])

        def on_spelling_button(doc, view, spelling):
            if not doc:
                raise gr.Error("Import a scan first.")
            if not spelling["text"]:
                gr.Info("The spelling check is off: turn it on in Options, then run again.")
            return mode("text") if view == "spelling" else mode("spelling")

        spelling_button.click(on_spelling_button, [doc_state, view_state, spelling_state], modes)

        def on_history(doc, step, ner, french_ner, translate, spellcheck, limit, zoom):
            if not doc:
                raise gr.Error("Import a scan first.")
            doc = move_in_history(doc, step, options_of(ner, french_ner, translate, spellcheck))
            return [doc, *render(doc, zoom, limit), *mode("text")]

        for button, step in ((undo, -1), (redo, 1)):
            button.click(lambda *args, s=step: on_history(args[0], s, *args[1:]),
                         [doc_state, *option_inputs, threshold, zoom_state], [doc_state, *panels, *modes])

        spelling_view.select(toggle_change, [spelling_state], review_outputs)
        back_to_rules.click(lambda state: set_all(state, True), [spelling_state], review_outputs)
        refuse_all.click(lambda state: set_all(state, False), [spelling_state], review_outputs)

        def on_apply(doc, spelling, ner, french_ner, translate, spellcheck, limit, zoom):
            if not doc or not spelling["text"]:
                raise gr.Error("No spelling check to apply.")
            text = apply_changes(spelling["text"], resolved(spelling), set(spelling["accepted"]))
            doc = dict(doc, spelling_notes=[n for n in spelling_notes(spelling).strip().split("\n") if n])
            doc = set_text(doc, text, options_of(ner, french_ner, translate, spellcheck), "c_spelling")
            return [doc, *render(doc, zoom, limit), *mode("text")]

        apply_spelling.click(on_apply, [doc_state, spelling_state, *option_inputs, threshold, zoom_state],
                             [doc_state, *panels, *modes])

        def on_json_mode(raw):
            raw = not raw
            return [raw, gr.update(visible=not raw), gr.update(visible=raw),
                    gr.update(value=I18N("table") if raw else I18N("raw"))]

        json_mode.click(on_json_mode, [json_raw], [json_raw, table_view, json_view, json_mode])

        # the panels but the table and the raw JSON, which the JSON editor also shows / hides
        other_panels = [p for p in panels if p is not table_view and p is not json_view]

        def json_tab(doc, zoom, limit, raw, editing, editor=None):
            """[panels but table and raw JSON] + [editor, table, raw JSON, edit button, raw/table button]: the
            editor, or the table / the raw JSON; the edit button shows ✓ while editing."""
            views = render(doc, zoom, limit)
            table, record = views[3], views[4]
            return [*(v for i, v in enumerate(views) if i not in (3, 4)),
                    gr.update(visible=editing) if editor is None else gr.update(value=editor, visible=editing),
                    gr.update(value=table, visible=not editing and not raw),
                    gr.update(value=record, visible=not editing and raw),
                    gr.update(value=I18N("done") if editing else I18N("edit"),
                              variant="primary" if editing else "secondary"),
                    gr.update(interactive=not editing)]

        json_outputs = [*other_panels, json_editor, table_view, json_view, json_edit_button, json_mode]

        def on_json_edit(doc, before, written, raw, ner, french_ner, translate, spellcheck, limit, zoom):
            if not doc or not doc.get("record"):
                raise gr.Error("Import a scan first.")
            if before is None:   # start editing: the record as JSON
                record = doc["record"]
                return [{"folder": doc["folder"], "record": record}, doc,
                        *json_tab(doc, zoom, limit, raw, True, json.dumps(record, ensure_ascii=False, indent=2))]
            if before["folder"] != doc["folder"]:   # another scan was opened meanwhile: the edit was for the other
                gr.Warning("Another scan was opened: the edit of the previous record is dropped.")
                return [None, doc, *json_tab(doc, zoom, limit, raw, False)]
            try:
                record = record_from_json(written or "")
            except ValueError as e:
                raise gr.Error(str(e))    # the editor stays open with what was written
            doc, _ = edit_record(doc, before["record"], record, options_of(ner, french_ner, translate, spellcheck))
            return [None, doc, *json_tab(doc, zoom, limit, raw, False)]

        json_edit_button.click(on_json_edit, [doc_state, json_editing, json_editor, json_raw, *option_inputs,
                                              threshold, zoom_state],
                               [json_editing, doc_state, *json_outputs])

        def on_translate(doc, editing, ner, french_ner, translate, spellcheck, limit, zoom,
                         progress=gr.Progress()):
            if not doc or not doc.get("record"):
                raise gr.Error("Import a scan first.")
            if editing is not None:
                raise gr.Error("Finish the edit of the record first (✓).")
            reason = translator.unavailable_reason()
            if reason:
                raise gr.Error(f"The translation model cannot be used: {reason}")
            progress(0.3, desc=f"Translation (local model {translator.MODEL})")
            doc = translate_document(doc, options_of(ner, french_ner, translate, spellcheck))
            return [doc, *render(doc, zoom, limit)]

        translate_button.click(on_translate, [doc_state, json_editing, *option_inputs, threshold, zoom_state],
                               [doc_state, *panels])
        options_button.click(lambda is_open: [not is_open, gr.update(visible=not is_open)],
                             [options_open], [options_open, options_panel])
        help_button.click(lambda is_open: [not is_open, gr.update(visible=not is_open)],
                          [help_open], [help_open, help_panel])
    return app


def logins(text):
    """"ana:1234,sara:abcd" -> [("ana", "1234"), ("sara", "abcd")] (empty text: no login asked)."""
    pairs = []
    for item in (text or "").split(","):
        user, _, password = item.strip().partition(":")
        if user and password:
            pairs.append((user, password))
        elif item.strip():
            raise SystemExit(f"--auth / MADOC_AUTH: {item.strip()!r} is not USER:PASSWORD")
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Gradio interface of the birth certificate pipeline.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="public gradio.live link")
    parser.add_argument("--device", default="cpu", help="kraken device: cpu, cuda:0 ...")
    parser.add_argument("--segmentation-model", type=Path, default=MODEL_SEGMENTATION)
    parser.add_argument("--recognition-model", type=Path, default=MODEL_RECOGNITION)
    parser.add_argument("--no-pwa", action="store_true", help="do not offer MaDoc as an installable application")
    parser.add_argument("--auth", metavar="USER:PASSWORD",
                        help="ask for a login (several: a:1,b:2); also MADOC_AUTH in .env, which keeps the password "
                             "out of the shell history")
    args = parser.parse_args()
    auth = logins(args.auth or os.environ.get("MADOC_AUTH", ""))
    if args.share and not auth:
        print("Warning: --share without --auth: anyone with the link can use MaDoc (and the GPU). Only demo "
              "documents: what is uploaded goes through gradio's relay server.", file=sys.stderr)
    # checked once at start: a GPU that torch cannot use falls back to the CPU with a warning
    app = build_app(resolve_device(args.device), args.segmentation_model, args.recognition_model)
    # i18n: English, French and Arabic (Settings at the bottom of the page); pwa: MaDoc can be installed as an
    # application, which shows a notification when a processing ends in the background
    # the square logo: icon of the browser tab and of the installed application (instead of gradio's)
    icon = find_logos().get("icon")
    launch = {"server_name": args.host, "server_port": args.port, "share": args.share, "i18n": I18N,
              "pwa": not args.no_pwa, "allowed_paths": [str(SAMPLES_DIR)],
              "favicon_path": str(icon) if icon else None}
    if auth:
        launch.update(auth=auth, auth_message=login_message(find_logos()))
    allowed = inspect.signature(app.launch).parameters
    app.queue().launch(**{k: v for k, v in launch.items() if k in allowed})


if __name__ == "__main__":
    main()