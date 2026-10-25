"""app.py - Gradio interface: import a scan, review the transcription and the JSON record.

    python src/app.py                      # http://127.0.0.1:7860
    python src/app.py --share --device cuda:0

Layout
  top bar     Files (import a scan: it is processed at once), Save (zip of everything), Options, Help
  left        Image / Segmentation tabs: the scan, and the scan with the polygon of every text line coloured by
              the kraken confidence; + / - zoom, a download button in each tab
  right       Text / JSON tabs. Text: the transcription, undo / redo, [S] the spelling review, [E] editing
              (leaving the editor parses the text again). JSON: the record as a table (or raw), download
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
gradio's cache, the downloads) is deleted after KEEP_SECONDS. The online stages (French NER on Hugging Face,
Groq translation) are off by default.
"""

import argparse
import base64
import html
import inspect
import io
import shutil
import tempfile
import time
import zipfile
from functools import lru_cache
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw, ImageOps

from common import normalize_text, resolve_device, write_json
from parser import find_zones, flatten
from pipeline import run_pipeline
from recognition import MODEL_RECOGNITION, recognize_lines
from segmentation import MODEL_SEGMENTATION, segment_image
from spelling_checker import MODEL as SPELL_MODEL, apply_changes, describe, line_units, propose
from validator import format_report
from ui_text import CSS as I18N_CSS, PERMISSION_JS, gradio_i18n, notification_js, tr

DISPLAY_MAX_SIDE = 1800   # the scan is shown at most this wide (polygons keep the original coordinates)
KEEP_SECONDS = 3600       # uploads and downloads of a run are deleted after one hour
RUNS_DIR = Path(tempfile.gettempdir()) / "madoc_runs"
ZOOM_STEPS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)

CSS = """
.md-bar {align-items:center !important; gap:8px !important; flex-wrap:nowrap !important;}
.md-bar button {min-width:7.5em !important; flex:0 0 auto !important; white-space:nowrap;}
.md-bar .md-help button, .md-bar button.md-help {min-width:2.6em !important;}
.md-title {text-align:right; font-weight:700; font-size:1.25em; padding-right:6px;}
.md-tools {gap:6px !important; justify-content:flex-end; flex-wrap:nowrap !important;}
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


def prepare_image(image_path, folder):
    """The scan as kraken and the viewer must see it: turned upright (phone photos store the rotation in the
    EXIF data, which kraken ignores) and in a mode kraken reads. Unchanged images are used as they are."""
    with Image.open(image_path) as image:
        rotated = image.getexif().get(0x0112, 1) not in (None, 1)   # 0x0112 = EXIF orientation
        if not rotated and image.mode in ("RGB", "L", "1"):
            return str(image_path)
        prepared = Path(folder) / "scan.png"
        ImageOps.exif_transpose(image).convert("RGB").save(prepared)
    return str(prepared)


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


def draw_segmentation(image_path, lines, threshold, target):
    """The scan with the line polygons drawn on it (the segmentation download)."""
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for line in lines:
            if len(line.get("polygon") or []) >= 3:
                color = confidence_color(line.get("confidence"), threshold)
                rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
                draw.polygon([tuple(p) for p in line["polygon"]], fill=rgb + (50,), outline=rgb + (255,))
        Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB").save(target)
    return str(target)


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
    return "⁨" + text + "⁩"


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
                            f"{change['reason']}: no correction found, check the scan")
                continue
            mark = "✓" if i in accepted else "✗"
            if change.get("options") and len(change["options"]) > 1:
                rule = ("from jobs.json, several jobs possible: " + " / ".join(isolate(" ".join(o)) for o in change["options"])
                        + " (click again for the next one)")
            elif change.get("source") == "names":
                rule = (f"the child's Arabic and Latin names did not match; proposed {change['context']}"
                        " (names are never applied automatically: click to apply)")
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
        text, use_ner=options["ner"], use_translation=options["groq"], use_french_ner=options["french_ner"],
        stages=stages,
    )
    doc.update(record=record, report=report, parser_notes=[f"[parser] {n}" for n in notes],
               field_lines=field_lines(text, stages.get("parsed")))
    doc["notes"] = doc.get("spelling_notes", []) + doc["parser_notes"]
    write_files(doc, text)
    return doc


def write_files(doc, text):
    """Every download of the run, and the zip of all of them (Save)."""
    folder = Path(doc["folder"])
    write_json(doc["record"], folder / "record.json")
    write_json(doc["report"], folder / "report.json")
    (folder / "report.txt").write_text(format_report(doc["report"]) + "\n\n" + "\n".join(doc["notes"]) + "\n",
                                       encoding="utf-8")
    (folder / "transcription.txt").write_text(text, encoding="utf-8")
    (folder / "transcription_htr.txt").write_text(doc["htr"], encoding="utf-8")
    (folder / "segmentation.xml").write_text(doc["xml"], encoding="utf-8")
    if not (folder / "segmentation.png").is_file():
        draw_segmentation(doc["image"], doc["lines"], doc.get("threshold", 0.8), folder / "segmentation.png")
    image = Path(doc["image"])
    names = {"record.json", "report.json", "report.txt", "transcription.txt", "transcription_htr.txt",
             "segmentation.xml", "segmentation.png"}
    stem = Path(doc.get("name") or "scan").stem
    with zipfile.ZipFile(folder / f"{stem}_madoc.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(image, "scan" + image.suffix.lower())
        for name in sorted(names):
            archive.write(folder / name, name)
    doc["files"] = {"image": str(image), "segmentation": str(folder / "segmentation.png"),
                    "text": str(folder / "transcription.txt"), "json": str(folder / "record.json"),
                    "zip": str(folder / f"{stem}_madoc.zip")}


def options_of(ner, french_ner, groq, spellcheck):
    return {"ner": ner, "french_ner": french_ner, "groq": groq, "spellcheck": spellcheck}


def check_spelling(doc, options, progress=None):
    """The spelling proposals on the HTR text (local model); the rules' choice is applied."""
    if not options["spellcheck"]:
        doc["spelling_notes"] = []
        return dict(NO_SPELLING), doc["htr"]
    if progress:
        progress(0.6, desc=f"Spelling check (local model {SPELL_MODEL})")
    changes, skipped = propose(doc["htr"])
    spelling = {"text": doc["htr"], "changes": changes, "selected": {},
                "accepted": [i for i, c in enumerate(changes) if c["accepted"]], "segments": []}
    doc["spelling_notes"] = [f"[spelling] {n}" for n in skipped] + spelling_notes(spelling).strip().split("\n")
    doc["spelling_notes"] = [n for n in doc["spelling_notes"] if n]
    return spelling, apply_changes(doc["htr"], changes)


def start_document(image_path, name, options, threshold, device, segmentation_model, recognition_model,
                   progress=None):
    """Scan -> segmentation -> recognition -> spelling -> parsing: the document state of the interface."""
    folder = new_run_folder()
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
    doc = {"folder": str(folder), "image": image_path, "name": name, "xml": xml, "lines": lines, "htr": htr,
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
        gr.HTML(f"<style>{CSS}{I18N_CSS}</style>")
        doc_state, spelling_state = gr.State(None), gr.State(dict(NO_SPELLING))
        zoom_state, options_open, help_open = gr.State(1.0), gr.State(False), gr.State(False)
        view_state = gr.State("text")   # what the Text tab shows: "text", "edit" or "spelling"
        json_raw = gr.State(False)      # the JSON tab shows the table (False) or the raw JSON (True)

        # top bar ------------------------------------------------------------------------------------------
        with gr.Row(elem_classes="md-bar", equal_height=True):
            files_button = make(gr.UploadButton, label=I18N("files"), file_types=["image"], file_count="single",
                                size="sm", scale=0, min_width=110)
            save_button = make(gr.DownloadButton, label=I18N("save"), size="sm", scale=0, min_width=110,
                               interactive=False)
            options_button = make(gr.Button, value=I18N("options"), size="sm", scale=0, min_width=110)
            gr.HTML('<div class="md-title">MaDoc</div>')
            help_button = make(gr.Button, value="?", size="sm", scale=0, min_width=48, elem_classes="md-help")
        with gr.Column(visible=False) as options_panel:
            with gr.Row():
                use_spellcheck = gr.Checkbox(True, label=I18N("opt_spellcheck"))
                use_ner = gr.Checkbox(True, label=I18N("opt_ner"))
                use_french_ner = gr.Checkbox(True, label=I18N("opt_french_ner"))
                use_groq = gr.Checkbox(True, label=I18N("opt_groq"))
            with gr.Row():
                threshold = gr.Slider(0.3, 0.99, value=0.8, step=0.01, label=I18N("opt_threshold"))
                rerun = make(gr.Button, value=I18N("rerun"), size="sm")
        with gr.Column(visible=False) as help_panel:
            gr.Markdown(I18N("help"))

        # left: image / segmentation, right: text / json ---------------------------------------------------
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=360):
                with gr.Tabs():
                    with gr.Tab(I18N("tab_image")):
                        with gr.Row(elem_classes="md-tools"):
                            zoom_out_1 = make(gr.Button, value="−", size="sm")
                            zoom_in_1 = make(gr.Button, value="+", size="sm")
                            download_image = make(gr.DownloadButton, label=I18N("dl_image"), size="sm", interactive=False)
                        image_view = gr.HTML(image_html(None, 1.0))
                    with gr.Tab(I18N("tab_segmentation")):
                        with gr.Row(elem_classes="md-tools"):
                            zoom_out_2 = make(gr.Button, value="−", size="sm")
                            zoom_in_2 = make(gr.Button, value="+", size="sm")
                            download_segmentation = make(gr.DownloadButton, label=I18N("dl_segmentation"), size="sm",
                                                         interactive=False)
                        segmentation_view = gr.HTML(segmentation_html(None, 1.0, 0.8))
            with gr.Column(scale=1, min_width=360):
                with gr.Tabs():
                    with gr.Tab(I18N("tab_text")):
                        with gr.Row(elem_classes="md-tools"):
                            undo = make(gr.Button, value="↶", size="sm")
                            redo = make(gr.Button, value="↷", size="sm")
                            spelling_button = make(gr.Button, value="S", size="sm")
                            edit_button = make(gr.Button, value="E", size="sm")
                            download_text = make(gr.DownloadButton, label=I18N("dl_txt"), size="sm", interactive=False)
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
                            download_json = make(gr.DownloadButton, label=I18N("dl_json"), size="sm", interactive=False)
                        table_view = gr.HTML(record_html(None))
                        json_view = gr.JSON(visible=False)
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
                    gr.update(value=I18N("done") if view == "edit" else "E", variant="primary" if view == "edit" else "secondary"),
                    gr.update(variant="primary" if view == "spelling" else "secondary")]

        option_inputs = [use_ner, use_french_ner, use_groq, use_spellcheck]
        review_outputs = [spelling_view, spelling_details, spelling_state]

        # events -------------------------------------------------------------------------------------------
        def on_upload(file, ner, french_ner, groq, spellcheck, limit, zoom, progress=gr.Progress()):
            if not file:
                raise gr.Error("Choose a scan.")
            path = file if isinstance(file, str) else getattr(file, "name", str(file))
            options = options_of(ner, french_ner, groq, spellcheck)
            doc, spelling = start_document(path, Path(path).name, options, limit, device, segmentation_model,
                                           recognition_model, progress)
            review = spelling_review(spelling)
            return [doc, *review, *render(doc, zoom, limit), *mode("text")]

        files_button.click(None, None, None, js=PERMISSION_JS)
        files_button.upload(on_upload, [files_button, *option_inputs, threshold, zoom_state],
                            [doc_state, *review_outputs, *panels, *modes]).then(None, None, None, js=notification_js())

        def on_rerun(doc, ner, french_ner, groq, spellcheck, limit, zoom, progress=gr.Progress()):
            if not doc:
                raise gr.Error("Import a scan first.")
            options = options_of(ner, french_ner, groq, spellcheck)
            doc = dict(doc)
            spelling, text = check_spelling(doc, options, progress)
            doc.update(history=[text], position=0)
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

        def on_edit(doc, editing_text, view, ner, french_ner, groq, spellcheck, limit, zoom):
            if not doc:
                raise gr.Error("Import a scan first.")
            if view != "edit":   # start editing
                return [doc, gr.update(value=doc["history"][doc["position"]]), *render(doc, zoom, limit),
                        *mode("edit")]
            text = editing_text if editing_text.endswith("\n") else editing_text + "\n"
            if not text.strip():
                raise gr.Error("The transcription is empty.")
            doc = set_text(doc, text, options_of(ner, french_ner, groq, spellcheck), "c_edit")
            return [doc, gr.update(), *render(doc, zoom, limit), *mode("text")]

        edit_button.click(on_edit, [doc_state, text_edit, view_state, *option_inputs, threshold, zoom_state],
                          [doc_state, text_edit, *panels, *modes])

        def on_spelling_button(doc, view, spelling):
            if not doc:
                raise gr.Error("Import a scan first.")
            if not spelling["text"]:
                gr.Info("The spelling check is off: turn it on in Options, then run again.")
            return mode("text") if view == "spelling" else mode("spelling")

        spelling_button.click(on_spelling_button, [doc_state, view_state, spelling_state], modes)

        def on_history(doc, step, ner, french_ner, groq, spellcheck, limit, zoom):
            if not doc:
                raise gr.Error("Import a scan first.")
            doc = move_in_history(doc, step, options_of(ner, french_ner, groq, spellcheck))
            return [doc, *render(doc, zoom, limit), *mode("text")]

        for button, step in ((undo, -1), (redo, 1)):
            button.click(lambda *args, s=step: on_history(args[0], s, *args[1:]),
                         [doc_state, *option_inputs, threshold, zoom_state], [doc_state, *panels, *modes])

        spelling_view.select(toggle_change, [spelling_state], review_outputs)
        back_to_rules.click(lambda state: set_all(state, True), [spelling_state], review_outputs)
        refuse_all.click(lambda state: set_all(state, False), [spelling_state], review_outputs)

        def on_apply(doc, spelling, ner, french_ner, groq, spellcheck, limit, zoom):
            if not doc or not spelling["text"]:
                raise gr.Error("No spelling check to apply.")
            text = apply_changes(spelling["text"], resolved(spelling), set(spelling["accepted"]))
            doc = dict(doc, spelling_notes=[n for n in spelling_notes(spelling).strip().split("\n") if n])
            doc = set_text(doc, text, options_of(ner, french_ner, groq, spellcheck), "c_spelling")
            return [doc, *render(doc, zoom, limit), *mode("text")]

        apply_spelling.click(on_apply, [doc_state, spelling_state, *option_inputs, threshold, zoom_state],
                             [doc_state, *panels, *modes])

        def on_json_mode(raw):
            raw = not raw
            return [raw, gr.update(visible=not raw), gr.update(visible=raw),
                    gr.update(value=I18N("table") if raw else I18N("raw"))]

        json_mode.click(on_json_mode, [json_raw], [json_raw, table_view, json_view, json_mode])
        options_button.click(lambda is_open: [not is_open, gr.update(visible=not is_open)],
                             [options_open], [options_open, options_panel])
        help_button.click(lambda is_open: [not is_open, gr.update(visible=not is_open)],
                          [help_open], [help_open, help_panel])
    return app


def main():
    parser = argparse.ArgumentParser(description="Gradio interface of the birth certificate pipeline.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="public gradio.live link")
    parser.add_argument("--device", default="cpu", help="kraken device: cpu, cuda:0 ...")
    parser.add_argument("--segmentation-model", type=Path, default=MODEL_SEGMENTATION)
    parser.add_argument("--recognition-model", type=Path, default=MODEL_RECOGNITION)
    parser.add_argument("--no-pwa", action="store_true", help="do not offer MaDoc as an installable application")
    args = parser.parse_args()
    # checked once at start: a GPU that torch cannot use falls back to the CPU with a warning
    app = build_app(resolve_device(args.device), args.segmentation_model, args.recognition_model)
    # i18n: English, French and Arabic (Settings at the bottom of the page); pwa: MaDoc can be installed as an
    # application, which shows a notification when a processing ends in the background
    launch = {"server_name": args.host, "server_port": args.port, "share": args.share, "i18n": I18N,
              "pwa": not args.no_pwa}
    allowed = inspect.signature(app.launch).parameters
    app.queue().launch(**{k: v for k, v in launch.items() if k in allowed})


if __name__ == "__main__":
    main()