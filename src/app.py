"""app.py - Gradio interface: upload a scan, get the transcription and the JSON record.

    python src/app.py                      # http://127.0.0.1:7860
    python src/app.py --share --device cuda:0

Like eScriptorium, the scan is shown with the polygon of every text line (coloured by the kraken confidence
score) next to the transcription, line by line. Hovering a line in the list highlights its polygon on the
scan and the other way round (CSS only). The lines under the confidence threshold are flagged.

Tabs: Transcription (editable txt, "parse again" re-runs the parser on the corrected text), JSON record,
validation report + spelling and parser notes, downloads (json, txt, PageXML of the segmentation).
The optional spelling correction runs a local model (Ollama, see spelling_checker.py). The "Spelling" tab shows
the text with every proposed change highlighted (green = applied, red = not applied, the badge is the proposed
spelling): click a change to accept / refuse it, then "Apply". A misread job close to several jobs of jobs.json
cycles through them at every click (refused -> first job -> second job ... -> refused). The changes the rules of spelling_checker.py
accept are applied at first; the transcription tab shows the corrected text, "Image and lines" what the HTR read.
The models are loaded the first time they are used and then kept by the running app.

The kraken device and models are chosen on the command line only (not in the page: a shared app must not let
visitors point kraken at other files). The scans are personal documents: every file of a run (the upload in
gradio's cache, the downloads) is deleted after KEEP_SECONDS.
"""

import argparse
import base64
import html
import inspect
import io
import shutil
import tempfile
import time
from functools import lru_cache
from pathlib import Path

import gradio as gr
from PIL import Image, ImageOps

from common import resolve_device, write_json
from pipeline import run_pipeline
from recognition import MODEL_RECOGNITION, recognize_lines
from segmentation import MODEL_SEGMENTATION, segment_image
from spelling_checker import MODEL as SPELL_MODEL, apply_changes, describe, line_units, propose
from validator import format_report

DISPLAY_MAX_SIDE = 1800   # the scan is shown at most this wide (polygons keep the original coordinates)
KEEP_SECONDS = 3600       # uploads and downloads of a run are deleted after one hour
RUNS_DIR = Path(tempfile.gettempdir()) / "madoc_runs"

CSS = """
.bv-wrap {display:flex; gap:12px; align-items:flex-start; width:100%; font-family:inherit;}
.bv-page {flex:1 1 55%; min-width:0; position:sticky; top:0;}
.bv-page svg {width:100%; height:auto; border:1px solid #8884; border-radius:6px; background:#fff;}
.bv-list {flex:1 1 45%; min-width:0; max-height:80vh; overflow:auto; border:1px solid #8884; border-radius:6px;}
.bv-row {display:grid; grid-template-columns:2.2em 1fr 4.6em; gap:8px; padding:6px 8px; align-items:center;
         border-bottom:1px solid #8883; cursor:default;}
.bv-row:hover, .bv-row.hl {background:#3b82f622;}
.bv-row.low {border-left:4px solid #dc2626;}
.bv-num {color:#888; font-size:.8em; text-align:right;}
.bv-text {direction:rtl; text-align:right; font-size:1.15em; line-height:1.5;}
.bv-conf {font-size:.85em; text-align:center; border-radius:4px; padding:2px 0; color:#111; font-weight:600;}
.bv-poly {fill-opacity:.22; stroke-width:3; vector-effect:non-scaling-stroke; transition:fill-opacity .1s;}
.bv-poly.hl {fill-opacity:.55; stroke-width:5;}
.bv-base {fill:none; stroke:#111; stroke-opacity:.55; stroke-width:1.5; vector-effect:non-scaling-stroke;}
.bv-sum {margin:0 0 8px; font-size:.9em; color:#888;}
"""


def confidence_color(confidence, threshold):
    """green (sure) > amber > red (under the threshold); grey when kraken gave no score."""
    if confidence is None:
        return "#9ca3af"
    if confidence < threshold:
        return "#dc2626"
    if confidence < min(0.98, threshold + 0.1):
        return "#f59e0b"
    return "#16a34a"


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


def encode_image(path):
    """(data URI of the scan reduced for display, original width, original height)"""
    return _encode_image(str(path), Path(path).stat().st_mtime)


@lru_cache(maxsize=8)
def _encode_image(path, mtime):
    """Cached by (path, modification time): moving the threshold slider does not encode the scan again."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        shown = image.copy()
        shown.thumbnail((DISPLAY_MAX_SIDE, DISPLAY_MAX_SIDE))
        buffer = io.BytesIO()
        shown.save(buffer, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"), width, height


def render_view(image_path, lines, threshold):
    """HTML: the scan with the polygons on the left, the lines with their confidence on the right."""
    if not image_path:
        return "<p>Upload a scan.</p>"
    uri, width, height = encode_image(image_path)
    scores = [line["confidence"] for line in lines if line.get("confidence") is not None]
    low = sum(1 for line in lines if line.get("confidence") is not None and line["confidence"] < threshold)
    summary = f"{len(lines)} lines"
    if scores:
        summary += f" · mean confidence {sum(scores) / len(scores):.1%} · {low} under {threshold:.0%}"
    else:
        summary += " · no confidence score in the kraken output"
    polygons, rows = [], []
    for number, line in enumerate(lines, 1):
        color = confidence_color(line.get("confidence"), threshold)
        score = line.get("confidence")
        label = "n/a" if score is None else f"{score:.0%}"
        if line.get("polygon"):
            points = " ".join(f"{x},{y}" for x, y in line["polygon"])
            polygons.append(
                f'<polygon id="bv-poly-{number}" class="bv-poly" points="{points}" fill="{color}" '
                f'stroke="{color}"><title>{number}: {html.escape(line["text"])} ({label})</title></polygon>'
            )
        if line.get("baseline"):
            polygons.append(f'<polyline class="bv-base" points="{" ".join(f"{x},{y}" for x, y in line["baseline"])}"/>')
        flagged = " low" if score is not None and score < threshold else ""
        rows.append(
            f'<div id="bv-row-{number}" class="bv-row{flagged}">'
            f'<span class="bv-num">{number}</span><span class="bv-text">{html.escape(line["text"])}</span>'
            f'<span class="bv-conf" style="background:{color}33; border:1px solid {color}">{label}</span></div>'
        )
    # hovering a polygon lights its row and the other way round: pure CSS (:has), no script needed
    link_css = "".join(
        f".bv-wrap:has(#bv-poly-{n}:hover) #bv-row-{n}, .bv-wrap:has(#bv-row-{n}:hover) #bv-poly-{n}"
        f"{{background:#3b82f633; fill-opacity:.55; stroke-width:5;}}"
        for n in range(1, len(lines) + 1)
    )
    return (
        f"<style>{CSS}{link_css}</style>"
        f'<p class="bv-sum">{summary}</p>'
        f'<div class="bv-wrap"><div class="bv-page"><svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
        f'<image href="{uri}" width="{width}" height="{height}"/>{"".join(polygons)}</svg></div>'
        f'<div class="bv-list">{"".join(rows)}</div></div>'
    )


def transcription(lines):
    return "\n".join(line["text"] for line in lines) + "\n"


def run_record(text, use_ner, use_translation):
    """text -> (record, report text, notes text) with the whole parsing chain."""
    record, report, notes = run_pipeline(text, use_ner=use_ner, use_translation=use_translation)
    notes_text = "\n".join(f"[parser] {note}" for note in notes) or "no parser note"
    return record, format_report(report), notes_text


def save_files(folder, record, text, xml):
    folder = Path(folder)
    paths = []
    for name, content in (("record.json", None), ("transcription.txt", text), ("segmentation.xml", xml)):
        path = folder / name
        if name == "record.json":
            write_json(record, path)
        elif content:
            path.write_text(content, encoding="utf-8")
        else:
            continue
        paths.append(str(path))
    return paths


ACCEPTED_COLOR, REFUSED_COLOR = "#16a34a", "#dc2626"
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
        return gr.HighlightedText(value=[]), "Tick \"Spelling correction\" in the options, then Transcribe.", state
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
            proposal = " ".join(change["new"]) or "(removed)"
            label = ("✓ " if index in accepted else "✗ ") + proposal
            color_map[label] = ACCEPTED_COLOR if index in accepted else REFUSED_COLOR
            pieces.append((line[begin:finish], label))
            segments.append(index)
            written = finish
        pieces.append((line[written:] + "\n", None))
        segments.append(None)
    shown = [c for c in changes if c["old"]]
    if not shown:
        details = "The model proposed no change."
    else:
        rows = []
        for i, change in enumerate(changes):
            if not change["old"]:
                continue
            mark = "✓" if i in accepted else "✗"
            if change.get("options") and len(change["options"]) > 1:
                rule = ("from jobs.json, several jobs possible: " + " / ".join(isolate(" ".join(o)) for o in change["options"])
                        + " (click again for the next one)")
            elif change.get("source") == "jobs.json":
                rule = "from jobs.json" + (", " + change["context"] if change.get("context") else "") + (
                    "" if change["accepted"] else f"; refused by the rules: {change['reason']}")
            elif change["accepted"]:
                rule = "accepted by the rules"
            else:
                rule = f"refused by the rules: {change['reason']}"
            rows.append(f"- {mark} line {change['line'] + 1}: **{isolate(' '.join(change['old']))}** → "
                        f"**{isolate(' '.join(change['new']) or '(removed)')}** ({rule})")
        details = (f"{len(accepted)} of {len(shown)} changes applied. Click a highlighted word to accept / refuse "
                   "it, then press Apply.\n\n" + "\n".join(rows))
    state = {**state, "segments": segments}
    return gr.HighlightedText(value=pieces, color_map=color_map), details, state


def spelling_notes(state):
    """The changes with the decision taken (by the rules, or by the person in the Spelling tab)."""
    accepted, notes = set(state["accepted"]), []
    for i, change in enumerate(resolved(state)):
        chosen = i in accepted
        if chosen == change["accepted"]:
            notes.append(describe(change))
        elif chosen:   # accepted by the person although the rules refused it
            notes.append(describe({**change, "accepted": True})
                         + f" (accepted in the Spelling tab; the rules refused it: {change['reason']})")
        else:
            notes.append(describe({**change, "accepted": False, "reason": "refused in the Spelling tab"}))
    return "\n".join(f"[spelling] {note}" for note in notes) + "\n\n" if notes else ""


def toggle_change(state, evt: gr.SelectData):
    """A click on a highlighted change accepts / refuses it. A change with several options (a misread job close
    to two jobs) goes: refused -> first option -> second option ... -> refused."""
    segments = state.get("segments") or []
    index = segments[evt.index] if isinstance(evt.index, int) and evt.index < len(segments) else None
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
    accepted = [i for i, c in enumerate(state["changes"]) if c["accepted"]] if accept else []
    return spelling_review({**state, "accepted": accepted, "selected": {}})


def apply_spelling(state, use_ner, use_translation):
    """The HTR text with the accepted changes -> transcription, record, report and downloads again."""
    if not state["text"]:
        raise gr.Error("No spelling check to apply: tick \"Spelling correction\" in the options and transcribe.")
    text = apply_changes(state["text"], resolved(state), set(state["accepted"]))
    record, report_text, notes_text = run_record(text, use_ner, use_translation)
    report = report_text + "\n\n" + spelling_notes(state) + notes_text
    return text, record, report, save_files(new_run_folder(), record, text, "")


def process(image_path, use_ner, use_translation, threshold, device, segmentation_model, recognition_model,
            progress=gr.Progress(), use_spellcheck=False):
    """Scan -> segmentation -> recognition -> pipeline. Returns what the interface shows."""
    if not image_path:
        raise gr.Error("Upload a scan first.")
    folder = new_run_folder()
    image_path = prepare_image(image_path, folder)
    progress(0.05, desc="Segmentation (kraken blla)")
    xml = segment_image(image_path, segmentation_model, device)
    progress(0.35, desc="Recognition (kraken HTR)")
    lines = recognize_lines(xml, image_path, recognition_model, device)
    if not lines:
        raise gr.Error("No text line was recognized on this image.")
    text = transcription(lines)
    spelling = dict(NO_SPELLING)
    spelling_text = ""
    if use_spellcheck:
        # local model only (Ollama): the changes the rules accept are applied, the Spelling tab shows them all
        progress(0.6, desc=f"Spelling check (local model {SPELL_MODEL})")
        changes, skipped = propose(text)
        spelling = {"text": text, "changes": changes, "selected": {},
                    "accepted": [i for i, c in enumerate(changes) if c["accepted"]], "segments": []}
        text = apply_changes(text, changes)
        # without the model (Ollama not running ...), the jobs are still checked against jobs.json
        spelling_text = "".join(f"[spelling] {note}\n" for note in skipped) + spelling_notes(spelling)
    progress(0.8, desc="Parsing, mapping, translation, validation")
    record, report_text, notes_text = run_record(text, use_ner, use_translation)
    notes_text = spelling_text + notes_text
    files = save_files(folder, record, text, xml)
    review, details, spelling = spelling_review(spelling)
    return (render_view(image_path, lines, threshold), text, record, report_text + "\n\n" + notes_text,
            files, lines, image_path, review, details, spelling)


def reparse(text, use_ner, use_translation):
    """The corrected transcription -> new record and report (no new recognition)."""
    if not text.strip():
        raise gr.Error("The transcription is empty.")
    record, report_text, notes_text = run_record(text, use_ner, use_translation)
    return record, report_text + "\n\n" + notes_text, save_files(new_run_folder(), record, text, "")


def rerender(image_path, lines, threshold):
    if image_path and not Path(image_path).is_file():
        return "<p>The scan of this run was deleted (older than one hour): transcribe it again.</p>"
    return render_view(image_path, lines or [], threshold)


def make(component, **kwargs):
    """Create a gradio component, dropping the options this gradio version does not have (they changed a lot)."""
    allowed = inspect.signature(component.__init__).parameters
    return component(**{k: v for k, v in kwargs.items() if k in allowed})


def build_app(device="cpu", segmentation_model=MODEL_SEGMENTATION, recognition_model=MODEL_RECOGNITION):
    def transcribe(image_path, use_ner, use_translation, use_spellcheck, threshold, progress=gr.Progress()):
        return process(image_path, use_ner, use_translation, threshold, device, segmentation_model,
                       recognition_model, progress, use_spellcheck)

    # delete_cache: gradio's copy of every upload / download is removed after KEEP_SECONDS
    with make(gr.Blocks, title="MaDoc", delete_cache=(KEEP_SECONDS, KEEP_SECONDS)) as app:
        gr.Markdown("## MaDoc: Birth certificates transcription and structuring")
        lines_state, image_state, spelling_state = gr.State([]), gr.State(""), gr.State(dict(NO_SPELLING))
        with gr.Row():
            with gr.Column(scale=1, min_width=260):
                image = make(gr.Image, type="filepath", label="Scan", sources=["upload", "clipboard"], height=300)
                run = gr.Button("Transcribe", variant="primary")
                with gr.Accordion("Options", open=False):
                    use_ner = gr.Checkbox(True, label="NER models as a second extractor")
                    use_translation = gr.Checkbox(True, label="Groq translation (needs GROQ_API_KEY)")
                    use_spellcheck = gr.Checkbox(False, label=f"Spelling correction, local model ({SPELL_MODEL}, needs Ollama)")
                    threshold = gr.Slider(0.3, 0.99, value=0.8, step=0.01, label="Confidence threshold (flag lines under)")
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.Tab("Image and lines"):
                        view = gr.HTML("<p>Upload a scan and press Transcribe.</p>")
                    with gr.Tab("Transcription"):
                        text_box = make(gr.Textbox, label="Transcription (editable)", lines=22, interactive=True,
                                        show_copy_button=True, rtl=True)
                        again = gr.Button("Parse the corrected text again")
                    with gr.Tab("Spelling"):
                        spelling_view = make(gr.HighlightedText, label="Proposed spelling corrections", rtl=True,
                                             show_legend=False, show_inline_category=True, combine_adjacent=False,
                                             interactive=False)
                        with gr.Row():
                            accept_all = gr.Button("Back to the rules' choice")
                            refuse_all = gr.Button("Refuse all")
                            apply_button = gr.Button("Apply and parse again", variant="primary")
                        spelling_details = gr.Markdown("Tick \"Spelling correction\" in the options, then Transcribe.",
                                                       rtl=False)
                    with gr.Tab("JSON record"):
                        json_out = gr.JSON(label="Record")
                    with gr.Tab("Validation"):
                        report_box = gr.Textbox(label="Validation report and parser notes", lines=18, interactive=False)
                    with gr.Tab("Downloads"):
                        files = make(gr.File, label="record.json · transcription.txt · segmentation.xml", file_count="multiple")
        run.click(
            transcribe,
            [image, use_ner, use_translation, use_spellcheck, threshold],
            [view, text_box, json_out, report_box, files, lines_state, image_state,
             spelling_view, spelling_details, spelling_state],
        )
        review_outputs = [spelling_view, spelling_details, spelling_state]
        spelling_view.select(toggle_change, [spelling_state], review_outputs)
        accept_all.click(lambda state: set_all(state, True), [spelling_state], review_outputs)
        refuse_all.click(lambda state: set_all(state, False), [spelling_state], review_outputs)
        apply_button.click(apply_spelling, [spelling_state, use_ner, use_translation],
                           [text_box, json_out, report_box, files])
        again.click(reparse, [text_box, use_ner, use_translation], [json_out, report_box, files])
        threshold.release(rerender, [image_state, lines_state, threshold], [view])
    return app


def main():
    parser = argparse.ArgumentParser(description="Gradio interface of the birth certificate pipeline.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="public gradio.live link")
    parser.add_argument("--device", default="cpu", help="kraken device: cpu, cuda:0 ...")
    parser.add_argument("--segmentation-model", type=Path, default=MODEL_SEGMENTATION)
    parser.add_argument("--recognition-model", type=Path, default=MODEL_RECOGNITION)
    args = parser.parse_args()
    # checked once at start: a GPU that torch cannot use falls back to the CPU with a warning
    app = build_app(resolve_device(args.device), args.segmentation_model, args.recognition_model)
    app.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()