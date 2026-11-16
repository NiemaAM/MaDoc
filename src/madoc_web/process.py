"""The steps of a run, each on a stored document (store.py): what the worker (jobs.py) and the routes call.

    segment      a page -> its lines (polygon and baseline), no text yet
    recognize    the lines of a page -> their text and confidence (all, or only the lines changed by hand)
    extract      the text of page 2 -> the record (parser, mapper, validator; the local model only when asked),
                 completed and compared with page 1 (names, act number, marginal notes)
    spelling     the spelling proposals on the text of page 2 (local model + rules), applied when the person
                 accepts them

The slow work (kraken, the models) is done outside the store's lock; the result is then written in one update.
"""

import hashlib
import io
import json
import time
import uuid
import zipfile
from collections import OrderedDict
from pathlib import Path

import translator
from common import TEMPLATE, normalize_text
from parser import find_zones, flatten
from pipeline import run_pipeline
from recognition import MODEL_RECOGNITION, parse_pagexml, recognize_lines
from segmentation import MODEL_SEGMENTATION, segment_image
from spelling_checker import apply_changes, describe, line_units, propose
from validator import validate

from madoc_web import config, pageone, pagexml, store

MODELS = {"segmentation": MODEL_SEGMENTATION, "recognition": MODEL_RECOGNITION}   # server.py may replace them
_ner = OrderedDict()     # (document, text hash) -> NER entities: the same text is not classified twice
NER_KEPT = 8
# the zone of the form where each field is read (to find the line it comes from)
FIELD_ZONES = {"child": ("NUMACTE", "DATENAISSAR", "DATENAISSFR", "HEURENAISS", "MINNAISS", "LIEUNAISS",
                         "PRENOMAR", "PRENOM", "SEXE", "NATIONALITE", "NOMAR", "NOM"),
               "father": ("PRENOMPEREAR", "NOMPEREAR", "NATIONALITEPERE", "ADRESSENAISSPERE", "DATENAISSPEREAR",
                          "DATENAISSPEREFR", "PROFESSIONPERE"),
               "mother": ("PRENOMMEREAR", "NOMMEREAR", "NATIONALITEMERE", "ADRESSENAISSMERE", "DATENAISSMEREAR",
                          "DATENAISSMEREFR", "PROFESSIONMERE"),
               "admin": ("ADRESSEPARENT", "TEXTEDECLA", "DATEDECLAAR", "DATEDECLAFR", "ANNEEDECLA",
                         "EMPLOYECREANT")}


class ProcessError(Exception):
    """A step that cannot run, with a message for the person."""


def line_id():
    return "L" + uuid.uuid4().hex[:10]


def new_line(polygon, baseline):
    return {"id": line_id(), "polygon": [[int(round(x)), int(round(y))] for x, y in polygon],
            "baseline": [[int(round(x)), int(round(y))] for x, y in baseline],
            "text": "", "htr": "", "confidence": None, "dirty": False}


# ---------------------------------------------------------------------------------------------------------
# segmentation, recognition
# ---------------------------------------------------------------------------------------------------------
def segment(doc_id, page):
    """Find the lines of a page -> how many."""
    doc = store.load(doc_id)
    image = store.image_path(doc, page)
    xml = segment_image(image, MODELS["segmentation"], config.device())
    lines = [new_line(found["polygon"], found["baseline"]) for found in parse_pagexml(xml)
             if len(found["polygon"]) >= 3 and len(found["baseline"]) >= 2]

    def change(doc):
        doc["pages"][page].update(lines=lines, state="segmented")
        doc["stale"] = bool(doc.get("record"))
        doc["spelling"] = None
    store.update(doc_id, change)
    return len(lines)


def pair(targets, found):
    """{id of a line sent to kraken: the line kraken read}: by id, else by polygon, else by order."""
    by_id = {line.get("id"): line for line in found}
    if any(target["id"] in by_id for target in targets):
        return {target["id"]: by_id[target["id"]] for target in targets if target["id"] in by_id}
    by_polygon = {tuple(map(tuple, line.get("polygon") or [])): line for line in found}
    paired = {target["id"]: by_polygon[tuple(map(tuple, target["polygon"]))] for target in targets
              if tuple(map(tuple, target["polygon"])) in by_polygon}
    if paired or len(found) != len(targets):
        return paired
    return {target["id"]: line for target, line in zip(targets, found)}


def recognize(doc_id, page, only_dirty=False):
    """Read the lines of a page (only_dirty: only the lines drawn or moved since the page was read) -> how
    many."""
    doc = store.load(doc_id)
    state = doc["pages"][page]
    image = store.image_path(doc, page)
    # a page never read is read whole, whatever was asked
    only_dirty = only_dirty and state["state"] == "recognized"
    targets = [line for line in state["lines"]
               if (line.get("dirty") or not only_dirty) and len(line["polygon"]) >= 3 and len(line["baseline"]) >= 2]
    read = {}
    if targets:
        xml = pagexml.build(targets, image, state["width"], state["height"])
        read = pair(targets, recognize_lines(xml, image, MODELS["recognition"], config.device()))
    wanted = {line["id"] for line in targets}

    def change(doc):
        for line in doc["pages"][page]["lines"]:
            if line["id"] in wanted:
                result = read.get(line["id"]) or {}
                text = normalize_text(result.get("text", ""))
                line.update(text=text, htr=text, confidence=result.get("confidence") if text else None,
                            dirty=False)
        doc["pages"][page]["state"] = "recognized"
        doc["stale"] = bool(doc.get("record"))
        doc["spelling"] = None
    store.update(doc_id, change)
    return len(targets)


def set_lines(doc_id, page, lines):
    """The lines of a page as the person left them in the editor: [{"id"?, "polygon", "baseline"}], in reading
    order. A line whose shape changed, or a new one, is marked to be read again; its text is kept meanwhile."""
    cleaned = []
    for line in lines:
        polygon, baseline = line.get("polygon") or [], line.get("baseline") or []
        try:
            polygon = [[int(round(float(x))), int(round(float(y)))] for x, y in polygon]
            baseline = [[int(round(float(x))), int(round(float(y)))] for x, y in baseline]
        except (TypeError, ValueError):
            raise store.BadRequest("a line has points that are not numbers") from None
        if len(polygon) < 3 or len(baseline) < 2 or len(polygon) > 2000:
            raise store.BadRequest("a line needs a polygon of 3 points or more and a baseline of 2 or more")
        cleaned.append({"id": str(line.get("id") or ""), "polygon": polygon, "baseline": baseline})
    if len(cleaned) > 1000:
        raise store.BadRequest("too many lines")

    def change(doc):
        state = doc["pages"][page]
        if state["state"] == "empty":
            raise store.BadRequest("this page has no scan")
        width, height = state["width"], state["height"]
        old = {line["id"]: line for line in state["lines"]}
        out, seen = [], set()
        for line in cleaned:
            for point in line["polygon"] + line["baseline"]:
                point[0], point[1] = min(max(point[0], 0), width), min(max(point[1], 0), height)
            before = old.get(line["id"]) if line["id"] not in seen else None
            if before is None:
                out.append({**new_line(line["polygon"], line["baseline"]), "dirty": True})
            else:
                seen.add(before["id"])
                moved = before["polygon"] != line["polygon"] or before["baseline"] != line["baseline"]
                out.append({**before, "polygon": line["polygon"], "baseline": line["baseline"],
                            "dirty": bool(before.get("dirty") or moved)})
        removed = len(old) - len(seen)
        state["lines"] = out
        if state["state"] == "uploaded":
            state["state"] = "segmented"
        if removed or any(line["dirty"] for line in out):
            doc["stale"] = bool(doc.get("record"))
            doc["spelling"] = None
    return store.update(doc_id, change)


def set_text(doc_id, page, line, text):
    """A line corrected by hand."""
    text = normalize_text(text)[:2000]

    def change(doc):
        for candidate in doc["pages"][page]["lines"]:
            if candidate["id"] == line:
                if candidate["text"] != text:
                    candidate["text"] = text
                    doc["stale"] = bool(doc.get("record"))
                    doc["spelling"] = None
                return
        raise store.NotFound(line)
    return store.update(doc_id, change)


# ---------------------------------------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------------------------------------
def read_lines(doc, page):
    """The lines of a page that have a text, in reading order."""
    return [line for line in doc["pages"][page]["lines"] if line["text"].strip()]


def page_text(doc, page):
    return "".join(line["text"] + "\n" for line in read_lines(doc, page))


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


def report_of(record, comparisons, translated):
    """The validation report of a record, with the differences between the two pages as warnings."""
    report = validate(record, check_translation=translated and translator.is_available())
    extra = pageone.issues(comparisons or [])
    if extra:
        report = {**report, "issues": report["issues"] + extra, "warnings": report["warnings"] + len(extra)}
    return report


def keep_edits(record, edits):
    """The hand edits of the record applied to a record just read -> (record, edits kept, notes): an edit is
    kept while the text gives the value it replaced; when the text gives another one, the text wins."""
    kept = {field: edit for field, edit in edits.items() if record.get(field) == edit["auto"]}
    notes = [f"[record] value read in the corrected text used instead of the hand edit: {field}"
             for field in sorted(set(edits) - set(kept))]
    if kept:
        record = {**record, **{field: edit["value"] for field, edit in kept.items()}}
        notes.append("[record] kept from the hand edits: " + ", ".join(sorted(kept)))
    return record, kept, notes


def extract(doc_id, translate=None):
    """The record of the document from the text of its pages. translate: fill the French fields with the local
    model (None: what the settings say) -> the document."""
    settings = config.load()
    translate = settings["use_translation"] if translate is None else translate
    doc = store.load(doc_id)
    lines = read_lines(doc, "2")
    if not lines:
        raise ProcessError("Page 2 has no text yet: read its lines first.")
    text = page_text(doc, "2")
    key = (doc_id, hashlib.sha256(text.encode("utf-8")).hexdigest(), settings["use_ner"], settings["use_french_ner"])
    stages = {}
    record, _, notes = run_pipeline(text, ner=_ner.get(key), use_ner=settings["use_ner"], use_translation=translate,
                                    use_french_ner=settings["use_french_ner"], stages=stages)
    if stages.get("ner") is not None:
        _ner[key] = stages["ner"]
        _ner.move_to_end(key)
        while len(_ner) > NER_KEPT:
            _ner.popitem(last=False)
    sources = {field: {"page": "2", "lines": [lines[i]["id"] for i in found if i < len(lines)]}
               for field, found in field_lines(text, stages.get("parsed")).items()}
    notes = [f"[parser] {note}" for note in notes]
    comparisons = []
    first = read_lines(doc, "1")
    if first:
        page_one = pageone.parse([line["text"] for line in first])
        record, comparisons, page_notes = pageone.merge(record, page_one)
        notes += page_notes
        for field, found in page_one["sources"].items():
            if field not in sources or not sources[field]["lines"]:
                sources[field] = {"page": "1", "lines": [first[i]["id"] for i in found]}
    auto = record
    version = doc["version"]

    def change(doc):
        if doc["version"] != version and (page_text(doc, "2") != text):
            raise ProcessError("The text changed while the record was read: extract it again.")
        final, kept, edit_notes = keep_edits(auto, doc.get("record_edits") or {})
        doc.update(record=final, auto_record=auto, record_edits=kept, comparisons=comparisons,
                   report=report_of(final, comparisons, translate), notes=notes + edit_notes,
                   field_lines=sources, stale=False, translated=bool(translate))
    return store.update(doc_id, change)


def edit_record(doc_id, changes):
    """Fields of the record written by hand: {field: value}. Each is remembered with the value read in the text
    it replaces, and kept when the text is read again as long as the text gives that same value."""
    for field, value in changes.items():
        if field not in TEMPLATE:
            raise store.BadRequest(f"unknown field: {field}")
        if field == "SEXE":
            if value not in (0, 1, ""):
                raise store.BadRequest("SEXE: 0, 1 or empty")
        elif isinstance(value, bool) or not isinstance(value, (str, int)) or len(str(value)) > 5000:
            raise store.BadRequest(f"{field}: a text or a number")

    def change(doc):
        if not doc.get("record"):
            raise store.BadRequest("this document has no record yet")
        auto = doc.get("auto_record") or doc["record"]
        for field, value in changes.items():
            # the hour and the minute are numbers in the record: typed in a field, they stay numbers
            if field in ("HEURENAISS", "MINNAISS") and isinstance(value, str) and value.strip().isdigit():
                value = int(value.strip())
            doc["record"][field] = value
            if value == auto.get(field):
                doc["record_edits"].pop(field, None)
            else:
                doc["record_edits"][field] = {"auto": auto.get(field), "value": value}
        doc["report"] = report_of(doc["record"], doc.get("comparisons"), doc.get("translated", False))
    return store.update(doc_id, change)


# ---------------------------------------------------------------------------------------------------------
# spelling
# ---------------------------------------------------------------------------------------------------------
def edited_words(before, after):
    """{(line, unit)} of the words of `after` that differ from `before` (same number of lines): written by hand."""
    from difflib import SequenceMatcher
    out = set()
    for index, (old, new) in enumerate(zip(before, after)):
        if old == new:
            continue
        a, b = [unit["text"] for unit in line_units(old)], [unit["text"] for unit in line_units(new)]
        for op, _, _, k1, k2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
            if op in ("replace", "insert"):
                out.update((index, k) for k in range(k1, k2))
    return out


def spelling(doc_id):
    """The spelling proposals on the text of page 2 -> how many. Nothing is applied: apply_spelling does it with
    the proposals the person kept. A correction on a word written by hand is proposed, never chosen by itself."""
    doc = store.load(doc_id)
    lines = read_lines(doc, "2")
    if not lines:
        raise ProcessError("Page 2 has no text yet: read its lines first.")
    base = "\n".join(line["text"] for line in lines)
    changes, skipped = propose(base)
    protect = edited_words([line.get("htr") or line["text"] for line in lines], [line["text"] for line in lines])
    out = []
    for item in changes:
        if not item.get("old"):
            continue      # nothing to replace: a note of the rules, not a proposal
        span = {(item["line"], k) for k in range(item["start"], max(item["end"], item["start"] + 1))}
        if item["accepted"] and span & protect:
            item = {**item, "accepted": False, "check": True, "reason": "written by hand"}
        out.append({**json.loads(json.dumps(item, ensure_ascii=False, default=str)), "label": describe(item),
                    "line_id": lines[item["line"]]["id"]})
    state = {"base": base, "line_ids": [line["id"] for line in lines], "changes": out,
             "notes": [str(note) for note in skipped]}
    store.update(doc_id, lambda doc: doc.update(spelling=state))
    return len(out)


def apply_spelling(doc_id, accepted):
    """Write the chosen proposals (indexes of the list) in the text of page 2."""
    def change(doc):
        state = doc.get("spelling")
        if not state:
            raise store.BadRequest("no spelling check to apply")
        lines = {line["id"]: line for line in doc["pages"]["2"]["lines"]}
        current = "\n".join(lines[i]["text"] for i in state["line_ids"] if i in lines)
        if current != state["base"]:
            raise store.BadRequest("the text changed since the check: check the spelling again")
        chosen = {int(i) for i in accepted if 0 <= int(i) < len(state["changes"])}
        corrected = apply_changes(state["base"], state["changes"], chosen).split("\n")
        if len(corrected) != len(state["line_ids"]):
            raise store.BadRequest("the corrections change the number of lines")
        for identifier, text in zip(state["line_ids"], corrected):
            lines[identifier]["text"] = normalize_text(text)
        doc["spelling"] = None
        doc["stale"] = bool(doc.get("record"))
    return store.update(doc_id, change)


# ---------------------------------------------------------------------------------------------------------
# downloads
# ---------------------------------------------------------------------------------------------------------
def export(doc, kind):
    """(file name, media type, bytes) of a download: json, text, xml1, xml2 or zip (everything)."""
    stem = Path(doc["name"] or "document").stem or "document"

    def xml(page):
        state = doc["pages"][page]
        return pagexml.build(state["lines"], state["name"], state["width"], state["height"], with_text=True)

    def text():
        parts = [page_text(doc, "2")]
        if read_lines(doc, "1"):
            parts.append("\n--- page 1 ---\n" + page_text(doc, "1"))
        return "".join(parts)

    if kind == "json":
        if not doc.get("record"):
            raise store.NotFound(kind)
        return f"{stem}.json", "application/json", json.dumps(doc["record"], ensure_ascii=False, indent=2).encode()
    if kind == "text":
        return f"{stem}.txt", "text/plain; charset=utf-8", text().encode("utf-8")
    if kind in ("xml1", "xml2"):
        page = kind[-1]
        if doc["pages"][page]["state"] == "empty":
            raise store.NotFound(kind)
        return f"{stem}_page{page}.xml", "application/xml", xml(page).encode("utf-8")
    if kind == "zip":
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for page in store.PAGES:
                state = doc["pages"][page]
                if state["state"] == "empty":
                    continue
                archive.write(store.image_path(doc, page, original=True), f"page{page}_{state['name']}")
                archive.writestr(f"{stem}_page{page}.xml", xml(page))
            archive.writestr(f"{stem}.txt", text())
            if doc.get("record"):
                archive.writestr(f"{stem}.json", json.dumps(doc["record"], ensure_ascii=False, indent=2))
        return f"{stem}.zip", "application/zip", buffer.getvalue()
    raise store.NotFound(kind)


def release_models():
    """Let the kraken models go and ask Ollama to unload its model -> {"kraken": n, "ollama": bool}."""
    import urllib.request

    import kraken_engine
    import spelling_checker
    released = kraken_engine.release()
    unloaded = False
    for model in {spelling_checker.MODEL, translator.MODEL}:
        try:
            request = urllib.request.Request(
                spelling_checker.ollama_url("/api/generate"), method="POST",
                data=json.dumps({"model": model, "keep_alive": 0}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=10):
                unloaded = True
        except Exception:
            pass      # Ollama is not there, or has nothing loaded
    return {"kraken": released, "ollama": unloaded, "at": time.time()}
