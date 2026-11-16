"""The documents of the web application: one folder per document under <data dir>/docs.

    <id>/doc.json        the state: pages (scan, lines, text), record, report, job
    <id>/page1.* page2.* the scans as kraken and the browser read them (upright, RGB)
    <id>/original/       the files as they were uploaded (for the download)

A document is an act: page 2 (the details, the page the record is read from) and, when given, page 1 (the names
and the marginal notes). The documents hold personal data: a document is deleted `retention_hours` after its last
use, and nothing of their content is written to the logs.
"""

import json
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

from PIL import Image, ImageOps

from madoc_web import config

PAGES = ("2", "1")          # the order of the interface: page 2 is the one that must be there
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")
MAX_NAME_LENGTH = 120
_lock = threading.RLock()   # one writer at a time (the API threads and the worker)
ID = re.compile(r"[0-9a-f]{32}")


class NotFound(KeyError):
    pass


class BadRequest(ValueError):
    pass


def root():
    return config.DATA_DIR / "docs"


def folder(doc_id):
    if not ID.fullmatch(doc_id or ""):
        raise NotFound(doc_id)
    return root() / doc_id


def file_name(name):
    """A safe file name from the uploaded one (no folder, no control character, not too long)."""
    name = Path(str(name or "scan").replace("\\", "/")).name
    name = re.sub(r"[\x00-\x1f<>:\"|?*]", "_", name).strip(" .") or "scan"
    stem, suffix = Path(name).stem, Path(name).suffix
    return stem[:MAX_NAME_LENGTH - len(suffix)] + suffix


def empty_page():
    return {"image": "", "original": "", "name": "", "width": 0, "height": 0, "state": "empty", "lines": []}


def new(name=""):
    doc_id = uuid.uuid4().hex
    now = time.time()
    doc = {"id": doc_id, "name": file_name(name) if name else "", "created": now, "updated": now, "version": 0,
           "pages": {page: empty_page() for page in PAGES},
           "record": None, "auto_record": None, "record_edits": {}, "report": None, "notes": [],
           "field_lines": {}, "stale": False, "spelling": None, "job": None, "log": []}
    with _lock:
        (folder(doc_id) / "original").mkdir(parents=True)
        _write(doc)
    return doc


def _write(doc):
    path = folder(doc["id"]) / "doc.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def load(doc_id):
    path = folder(doc_id) / "doc.json"
    with _lock:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise NotFound(doc_id) from None


def update(doc_id, change, touch=True):
    """Load the document, let `change(doc)` modify it, save it with a new version -> the document."""
    with _lock:
        doc = load(doc_id)
        change(doc)
        doc["version"] += 1
        if touch:
            doc["updated"] = time.time()
        _write(doc)
        return doc


def delete(doc_id):
    with _lock:
        shutil.rmtree(folder(doc_id), ignore_errors=True)


def summaries():
    """The documents, the last used first: what the list of the home page shows."""
    out = []
    if not root().is_dir():
        return out
    for path in root().iterdir():
        try:
            doc = load(path.name)
        except NotFound:
            continue
        record = doc.get("record") or {}
        out.append({"id": doc["id"], "name": doc["name"], "updated": doc["updated"],
                    "pages": [p for p in PAGES if doc["pages"][p]["state"] != "empty"],
                    "act": record.get("NUMACTE") or "", "valid": (doc.get("report") or {}).get("valid"),
                    "job": (doc.get("job") or {}).get("state")})
    return sorted(out, key=lambda d: -d["updated"])


def purge(now=None):
    """Delete the documents not used for `retention_hours` -> how many."""
    limit = (now or time.time()) - config.load()["retention_hours"] * 3600
    count = 0
    if not root().is_dir():
        return count
    for path in list(root().iterdir()):
        try:
            if load(path.name)["updated"] < limit:
                delete(path.name)
                count += 1
        except NotFound:
            if path.is_dir() and path.stat().st_mtime < limit:   # a folder without its state: nothing to keep
                shutil.rmtree(path, ignore_errors=True)
    return count


def scan_to_rgb(image):
    """The scan in 8-bit RGB (a 16-bit TIFF of a scanner is scaled, not clipped)."""
    if image.mode.startswith("I;16") or image.mode in ("I", "F"):
        import numpy as np
        values = np.asarray(image, dtype=np.float64)
        low, high = float(values.min()), float(values.max())
        if image.mode.startswith("I;16") and high > 255:
            low, high = 0.0, 65535.0
        scaled = (values - low) * (255.0 / (high - low)) if high > low else np.zeros_like(values)
        image = Image.fromarray(scaled.clip(0, 255).astype(np.uint8), "L")
    return image.convert("RGB")


def set_page(doc_id, page, data, name):
    """Store an uploaded scan as page `page` of the document (its lines and the record start again)."""
    if page not in PAGES:
        raise BadRequest("page: 1 or 2")
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise BadRequest(f"the file is larger than {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    name = file_name(name)
    suffix = Path(name).suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        suffix = ".jpg"
        name += suffix
    base = folder(doc_id)
    with _lock:
        load(doc_id)   # the document exists
        for old in list(base.glob(f"page{page}.*")) + list((base / "original").glob(f"page{page}_*")):
            old.unlink()
        original = base / "original" / f"page{page}_{name}"
        original.write_bytes(data)
        try:
            with Image.open(original) as image:
                image.load()
                rotated = image.getexif().get(0x0112, 1) not in (None, 1)   # EXIF orientation (phone photos)
                if not rotated and image.mode in ("RGB", "L") and suffix in (".jpg", ".jpeg", ".png"):
                    prepared = base / f"page{page}{suffix}"
                    shutil.copyfile(original, prepared)
                    size = image.size
                else:
                    prepared = base / f"page{page}.png"
                    upright = scan_to_rgb(ImageOps.exif_transpose(image))
                    upright.save(prepared)
                    size = upright.size
        except Exception as e:   # not an image, a truncated file, a decompression bomb
            original.unlink(missing_ok=True)
            raise BadRequest(f"this file cannot be read as a scan ({type(e).__name__}): JPEG, PNG or TIFF") from None

        def change(doc):
            doc["pages"][page] = {**empty_page(), "image": prepared.name, "original": original.name, "name": name,
                                  "width": size[0], "height": size[1], "state": "uploaded"}
            if not doc["name"] or page == "2":
                doc["name"] = name
            doc.update(record=None, auto_record=None, report=None, notes=[], field_lines={}, stale=False,
                       spelling=None, record_edits={})
        return update(doc_id, change)


def remove_page(doc_id, page):
    if page not in PAGES:
        raise BadRequest("page: 1 or 2")
    base = folder(doc_id)
    with _lock:
        for old in list(base.glob(f"page{page}.*")) + list((base / "original").glob(f"page{page}_*")):
            old.unlink()

        def change(doc):
            doc["pages"][page] = empty_page()
            doc["stale"] = bool(doc.get("record"))
        return update(doc_id, change)


def image_path(doc, page, original=False):
    name = doc["pages"][page]["original" if original else "image"]
    if not name:
        raise NotFound(page)
    path = folder(doc["id"]) / ("original" if original else "") / name
    if not path.is_file():
        raise NotFound(page)
    return path
