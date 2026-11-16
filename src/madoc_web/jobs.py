"""The runs: one worker thread takes the jobs one at a time (the GPU is shared), a job is a list of steps on a
document. Its state is kept in the document ("job"), so the interface shows every result as soon as its step
ends: the segmentation, then the text, then the record.

    {"id", "state": "queued" | "running" | "done" | "error", "steps": ["segment 2", ...], "index": 1,
     "label": "recognize 2", "error": ""}

The log (logger "madoc") has one line per step: the step, its duration, the device, the number of lines. Never
the text of a document."""

import logging
import queue
import threading
import time
import uuid

from madoc_web import config, monitor, process, store

log = logging.getLogger("madoc")
_queue = queue.Queue()
_thread = []
STEPS = ("segment", "recognize", "extract", "translate", "spelling")
monitor.queue_length[0] = lambda: _queue.qsize()


class Busy(Exception):
    pass


def plan(doc, kind, pages=None, only_dirty=False):
    """The steps of a run asked by the interface.

    kind: "auto" (everything that is missing, for the pages that have a scan), or one of STEPS."""
    settings = config.load()
    present = [page for page in store.PAGES if doc["pages"][page]["state"] != "empty"]
    pages = [page for page in (pages or present) if page in present]
    if kind == "auto":
        steps = []
        for page in pages:      # page 2 first, lines then text: its text is on the screen as early as possible
            if doc["pages"][page]["state"] == "uploaded":
                steps.append({"name": "segment", "page": page})
            if doc["pages"][page]["state"] in ("uploaded", "segmented"):
                steps.append({"name": "recognize", "page": page})
        if "2" in present:
            steps.append({"name": "extract", "translate": False})
            if settings["use_translation"]:
                steps.append({"name": "translate"})
        return steps
    if kind == "segment":
        return [{"name": "segment", "page": page} for page in pages]
    if kind == "recognize":
        if only_dirty:      # only the pages that have something to read
            pages = [page for page in pages if doc["pages"][page]["state"] != "recognized"
                     or any(line.get("dirty") for line in doc["pages"][page]["lines"])]
        return [{"name": "recognize", "page": page, "only_dirty": only_dirty} for page in pages]
    if kind == "extract":
        steps = [{"name": "extract", "translate": False}]
        return steps + ([{"name": "translate"}] if settings["use_translation"] else [])
    if kind in ("translate", "spelling"):
        return [{"name": kind}]
    raise store.BadRequest(f"unknown run: {kind}")


def label(step):
    return step["name"] + (f" {step['page']}" if step.get("page") else "")


def submit(doc_id, steps):
    """Queue the steps for a document -> the document. Busy when it already has a run."""
    if not steps:
        raise store.BadRequest("nothing to run")
    job = {"id": uuid.uuid4().hex[:12], "state": "queued", "steps": [label(step) for step in steps], "index": 0,
           "label": label(steps[0]), "error": "", "queued": time.time()}

    def change(doc):
        if (doc.get("job") or {}).get("state") in ("queued", "running"):
            raise Busy("this document is already being processed")
        doc["job"] = job
    doc = store.update(doc_id, change)
    _queue.put((doc_id, job["id"], steps))
    start()
    return doc


def run_step(doc_id, step):
    """One step -> the number of lines it worked on (None when that has no meaning)."""
    name = step["name"]
    if name == "segment":
        return process.segment(doc_id, step["page"])
    if name == "recognize":
        return process.recognize(doc_id, step["page"], step.get("only_dirty", False))
    if name == "extract":
        process.extract(doc_id, translate=step.get("translate"))
    elif name == "translate":
        process.extract(doc_id, translate=True)
    elif name == "spelling":
        process.spelling(doc_id)
    return None


def _set(doc_id, job_id, **values):
    def change(doc):
        if (doc.get("job") or {}).get("id") == job_id:
            doc["job"].update(values)
    try:
        store.update(doc_id, change, touch=False)
    except store.NotFound:
        pass      # the document was deleted while it was processed


def work(doc_id, job_id, steps):
    for index, step in enumerate(steps):
        _set(doc_id, job_id, state="running", index=index, label=label(step))
        begin = time.monotonic()
        device = config.device() if step["name"] in ("segment", "recognize") else "-"
        try:
            lines = run_step(doc_id, step)
        except store.NotFound:
            return
        except Exception as e:
            seconds = time.monotonic() - begin
            monitor.record_step(step["name"], seconds, False, device)
            # the type only in the log: a message of a library could quote the text of the document
            log.error("job=%s doc=%s step=%s seconds=%.2f device=%s ok=0 error=%s", job_id, doc_id[:8],
                      label(step), seconds, device, type(e).__name__)
            message = str(e) if isinstance(e, (process.ProcessError, store.BadRequest)) else (
                f"{label(step)} failed ({type(e).__name__}: {str(e)[:300]})")
            _set(doc_id, job_id, state="error", error=message)
            return
        seconds = time.monotonic() - begin
        monitor.record_step(step["name"], seconds, True, device, lines)
        log.info("job=%s doc=%s step=%s seconds=%.2f device=%s lines=%s ok=1", job_id, doc_id[:8], label(step),
                 seconds, device, lines)
    _set(doc_id, job_id, state="done", index=len(steps), label="")


def _run():
    while True:
        doc_id, job_id, steps = _queue.get()
        try:
            work(doc_id, job_id, steps)
        except Exception as e:      # the worker must survive anything
            log.error("job=%s worker error=%s", job_id, type(e).__name__)
            _set(doc_id, job_id, state="error", error=f"unexpected error ({type(e).__name__})")
        finally:
            _queue.task_done()


def start():
    if not _thread or not _thread[0].is_alive():
        _thread[:] = [threading.Thread(target=_run, daemon=True, name="madoc-worker")]
        _thread[0].start()


def wait(timeout=60):
    """Until the queue is empty (the tests)."""
    end = time.time() + timeout
    while _queue.unfinished_tasks and time.time() < end:
        time.sleep(0.02)
    return not _queue.unfinished_tasks


def recover():
    """At start: a run that a restart interrupted is marked as failed (its steps can be run again)."""
    if not store.root().is_dir():
        return
    for path in store.root().iterdir():
        try:
            doc = store.load(path.name)
        except store.NotFound:
            continue
        if (doc.get("job") or {}).get("state") in ("queued", "running"):
            _set(doc["id"], doc["job"]["id"], state="error", error="interrupted by a restart of the server: run again")
