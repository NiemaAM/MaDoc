"""The HTTP routes of the web application (FastAPI).

    GET  /                       the interface (static/index.html)
    GET  /healthz                the server answers (no login: the healthcheck of Docker)
    GET  /metrics                the numbers of the Monitoring page for Prometheus
    /api/meta, /api/settings     what the interface needs to draw itself; the settings
    /api/docs...                 the documents: scans, lines, text, record, runs, downloads
    /api/monitor                 the samples of the Monitoring page

A login is asked (HTTP Basic) when MADOC_AUTH=user:password[,user2:password2] is set. Every change to a document
is refused (409) while a run works on it.
"""

import asyncio
import base64
import json
import secrets
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

import spelling_checker
import translator
from common import TEMPLATE

from madoc_web import config, jobs, monitor, process, store

STATIC = Path(__file__).resolve().parent / "static"
MODEL_CARD = Path(__file__).resolve().parents[2] / "models" / "MODEL_CARD.json"
LOGOS = Path(__file__).resolve().parents[2] / "logo"
# the record as the interface shows it: (group, label, Arabic field, French field or None)
RECORD_ROWS = [
    ("act", "act_number", "NUMACTE", None),
    ("child", "first_name", "PRENOMAR", "PRENOM"), ("child", "family_name", "NOMAR", "NOM"),
    ("child", "sex", "SEXE", None), ("child", "birth_date", "DATENAISSAR", "DATENAISSFR"),
    ("child", "hour", "HEURENAISS", None), ("child", "minute", "MINNAISS", None),
    ("child", "birthplace", "LIEUNAISS", "LIEUNAISSFR"), ("child", "nationality", "NATIONALITE", "NATIONALITEFR"),
    ("father", "name", "PRENOMPEREAR", "PRENOMPEREFR"), ("father", "lineage", "NOMPEREAR", "NOMPEREFR"),
    ("father", "birth_date", "DATENAISSPEREAR", "DATENAISSPEREFR"), ("father", "birthplace", "ADRESSENAISSPERE", None),
    ("father", "nationality", "NATIONALITEPERE", "NATIONALITEPEREFR"),
    ("father", "job", "PROFESSIONPERE", "PROFESSIONPEREFR"),
    ("mother", "name", "PRENOMMEREAR", "PRENOMMEREFR"), ("mother", "lineage", "NOMMEREAR", "NOMMEREFR"),
    ("mother", "birth_date", "DATENAISSMEREAR", "DATENAISSMEREFR"), ("mother", "birthplace", "ADRESSENAISSMERE", None),
    ("mother", "nationality", "NATIONALITEMERE", "NATIONALITEMEREFR"),
    ("mother", "job", "PROFESSIONMERE", "PROFESSIONMEREFR"),
    ("declaration", "address", "ADRESSEPARENT", None), ("declaration", "declaration", "TEXTEDECLA", None),
    ("declaration", "declaration_date", "DATEDECLAAR", "DATEDECLAFR"), ("declaration", "year", "ANNEEDECLA", None),
    ("declaration", "officer", "EMPLOYECREANT", "EMPLOYECREANTFR"), ("declaration", "delegation", "BECNAME", "BECNAMEFR"),
    ("notes", "marginal_notes", "TEXTEMENTIONS", "TEXTEMENTIONSFR"),
]
PUBLIC = ("/healthz",)      # asked without a login


def logins(text):
    """"ana:1234,sara:abcd" -> {"ana": "1234", "sara": "abcd"} (empty text: no login asked)."""
    pairs = {}
    for item in (text or "").split(","):
        user, _, password = item.strip().partition(":")
        if user and password:
            pairs[user] = password
        elif item.strip():
            raise SystemExit(f"--auth / MADOC_AUTH: {item.strip()!r} is not USER:PASSWORD")
    return pairs


def allowed(header, users):
    """Is this Authorization header one of the logins (compared in constant time)?"""
    if not header.lower().startswith("basic "):
        return False
    try:
        user, _, password = base64.b64decode(header[6:]).decode("utf-8").partition(":")
    except (ValueError, UnicodeDecodeError):
        return False
    ok = False
    for known, secret in users.items():      # every login is compared: the time does not tell which user exists
        match = secrets.compare_digest(user.encode(), known.encode()) and secrets.compare_digest(
            password.encode(), secret.encode())
        ok = ok or match
    return ok


def public(doc):
    """The document as the interface gets it (without what only the server needs)."""
    return {key: value for key, value in doc.items() if key != "auto_record"}


def create_app(auth=None):
    users = dict(auth or {})

    @asynccontextmanager
    async def lifespan(app):
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        config.apply()
        jobs.recover()
        jobs.start()
        monitor.start()
        stop = threading.Event()

        def cleaner():
            while not stop.wait(600):
                try:
                    store.purge()
                except Exception:
                    pass
        store.purge()
        threading.Thread(target=cleaner, daemon=True, name="madoc-purge").start()
        threading.Thread(target=config.devices, daemon=True, name="madoc-devices").start()   # imports torch
        yield
        stop.set()

    # no Swagger page: it loads its scripts from a CDN, which this server never does (/api/openapi.json stays)
    app = FastAPI(title="MaDoc", docs_url=None, redoc_url=None, openapi_url="/api/openapi.json", lifespan=lifespan)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if users and request.url.path not in PUBLIC and not allowed(request.headers.get("authorization", ""), users):
            return Response("Login required.", status_code=401, headers={"WWW-Authenticate": 'Basic realm="MaDoc"'})
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "DENY")
        # Alpine evaluates the expressions of the page: 'unsafe-eval'; everything is served by this server
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; "
            "script-src 'self' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")     # personal data: never kept by a cache
        return response

    @app.exception_handler(store.NotFound)
    async def not_found(request, error):
        return JSONResponse({"detail": "not found"}, status_code=404)

    @app.exception_handler(store.BadRequest)
    async def bad_request(request, error):
        return JSONResponse({"detail": str(error)}, status_code=400)

    @app.exception_handler(jobs.Busy)
    async def busy(request, error):
        return JSONResponse({"detail": str(error)}, status_code=409)

    def idle(doc_id):
        doc = store.load(doc_id)
        if (doc.get("job") or {}).get("state") in ("queued", "running"):
            raise jobs.Busy("this document is being processed: wait for the end of the run")
        return doc

    def page_of(page):
        if page not in store.PAGES:
            raise store.NotFound(page)
        return page

    # -- the interface ---------------------------------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/logo/{kind}", include_in_schema=False)
    def logo(kind: str):
        name = {"text": "with_text.png", "dark": "madoc_dark.png", "icon": "single.png"}.get(kind)
        path = LOGOS / name if name else None
        if not path or not path.is_file():
            raise store.NotFound(kind)
        return FileResponse(path, headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/healthz", include_in_schema=False)
    def health():
        return {"status": "ok"}

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        return PlainTextResponse(monitor.prometheus())

    @app.get("/api/meta")
    def meta():
        try:
            card = json.loads(MODEL_CARD.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            card = None
        return {"rows": RECORD_ROWS, "fields": list(TEMPLATE), "pages": list(store.PAGES),
                "devices": config.devices(), "device": config.device(), "max_upload_mb":
                config.MAX_UPLOAD_BYTES // (1024 * 1024), "model_card": card,
                "models": {kind: Path(path).name for kind, path in process.MODELS.items()},
                "spell_model": spelling_checker.MODEL, "translate_model": translator.MODEL}

    # -- settings --------------------------------------------------------------------------------------------
    @app.get("/api/settings")
    def get_settings():
        return {"values": config.load(), "limits": config.LIMITS, "devices": config.devices(),
                "device": config.device(), "spell_model": spelling_checker.MODEL,
                "translate_model": translator.MODEL, "translation_unavailable": translator.unavailable_reason()}

    @app.put("/api/settings")
    def put_settings(values: dict = Body(...)):
        try:
            config.save(values)
        except ValueError as e:
            raise store.BadRequest(str(e)) from None
        return get_settings()

    @app.post("/api/settings/release")
    def release():
        return process.release_models()

    # -- documents -------------------------------------------------------------------------------------------
    @app.get("/api/docs")
    def list_docs():
        return {"docs": store.summaries(), "retention_hours": config.load()["retention_hours"]}

    @app.post("/api/docs")
    def create_doc():
        store.purge()
        return public(store.new())

    @app.get("/api/docs/{doc_id}")
    def get_doc(doc_id: str):
        return public(store.load(doc_id))

    @app.get("/api/docs/{doc_id}/status")
    def status(doc_id: str):
        doc = store.load(doc_id)
        return {"version": doc["version"], "job": doc.get("job")}

    @app.delete("/api/docs/{doc_id}")
    def delete_doc(doc_id: str):
        idle(doc_id)
        store.delete(doc_id)
        return {"deleted": doc_id}

    @app.put("/api/docs/{doc_id}/pages/{page}")
    async def put_page(doc_id: str, page: str, file: UploadFile = File(...)):
        idle(doc_id)
        data = await file.read(config.MAX_UPLOAD_BYTES + 1)
        # decoding a large photo takes a moment: not on the event loop
        doc = await asyncio.to_thread(store.set_page, doc_id, page_of(page), data, file.filename or "scan.jpg")
        return public(doc)

    @app.delete("/api/docs/{doc_id}/pages/{page}")
    def delete_page(doc_id: str, page: str):
        idle(doc_id)
        return public(store.remove_page(doc_id, page_of(page)))

    @app.get("/api/docs/{doc_id}/pages/{page}/image")
    def page_image(doc_id: str, page: str):
        doc = store.load(doc_id)
        return FileResponse(store.image_path(doc, page_of(page)), headers={"Cache-Control": "private, max-age=3600"})

    @app.put("/api/docs/{doc_id}/pages/{page}/lines")
    def put_lines(doc_id: str, page: str, body: dict = Body(...)):
        idle(doc_id)
        lines = body.get("lines")
        if not isinstance(lines, list):
            raise store.BadRequest("lines: a list")
        return public(process.set_lines(doc_id, page_of(page), lines))

    @app.patch("/api/docs/{doc_id}/pages/{page}/lines/{line}")
    def patch_line(doc_id: str, page: str, line: str, body: dict = Body(...)):
        idle(doc_id)
        if not isinstance(body.get("text"), str):
            raise store.BadRequest("text: a text")
        return public(process.set_text(doc_id, page_of(page), line, body["text"]))

    @app.post("/api/docs/{doc_id}/run")
    def run(doc_id: str, body: dict = Body(...)):
        doc = idle(doc_id)
        pages = body.get("pages")
        if pages is not None and (not isinstance(pages, list) or any(p not in store.PAGES for p in pages)):
            raise store.BadRequest("pages: a list of 1 and 2")
        steps = jobs.plan(doc, str(body.get("kind", "auto")), pages, bool(body.get("only_dirty")))
        if not steps:
            return public(doc)
        return public(jobs.submit(doc_id, steps))

    @app.post("/api/docs/{doc_id}/spelling/apply")
    def apply_spelling(doc_id: str, body: dict = Body(...)):
        idle(doc_id)
        accepted = body.get("accepted")
        if not isinstance(accepted, list) or any(isinstance(i, bool) or not isinstance(i, int) for i in accepted):
            raise store.BadRequest("accepted: a list of indexes")
        return public(process.apply_spelling(doc_id, accepted))

    @app.delete("/api/docs/{doc_id}/spelling")
    def drop_spelling(doc_id: str):
        idle(doc_id)
        return public(store.update(doc_id, lambda doc: doc.update(spelling=None)))

    @app.patch("/api/docs/{doc_id}/record")
    def patch_record(doc_id: str, body: dict = Body(...)):
        idle(doc_id)
        return public(process.edit_record(doc_id, body))

    @app.get("/api/docs/{doc_id}/download/{kind}")
    def download(doc_id: str, kind: str):
        name, media, data = process.export(store.load(doc_id), kind)
        from urllib.parse import quote
        return Response(data, media_type=media, headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}"})

    # -- monitoring ------------------------------------------------------------------------------------------
    @app.get("/api/monitor")
    def monitoring(minutes: int = 15):
        if not 1 <= minutes <= 60:
            raise HTTPException(400, "minutes: 1 to 60")
        return monitor.snapshot(minutes)

    return app
