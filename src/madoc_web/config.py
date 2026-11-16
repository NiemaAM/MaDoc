"""The settings of the web application: defaults, then the environment, then what the Settings page saved.

They are kept in <data dir>/settings.json and applied to the modules that use them (kraken_engine,
spelling_checker, translator) at start and at every change. MADOC_DATA_DIR chooses the data folder (default: a
folder of the temporary directory; in Docker, a volume)."""

import json
import os
import tempfile
import threading
from pathlib import Path

DATA_DIR = Path(os.environ.get("MADOC_DATA_DIR") or Path(tempfile.gettempdir()) / "madoc_data")
MAX_UPLOAD_BYTES = int(os.environ.get("MADOC_MAX_UPLOAD_MB", "25")) * 1024 * 1024

DEFAULTS = {
    "device": "auto",              # "auto" (the first GPU when there is one), "cpu", "cuda:0" ...
    "cpu_threads": 1,              # CPU threads of kraken (1 gives exactly the command line's output)
    "keep_models_seconds": 900,    # the kraken models are let go after this long without a scan (0: kept)
    "auto_run": True,              # after the upload: segmentation, recognition and formatting in a row
    "use_ner": True,
    "use_french_ner": True,
    "use_translation": False,      # the local model fills the French fields at every formatting (else: its button)
    "confidence_threshold": 0.8,
    "spell_model": "",             # "" = MADOC_SPELL_MODEL / the default of spelling_checker.py
    "translate_model": "",         # "" = MADOC_TRANSLATE_MODEL / the spelling model
    "retention_hours": 1,          # a document is deleted this long after its last use
}
LIMITS = {"cpu_threads": (1, 64), "keep_models_seconds": (0, 86400), "confidence_threshold": (0.3, 0.99),
          "retention_hours": (1, 168)}
_lock = threading.Lock()
_cache = {}


def _file():
    return DATA_DIR / "settings.json"


def clean(values):
    """Only the known keys, with the type of their default and inside their limits (ValueError otherwise)."""
    out = {}
    for key, value in values.items():
        if key not in DEFAULTS:
            raise ValueError(f"unknown setting: {key}")
        default = DEFAULTS[key]
        if isinstance(default, bool):
            if not isinstance(value, bool):
                raise ValueError(f"{key}: true or false")
        elif isinstance(default, (int, float)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key}: a number")
            low, high = LIMITS[key]
            if not low <= value <= high:
                raise ValueError(f"{key}: between {low} and {high}")
            value = int(value) if isinstance(default, int) else float(value)
        else:
            if not isinstance(value, str) or len(value) > 100:
                raise ValueError(f"{key}: a short text")
            value = value.strip()
            if key == "device" and value not in ("auto", "cpu") and not (
                    value.startswith("cuda:") and value[5:].isdigit()):
                raise ValueError("device: auto, cpu or cuda:N")
        out[key] = value
    return out


def load():
    """The settings now (a copy)."""
    with _lock:
        if not _cache:
            values = dict(DEFAULTS)
            if os.environ.get("MADOC_DEVICE"):
                values["device"] = os.environ["MADOC_DEVICE"]
            if os.environ.get("MADOC_KRAKEN_KEEP_SECONDS", "").isdigit():
                values["keep_models_seconds"] = int(os.environ["MADOC_KRAKEN_KEEP_SECONDS"])
            if os.environ.get("MADOC_KRAKEN_THREADS", "").isdigit():
                values["cpu_threads"] = int(os.environ["MADOC_KRAKEN_THREADS"])
            try:
                saved = json.loads(_file().read_text(encoding="utf-8"))
                values.update(clean({k: v for k, v in saved.items() if k in DEFAULTS}))
            except (OSError, ValueError):
                pass    # no file yet, or a file written by another version: the defaults
            _cache.update(values)
        return dict(_cache)


def save(changes):
    """Change some settings (checked), keep them and apply them -> the settings."""
    changes = clean(changes)
    values = {**load(), **changes}
    with _lock:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        temporary = _file().with_suffix(".tmp")
        temporary.write_text(json.dumps(values, indent=2), encoding="utf-8")
        os.replace(temporary, _file())
        _cache.clear()
        _cache.update(values)
    apply(values)
    return dict(values)


def reset_cache():
    with _lock:
        _cache.clear()


_devices = []


def devices():
    """The devices kraken can use here: [{"id": "cpu", "name": ...}, {"id": "cuda:0", "name": ...}].

    Asked once (importing torch takes seconds; the server asks it in the background at start)."""
    if _devices:
        return list(_devices)
    found = [{"id": "cpu", "name": f"CPU ({os.cpu_count() or 1} threads)"}]
    try:
        import torch
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                found.append({"id": f"cuda:{index}", "name": torch.cuda.get_device_name(index)})
    except Exception:   # no torch (the tests), or a broken CUDA: the CPU only
        pass
    _devices[:] = found
    return found


def device(values=None):
    """The kraken device of the settings ("auto": the first GPU, else the CPU)."""
    wanted = (values or load())["device"]
    available = [d["id"] for d in devices()]
    if wanted == "auto":
        return available[1] if len(available) > 1 else "cpu"
    return wanted if wanted in available else "cpu"


def apply(values=None):
    """Give the settings to the modules that read them as module variables."""
    values = values or load()
    import kraken_engine
    import spelling_checker
    import translator
    kraken_engine.THREADS = values["cpu_threads"]
    kraken_engine.KEEP_SECONDS = values["keep_models_seconds"]
    if values["spell_model"]:
        spelling_checker.MODEL = values["spell_model"]
    if values["translate_model"] or values["spell_model"]:
        translator.MODEL = values["translate_model"] or values["spell_model"]
