"""Shared helpers used by every stage: paths, dictionaries, JSON template and Arabic text normalisation.

The stages (classificator > parser > converter > mapper > translator > validator) only talk to each
other through plain JSON objects, everything they all need lives here.
"""

import json
import os
import re
import sys
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

# project root = the folder that contains src/ (this file is src/common.py)
ROOT = Path(__file__).resolve().parents[1]


def _find_dictionary_dir():
    """Dictionaries: $BIRTH_DICT_DIR, ./dictionaries, the project root, or next to the scripts."""
    candidates = [
        os.environ.get("BIRTH_DICT_DIR"),
        ROOT / "dictionaries",
        ROOT,
        Path(__file__).resolve().parent,
    ]
    for candidate in candidates:
        if candidate and (Path(candidate) / "date_time.json").is_file():
            return Path(candidate)
    raise FileNotFoundError(
        "dictionaries not found: put date_time.json, jobs.json and nationalities.json "
        "in <project>/dictionaries (or set BIRTH_DICT_DIR)"
    )


BASE_DIR = _find_dictionary_dir()


def _load_json(folder, name):
    with open(Path(folder) / name, encoding="utf-8") as f:
        return json.load(f)


# Load dictionaries
dates_data = _load_json(BASE_DIR, "date_time.json")
jobs_data = _load_json(BASE_DIR, "jobs.json")
nationalities_data = _load_json(BASE_DIR, "nationalities.json")

# official place names (Arabic -> official French spelling)
places_data = {}
# 1. cities.json: list of {"name": French, "nameAr": Arabic, ...} with every Moroccan city
for _folder in (BASE_DIR, ROOT):
    if (_folder / "cities.json").is_file():
        for _city in _load_json(_folder, "cities.json"):
            if isinstance(_city, dict) and _city.get("nameAr") and _city.get("name"):
                places_data[_city["nameAr"].strip()] = _city["name"].strip()
        break
# 2. places.json (optional): manual entries, they override cities.json
for _folder in (BASE_DIR, ROOT):
    if (_folder / "places.json").is_file():
        places_data.update(_load_json(_folder, "places.json"))
        break

# Json teplate (the output of the parser, every later stage fills / converts some of its fields)
TEMPLATE = {
    # child
    "DATENAISSFR": "",
    "DATENAISSAR": "",
    "HEURENAISS": "",
    "MINNAISS": "",
    "LIEUNAISS": "",
    "LIEUNAISSFR": "",
    "PRENOMAR": "",
    "PRENOM": "",
    "SEXE": "",
    "NATIONALITE": "",
    "NATIONALITEFR": "",
    "NOMAR": "",
    "NOM": "",
    # father
    "NOMPEREFR": "",
    "NOMPEREAR": "",
    "PRENOMPEREAR": "",
    "PRENOMPEREFR": "",
    "NATIONALITEPERE": "",
    "NATIONALITEPEREFR": "",
    "ADRESSENAISSPERE": "",
    "DATENAISSPEREFR": "",
    "DATENAISSPEREAR": "",
    "PROFESSIONPERE": "",
    "PROFESSIONPEREFR": "",
    # mother
    "NOMMEREFR": "",
    "NOMMEREAR": "",
    "PRENOMMEREAR": "",
    "PRENOMMEREFR": "",
    "NATIONALITEMERE": "",
    "NATIONALITEMEREFR": "",
    "ADRESSENAISSMERE": "",
    "DATENAISSMEREFR": "",
    "DATENAISSMEREAR": "",
    "PROFESSIONMERE": "",
    "PROFESSIONMEREFR": "",
    # administration
    "ANNEEDECLA": "",
    "NUMACTE": "",
    "ADRESSEPARENT": "",
    "TEXTEDECLA": "",
    "DATEDECLAFR": "",
    "DATEDECLAAR": "",
    "EMPLOYECREANT": "",
    "EMPLOYECREANTFR": "",  # NAME + FONCTIONAIRE ET OFFECIER DETAT CIVIL
    "BECNAME": "بالتفويض",
    "BECNAMEFR": "PAR DELEGATION",
}

# fixed French job title of the civil status employee
EMPLOYEE_TITLE = "FONCTIONAIRE ET OFFECIER DETAT CIVIL"

# --- Arabic normalisation helpers (used for reliable dictionary matching) ---
# diacritics (tashkeel) + tatweel
ARABIC_DIACRITICS = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed\u0640]")
# invisible / bidi control characters that often come from copy-paste or OCR
INVISIBLE_CHARS = re.compile("[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
# Arabic-Indic and Persian digits -> ASCII digits
DIGIT_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
# letter variants -> one canonical letter (only used to build comparison keys)
CHAR_MAP = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي",
    "ڭ": "ك", "گ": "ك", "ڤ": "ف", "پ": "ب", "چ": "ج",
})
# common spoken / misspelled variants found in civil registry scans
SPELLING_VARIANTS = {"تاني": "ثاني", "تالت": "ثالث", "عشري": "عشرين"}


# clean txt from spaces
def normalize_text(value):
    """Normalize whitespace for reliable dictionary matching."""
    return re.sub(r"\s+", " ", str(value)).strip()


@lru_cache(maxsize=65536)
def _ar_key(text):
    text = normalize_text(text)
    text = INVISIBLE_CHARS.sub("", text)
    text = ARABIC_DIACRITICS.sub("", text)
    text = text.translate(CHAR_MAP)
    tokens = []
    for token in text.split():
        # drop the definite article so "الخمسين" matches "خمسين"
        if token.startswith("ال") and len(token) > 3:
            token = token[2:]
        tokens.append(SPELLING_VARIANTS.get(token, token))
    return " ".join(tokens)


def ar_key(value):
    """Build a comparison key: no diacritics, unified letters, no leading 'ال' on words.

    Called thousands of times per record with the same words: the result is cached (see _ar_key).
    """
    return _ar_key(value if isinstance(value, str) else str(value))


# extract based on patterns
def extract(pattern, text):
    """Return the first captured value, or an empty string."""
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    return normalize_text(match.group(1)) if match else ""


def extract_any(patterns, text):
    """Try several patterns in order and return the first non empty capture."""
    for pattern in patterns:
        value = extract(pattern, text)
        if value:
            return value
    return ""


@lru_cache(maxsize=None)
def get_jobs_lookup():
    """Normalised Arabic job -> French job."""
    lookup = {}
    for arabic, french in jobs_data.items():
        if isinstance(french, list):
            french = french[0] if french else ""
        if isinstance(french, str) and french:
            lookup[ar_key(arabic)] = french
    return lookup


@lru_cache(maxsize=None)
def get_nationality_lookup():
    """Normalised Arabic nationality -> French nationality (first French variant, upper case)."""
    lookup = {}
    for entry in nationalities_data.values():
        if not isinstance(entry, dict):
            continue
        arabic_variants = entry.get("ar", [])
        french_variants = entry.get("fr", [])
        if not french_variants:
            continue
        for arabic in arabic_variants:
            lookup[ar_key(arabic)] = french_variants[0].upper()
    return lookup


@lru_cache(maxsize=None)
def get_places_lookup():
    """Normalised Arabic place name -> official French name (cities.json + places.json)."""
    return {ar_key(arabic): french for arabic, french in places_data.items()
            if isinstance(french, str) and french.strip()}


@lru_cache(maxsize=None)
def get_place_first_words():
    """First word of every known place key (to tell "بخريبڭة" = ب + a city from a word that starts with ب)."""
    return frozenset(key.split()[0] for key in get_places_lookup() if key)


def fuzzy_lookup(key, lookup, cutoff=0.85):
    """lookup[key], else the value of the closest key (typos: "ربة البيتت" -> "ربة البيت"), else None.

    Keys are ar_key() forms. A near match must start with the same letter, have a similar length and a
    difflib ratio >= cutoff, so that two different professions are never confused.
    """
    if key in lookup:
        return lookup[key]
    if not key:
        return None
    best, best_ratio = None, 0.0
    for candidate in lookup:
        if candidate[:1] != key[:1] or abs(len(candidate) - len(key)) > 3:
            continue
        ratio = SequenceMatcher(None, key, candidate).ratio()
        if ratio > best_ratio:
            best, best_ratio = candidate, ratio
    return lookup[best] if best is not None and best_ratio >= cutoff else None


def read_text_file(path):
    """Read the input file: UTF-16 (with its BOM), UTF-8, else Windows Arabic (cp1256).

    cp1256 decodes almost any bytes without error, so it must be the last try: a UTF-16 file tried with it
    would come back as garbage instead of failing. Line ends are always "\\n" (as in text mode), and a
    missing file raises FileNotFoundError as usual.
    """
    data = Path(path).read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = data.decode("utf-16")
    else:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = data.decode("cp1256")
    # same line ends as a file opened in text mode (the NER offsets are positions in this string)
    return text.replace("\r\n", "\n").replace("\r", "\n")


def read_json_file(path):
    """Read a JSON file written by another stage."""
    with open(path, encoding="utf-8-sig") as file:
        return json.load(file)


def write_json(obj, path=None):
    """Write a JSON object to a file (or to stdout when path is None)."""
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    if path is None:
        print(text)
    else:
        Path(path).write_text(text + "\n", encoding="utf-8")


def get_number_keys_source():
    """Every written number alias of date_time.json ("numbers" and "time"), as in the file."""
    aliases = []
    for section in ("numbers", "time"):
        for values in dates_data.get(section, {}).values():
            aliases.extend(a for a in values if isinstance(a, str))
    return aliases


def kraken_command():
    """Command that starts the kraken CLI with the python running this program.

    The kraken.exe launcher of a virtual environment stores the absolute path of the python it was installed with,
    so it breaks as soon as the project or the .venv folder is moved or renamed ("Fatal error in launcher"). Starting
    kraken through the current interpreter avoids it. Falls back to the plain `kraken` command if the module is missing.
    """
    try:
        import importlib.util
        if importlib.util.find_spec("kraken.kraken") is not None:
            return [sys.executable, "-c", "from kraken.kraken import cli; cli()"]
    except (ImportError, ValueError):
        pass
    return ["kraken"]

@lru_cache(maxsize=None)
def resolve_device(device):
    """The kraken device to use: a CUDA device that torch cannot use falls back to "cpu" with a warning.

    The torch installed by a plain `pip install torch` on Windows is CPU only: kraken then fails inside its
    segmentation step with no useful message. torch is only imported when a GPU is asked for.
    """
    device = (device or "cpu").strip()
    if not device.startswith("cuda"):
        return device
    try:
        import torch
    except ImportError:
        return device  # kraken will say what is wrong
    index = int(device.split(":", 1)[1]) if ":" in device and device.split(":", 1)[1].isdigit() else 0
    if torch.cuda.is_available() and index < torch.cuda.device_count():
        return device
    reason = ("this torch build has no CUDA support (torch " + torch.__version__ + ")"
              if not torch.backends.cuda.is_built() else "no CUDA GPU " + str(index) + " is visible")
    print(f"Warning: device {device!r} cannot be used, {reason}: kraken runs on the CPU.", file=sys.stderr)
    return "cpu"


def run_kraken(args, output, step):
    """Run the kraken CLI with these arguments and check that it wrote `output`.

    -r (--raise-on-error) makes kraken stop with its real error and a non-zero exit code: without it a failed
    image is only logged ("Failed processing ...: 1") and the command still ends normally.
    """
    import subprocess

    cmd = [*kraken_command(), "-r", *args]
    print("Running:", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)
    if not Path(output).is_file():
        raise RuntimeError(f"kraken {step} failed: no output written (see the kraken messages above)")