"""Stage 1 - classificator: txt file -> JSON object with the entities found by the two NER models.

    python src/classificator.py samples/6.txt                 # prints the JSON
    python src/classificator.py samples/6.txt -o ner.json     # writes it to a file
    python src/classificator.py --download                    # downloads / checks both models, then exits

Output:
    {
      "models": {"ar": "...", "fr": "..."},
      "arabic_ner": [{"entity": "خريبڭة", "label": "GPE", "start": 120, "end": 126, "score": 0.99}, ...],
      (entity = exact text of the document; "word" = the model's own string, only when it differs)
      "french_ner": [{"entity": "ZAKARIYA", "label": "PERSON", "start": 210, "end": 218, "score": 0.98}, ...]
      ("errors": [...] only when a model could not be used)
    }

"start" / "end" are character offsets in the text exactly as read from the file, the parser uses them to
know in which part of the document (child, father, mother, administration) every entity is.

Both models run on this computer with transformers (nothing is sent online):
  - Arabic: TypicaAI/MagBERT-NER-AR, on the whole text,
  - French: TypicaAI/magbert-ner (the model behind the Hugging Face Space TypicaAI/MagBERT-NER-Fr), only on the
    lines written in Latin letters ("PRENOM ZAKARIYA", "NOM KABLI"): the rest of a certificate is Arabic.
The French model is gated on Hugging Face: accept its conditions once and log in with a token (README), then it
is kept in the Hugging Face cache; HF_HUB_OFFLINE=1 (environment or .env) forbids any further download.
The online Space is used only when MADOC_FRENCH_NER_SPACE=1 is set and the local model cannot be loaded.
The models are loaded the first time they are needed, so importing this module is cheap.
"""

import argparse
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from common import read_json_file, read_text_file, write_json  # noqa: F401  (read_json_file re-exported)

# .env may hold HF_TOKEN / HF_HUB_OFFLINE, which must be set before transformers is imported
load_dotenv()

# Connect models
MODEL_AR = "TypicaAI/MagBERT-NER-AR"
MODEL_FR = "TypicaAI/magbert-ner"
SPACE_FR = "TypicaAI/MagBERT-NER-Fr"   # the online demo of MODEL_FR, only with MADOC_FRENCH_NER_SPACE=1
# BERT models read at most 512 tokens: longer texts are classified in windows of this many characters
MAX_CHARS = 1000
# a line is sent to the French model when it has at least this many Latin letters
MIN_LATIN_LETTERS = 2

_pipelines = {}
_french_client = None


class ModelUnavailable(RuntimeError):
    """A model that cannot be loaded, with what to do about it."""


# initialize models
def get_pipeline(model_id):
    """Load a NER model of this computer once (needs torch + transformers), kept for the next documents."""
    if model_id not in _pipelines:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForTokenClassification.from_pretrained(model_id)
        except OSError as e:   # gated model not accepted, no token, offline without the files ...
            raise ModelUnavailable(
                f"{model_id} cannot be loaded ({e}). If it is the first time: accept its conditions on "
                f"https://huggingface.co/{model_id}, log in with a Hugging Face token, then run "
                "python src/classificator.py --download (see the README)."
            ) from e
        model = model.to(device).eval()
        _pipelines[model_id] = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="first",
            device=0 if device == "cuda" else -1,
        )
    return _pipelines[model_id]


def get_arabic_pipeline():
    """Load the Arabic NER model once."""
    return get_pipeline(MODEL_AR)


def get_french_pipeline():
    """Load the French NER model once."""
    return get_pipeline(MODEL_FR)


def get_french_client():
    """Connect to the French NER Space once (needs gradio_client): only with MADOC_FRENCH_NER_SPACE=1."""
    global _french_client
    if _french_client is None:
        from gradio_client import Client

        _french_client = Client(SPACE_FR)
    return _french_client


def chunk_text(text, max_chars=MAX_CHARS):
    """Split text into [(offset, chunk)] at line / word boundaries. "".join(chunks) == text."""
    chunks = []
    current, current_start, position = "", 0, 0
    for line in text.splitlines(keepends=True):
        # a very long line is cut at word boundaries
        pieces = [line] if len(line) <= max_chars else re.findall(r"\S+\s*|\s+", line)
        for piece in pieces:
            if current and len(current) + len(piece) > max_chars:
                chunks.append((current_start, current))
                current, current_start = "", position
            if not current:
                current_start = position
            current += piece
            position += len(piece)
    if current:
        chunks.append((current_start, current))
    return chunks


def latin_lines(text):
    """[(offset, line)] of the lines written in Latin letters (at least MIN_LATIN_LETTERS of them)."""
    out, position = [], 0
    for line in text.splitlines(keepends=True):
        if len(re.findall(r"[A-Za-zÀ-ÿ]", line)) >= MIN_LATIN_LETTERS:
            out.append((position, line.rstrip("\r\n")))
        position += len(line)
    return out


def entities(ner, offset, chunk):
    """The pipeline's entities of one piece of text, with offsets in the whole text."""
    found = []
    for item in ner(chunk):
        start, end = int(item["start"]), int(item["end"])
        entity = {
            # the exact text of the document: the model's "word" is rebuilt from word pieces and can
            # differ from it ("ميلادية" -> "ميلاي")
            "entity": chunk[start:end].strip() or item["word"],
            "label": item["entity_group"],
            "start": offset + start,
            "end": offset + end,
            "score": round(float(item.get("score", 0.0)), 4),
        }
        if item["word"] != entity["entity"]:
            entity["word"] = item["word"]
        found.append(entity)
    return found


# NLP labeling AR
def classify_arabic(text):
    """Run Arabic NER, offsets are relative to the whole text."""
    ner = get_arabic_pipeline()
    arabic_ner = []
    for offset, chunk in chunk_text(text):
        if chunk.strip():
            arabic_ner.extend(entities(ner, offset, chunk))
    return arabic_ner


# NLP labeling FR
def classify_french(text):
    """Run the French NER model of this computer on the Latin lines; offsets relative to the whole text.

    The online Space is used instead only when MADOC_FRENCH_NER_SPACE=1 and the local model is missing."""
    lines = latin_lines(text)
    if not lines:
        return []
    try:
        ner = get_french_pipeline()
    except (ModelUnavailable, ImportError):
        if os.environ.get("MADOC_FRENCH_NER_SPACE") == "1":
            return classify_french_space(text)
        raise
    french_ner = []
    for offset, line in lines:
        french_ner.extend(french_line_entities(ner, offset, line))
    return french_ner


# the labels of the Latin lines: never a name themselves
FRENCH_LABEL = re.compile(r"^(\s*)(PR[EÉ]NOMS?|NOMS?)\b[\s:.\-]*", re.IGNORECASE)
LABEL_WORDS = {"PRENOM", "PRÉNOM", "PRENOMS", "PRÉNOMS", "NOM", "NOMS"}
# the model reads names in a sentence better than a lone word in capitals
CONTEXT = {"pr": "Son prénom est {}.", "no": "Son nom est {}."}


def french_queries(line):
    """[(query, shift)]: texts given to the French model, in order; query[i + shift] == line[i] for the value.

    1. the line as it is, 2. the line in title case (same length), 3. the value alone in title case inside a
    short French sentence ("Son prénom est Hicham.")."""
    queries = [(line, 0)]
    title = line.title()
    if len(title) == len(line) and title != line:
        queries.append((title, 0))
    m = FRENCH_LABEL.match(line)
    value = line[m.end():].rstrip() if m else ""
    if value and len(value.title()) == len(value):
        template = CONTEXT["pr" if m.group(2).upper().startswith("PR") else "no"]
        prefix = template.split("{}")[0]
        queries.append((template.format(value.title()), len(prefix) - m.end()))
    return queries


def french_line_entities(ner, offset, line):
    """The entities of one Latin line, from the first query where the model finds a name (not a label)."""
    for query, shift in french_queries(line):
        found = []
        for item in ner(query):
            start = max(0, int(item["start"]) - shift)
            end = min(len(line), int(item["end"]) - shift)
            text = line[start:end].strip() if end > start else ""
            if not text or text.upper() in LABEL_WORDS:
                continue
            start += len(line[start:end]) - len(line[start:end].lstrip())   # offsets of the stripped text
            found.append({
                "entity": text,
                "label": item["entity_group"],
                "start": offset + start,
                "end": offset + start + len(text),
                "score": round(float(item.get("score", 0.0)), 4),
            })
        if found:
            return found
    return []


def classify_french_space(text):
    """The French NER Space (online: the text is sent to Hugging Face). Only with MADOC_FRENCH_NER_SPACE=1."""
    res_fr = get_french_client().predict(text=text, api_name="/predict")
    # Keep French results that have a class or label
    french_ner = []
    for item in res_fr:
        label = item.get("class_or_confidence")
        if label is not None:
            french_ner.append({
                "entity": item.get("token", "").strip(),
                "label": label,
            })
    return french_ner


_results = {}          # (text, use_arabic, use_french) -> result, the last few documents
CACHE_SIZE = 16


def classify(text, use_arabic=True, use_french=True):
    """Run both models. A model that cannot be loaded is reported in "errors", not raised.

    The result of a text is kept (CACHE_SIZE texts): the web interface parses the same text again after an
    undo, a redo or a spelling choice, and the models are the slow part. A result with errors is not kept (the
    model may be downloaded meanwhile)."""
    import copy

    key = (text, use_arabic, use_french)
    if key in _results:
        return copy.deepcopy(_results[key])
    result = _classify(text, use_arabic, use_french)
    if "errors" not in result:
        if len(_results) >= CACHE_SIZE:
            _results.pop(next(iter(_results)))
        _results[key] = copy.deepcopy(result)
    return result


def _classify(text, use_arabic, use_french):
    result = {
        "models": {"ar": MODEL_AR, "fr": MODEL_FR},
        "arabic_ner": [],
        "french_ner": [],
    }
    errors = []
    for enabled, key, function in (
        (use_arabic, "arabic_ner", classify_arabic),
        (use_french, "french_ner", classify_french),
    ):
        if not enabled:
            continue
        try:
            result[key] = function(text)
        except Exception as e:  # missing package, model not downloaded / not accepted ...
            errors.append(f"{key}: {type(e).__name__}: {e}")
    if errors:
        result["errors"] = errors
    return result


def classify_file(path, **options):
    """txt file -> NER JSON object."""
    return classify(read_text_file(path), **options)


def download():
    """Download (or check) both models into the Hugging Face cache; True when both can be loaded."""
    ok = True
    for model_id in (MODEL_AR, MODEL_FR):
        try:
            get_pipeline(model_id)
            print(f"{model_id}: ready", file=sys.stderr)
        except Exception as e:
            print(f"{model_id}: NOT available: {e}", file=sys.stderr)
            ok = False
    return ok


def main():
    parser = argparse.ArgumentParser(description="Run the Arabic and French NER models on a text file.")
    parser.add_argument("input_file", type=Path, nargs="?", help="Path to the input .txt file")
    parser.add_argument("-o", "--output", type=Path, help="write the JSON here instead of stdout")
    parser.add_argument("--no-french", action="store_true", help="skip the French model")
    parser.add_argument("--no-arabic", action="store_true", help="skip the Arabic model")
    parser.add_argument("--download", action="store_true", help="download / check both models, then exit")
    args = parser.parse_args()
    if args.download:
        sys.exit(0 if download() else 1)
    if not args.input_file or not args.input_file.is_file():
        parser.error(f"Input file not found: {args.input_file}")
    result = classify_file(args.input_file, use_arabic=not args.no_arabic, use_french=not args.no_french)
    for error in result.get("errors", []):
        print(f"Warning: {error}", file=sys.stderr)
    write_json(result, args.output)


if __name__ == "__main__":
    main()