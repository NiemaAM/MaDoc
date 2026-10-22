"""Stage 1 - classificator: txt file -> JSON object with the entities found by the two NER models.

    python src/classificator.py samples/6.txt                 # prints the JSON
    python src/classificator.py samples/6.txt -o ner.json     # writes it to a file

Output:
    {
      "models": {"ar": "...", "fr": "..."},
      "arabic_ner": [{"entity": "خريبڭة", "label": "GPE", "start": 120, "end": 126, "score": 0.99}, ...],
      (entity = exact text of the document; "word" = the model's own string, only when it differs)
      "french_ner": [{"entity": "ZAKARIYA", "label": "PERSON"}, ...]
      ("errors": [...] only when a model could not be used)
    }

"start" / "end" are character offsets in the text exactly as read from the file, the parser uses them to
know in which part of the document (child, father, mother, administration) every entity is.
The models are loaded the first time they are needed, so importing this module is cheap.
"""

import argparse
import re
import sys
from pathlib import Path

from common import read_json_file, read_text_file, write_json  # noqa: F401  (read_json_file re-exported)

# Connect models
MODEL_AR = "TypicaAI/MagBERT-NER-AR"
MODEL_FR = "TypicaAI/MagBERT-NER-Fr"
# BERT models read at most 512 tokens: longer texts are classified in windows of this many characters
MAX_CHARS = 1000

_arabic_pipeline = None
_french_client = None


# initialize models
def get_arabic_pipeline():
    """Load the Arabic NER model once (needs torch + transformers)."""
    global _arabic_pipeline
    if _arabic_pipeline is None:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_AR)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_AR)
        model = model.to(device).eval()
        _arabic_pipeline = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="first",
            device=0 if device == "cuda" else -1,
        )
    return _arabic_pipeline


def get_french_client():
    """Connect to the French NER space once (needs gradio_client)."""
    global _french_client
    if _french_client is None:
        from gradio_client import Client

        _french_client = Client(MODEL_FR)
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


# NLP labeling AR
def classify_arabic(text):
    """Run Arabic NER, offsets are relative to the whole text."""
    ner = get_arabic_pipeline()
    arabic_ner = []
    for offset, chunk in chunk_text(text):
        if not chunk.strip():
            continue
        for item in ner(chunk):
            start, end = int(item["start"]), int(item["end"])
            # Convert Arabic results to JSON-compatible values
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
            arabic_ner.append(entity)
    return arabic_ner


# NLP labeling FR
def classify_french(text):
    """Run the French NER space."""
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


def classify(text, use_arabic=True, use_french=True):
    """Run both models. A model that cannot be loaded / reached is reported in "errors", not raised."""
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
        except Exception as e:  # missing package, no network, space asleep ...
            errors.append(f"{key}: {type(e).__name__}: {e}")
    if errors:
        result["errors"] = errors
    return result


def classify_file(path, **options):
    """txt file -> NER JSON object."""
    return classify(read_text_file(path), **options)


def main():
    parser = argparse.ArgumentParser(description="Run the Arabic and French NER models on a text file.")
    parser.add_argument("input_file", type=Path, help="Path to the input .txt file")
    parser.add_argument("-o", "--output", type=Path, help="write the JSON here instead of stdout")
    parser.add_argument("--no-french", action="store_true", help="skip the French model")
    parser.add_argument("--no-arabic", action="store_true", help="skip the Arabic model")
    args = parser.parse_args()
    if not args.input_file.is_file():
        parser.error(f"Input file not found: {args.input_file}")
    result = classify_file(args.input_file, use_arabic=not args.no_arabic, use_french=not args.no_french)
    for error in result.get("errors", []):
        print(f"Warning: {error}", file=sys.stderr)
    write_json(result, args.output)


if __name__ == "__main__":
    main()
