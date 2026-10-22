# MaDoc: Moroccan Administrative Documents transcription and structuring

MaDoc turns a scan of a Moroccan Arabic birth certificate (رسم ولادة) into a validated, structured JSON record.
It combines handwritten text recognition (kraken), named entity recognition (MagBERT), a rule based parser,
dictionaries for dates, numbers, jobs and places, an LLM for the French transliteration, and a validator that
checks the result.

```
scan.jpg
  │  segmentation.py   kraken blla      image  → PageXML (text lines, polygons)
  │  recognition.py    kraken HTR       PageXML → Arabic text (bidi order restored)
  │  classificator.py  MagBERT NER      text   → entities (Arabic + French models)
  │  parser.py         rules + NER      text   → record (zones: child / father / mother / administration)
  │  converter.py      dictionaries     Arabic words → numbers and dates (day, month, year, hour, minute)
  │  mapper.py         dictionaries     jobs, nationalities, official place names (cities.json)
  │  translator.py     Groq LLM         names, addresses, unknown jobs → French
  ▼  validator.py       checks           errors and warnings per field
record.json + report
```

Each stage can be run on its own, and the input decides where the chain starts: an image runs everything,
a PageXML starts at recognition, a `.txt` starts at the classificator.

## Installation

Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows      (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

Do not rename or move the project folder after creating the `.venv`: the launchers inside it (`kraken.exe`,
`pip.exe` ...) store absolute paths. If you did, delete `.venv` and create it again.

### What is not in the repository

| What | Where to put it |
|---|---|
| kraken segmentation model | `models/blla.mlmodel` |
| kraken recognition (HTR) model | `models/htr.mlmodel` |
| dictionaries `date_time.json`, `jobs.json`, `nationalities.json`, `cities.json` (and optionally `places.json`) | `dictionaries/` (or the folder in `BIRTH_DICT_DIR`) |
| Groq API key | `GROQ_API_KEY` in the environment or in a `.env` file |

The NER models are downloaded the first time they are used (`TypicaAI/MagBERT-NER-AR` from Hugging Face; the French
model `TypicaAI/MagBERT-NER-Fr` is called through `gradio_client`). Without a Groq key the translation stage
is skipped and the original Arabic values are kept.

## Usage

### Web interface

```bash
python src/app.py                 # http://127.0.0.1:7860
python src/app.py --device cuda:0 --share
```

Upload a scan and press Transcribe. The page shows the image with the detected lines as polygons (with the
confidence score of each line, in the style of eScriptorium), the transcription, and the JSON record with its
validation report.

### Command line

```bash
python src/pipeline.py scan.jpg                          # whole chain, record on stdout
python src/pipeline.py scan.jpg -o record.json --report report.json
python src/pipeline.py scan.xml --image scan.jpg         # the scan is already segmented
python src/pipeline.py samples_anonymized/anon_1.txt     # start from the text
python src/pipeline.py text.txt --no-ner                 # rules only, no model is loaded
python src/pipeline.py text.txt --no-translation         # no Groq call
python src/pipeline.py scan.jpg --save-stages out/       # the output of every stage, to see what changed what
python src/pipeline.py scan.jpg --strict                 # exit code 1 if the report has errors
```

The individual stages:

```bash
python src/segmentation.py scan.jpg scan.xml
python src/recognition.py scan.xml scan.txt              # --raw keeps kraken's visual order
python src/classificator.py scan.txt -o ner.json
```

## The record

The output is one flat JSON object (arabic fields, french fields, dates as `YYYY-MM-DD`, hour and minute as
integers). A few rules worth knowing:

- Dates and times are read from the written words (`ثالث و عشري شوال عام ألف و أربعمائة و أربعة و عشرين`) with
  the dictionaries, in every spelling variant seen in the documents.
- Place names are never transliterated by the LLM: they come from `cities.json` (official spelling).
- A father's surname is the child's surname sent along with the request, so the two always match.
- Jobs are looked up in `jobs.json`; a job that is not found is sent to the LLM and upper-cased.
- Gregorian and Hijri years that do not match are a **warning**, not an error: municipality staff sometimes
  enter them like that on purpose.

## Validation

`validator.py` checks every field of the record (types and ranges, date order, ages, script of each field,
fixed values, labels leaking into values, dictionary coverage) and returns a report of errors and warnings.
A record is valid when it has no error.

## Tests

```bash
pip install -r requirements.txt
python -m pytest src/test.py
```

The tests run the whole offline chain (no kraken models, no NER models, no network): the sample texts are
parsed, converted, mapped and validated, kraken and Groq are replaced by stand-ins. The same command runs in
GitHub Actions (`.github/workflows/tests.yml`) on Python 3.10 and 3.12.

## Repository layout

```
src/                 the pipeline stages and the web interface
dictionaries/        date_time.json, jobs.json, nationalities.json, cities.json ...
samples/             anonymized testing samples
models/              kraken models (not committed)
tests/               pytest suite
```