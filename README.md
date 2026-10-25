# MaDoc: Moroccan Administrative Documents transcription and structuring

MaDoc reads scanned Moroccan birth certificates (رسم ولادة) and turns them into structured, validated records.
A certificate goes in as an image; a JSON record comes out with the child's, the parents' and the declaration's
information in Arabic and in French, together with a report of everything that looks wrong or uncertain.

The certificates are handwritten or typed in Arabic on a fixed civil status form. Reading them reliably involves
several separate problems: locating the lines on the page, recognising Moroccan administrative handwriting,
coping with recognition errors, finding each field in a text whose layout varies from one office to another,
converting dates and times that are written out in words, and producing the official French spelling of names,
places and jobs. MaDoc handles each of these in its own stage, so every stage can be run, tested and inspected
separately.

## Contents

- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Using the web interface](#using-the-web-interface)
- [Using the command line](#using-the-command-line)
- [The output record](#the-output-record)
- [Stage details](#stage-details)
- [Dictionaries](#dictionaries)
- [Data protection](#data-protection)
- [Tests and continuous integration](#tests-and-continuous-integration)
- [Repository layout](#repository-layout)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)

## How it works

```
scan (jpg, png, tif)
  │
  │  segmentation.py      kraken baseline model     image   → PageXML (text lines, polygons, baselines)
  │  recognition.py       kraken HTR model          PageXML → Arabic text, one line per text line
  │  spelling_checker.py  local LLM (optional)      text    → text with recognition mistakes corrected
  │  classificator.py     MagBERT NER models        text    → named entities (persons, places, dates)
  │  parser.py            rules + NER entities      text    → record with the raw Arabic values
  │  converter.py         date_time.json            record  → dates, hours, minutes and sex as numbers
  │  mapper.py            jobs / nationalities      record  → French jobs and nationalities
  │  translator.py        Groq LLM                  record  → French names, birthplace, officer
  ▼  validator.py         checks                    record  → report of errors and warnings
record.json + report
```

The input decides where the chain starts. An image goes through every stage; a PageXML file that has already
been segmented starts at recognition; a `.txt` transcription starts at the spelling checker (when enabled) or
at the classificator. Each stage is also a command line tool that reads the output of the previous one, which
makes it easy to see which stage introduced a problem.

| Stage | Method | Runs on |
|---|---|---|
| Segmentation | kraken `blla` baseline segmentation | this computer |
| Recognition | kraken HTR model, then Unicode bidi reordering | this computer |
| Spelling correction | a local model through Ollama, filtered by dictionary rules | this computer |
| Named entities | `TypicaAI/MagBERT-NER-AR` (local) and `TypicaAI/MagBERT-NER-Fr` (Hugging Face Space) | partly online |
| Parsing | regular expressions on the fixed labels of the form, checked against the entities | this computer |
| Conversion and mapping | dictionary lookups | this computer |
| Translation | Groq API (`openai/gpt-oss-120b` by default) | online |
| Validation | field by field checks | this computer |

See [Data protection](#data-protection) before processing real certificates with the online stages enabled.

## Requirements

- Python 3.10 or newer. MaDoc is developed on Windows with Python 3.11; the CI runs on Linux with 3.10 and 3.12.
- Several GB of free disk space for PyTorch, kraken and the NER model (more with a CUDA build of PyTorch).
- Optional: an NVIDIA GPU with a CUDA build of PyTorch, to speed up kraken and the NER model.
- Optional: [Ollama](https://ollama.com) and about 8 GB of RAM for the spelling correction.
- Optional: a [Groq](https://console.groq.com) API key for the French translation of names.

## Installation

```bash
git clone https://github.com/NiemaAM/MaDoc.git
cd MaDoc
python -m venv .venv
.venv\Scripts\activate            # Linux / macOS: source .venv/bin/activate
pip install -r requirements.txt
```

The kraken models (`models/blla.mlmodel`, `models/htr.mlmodel`) and the dictionaries (`dictionaries/`) are part
of the repository. The Arabic NER model is downloaded from Hugging Face the first time it is used.

**GPU.** On Windows, `pip install torch` installs a CPU-only build. To use an NVIDIA GPU, install the CUDA build
from [pytorch.org](https://pytorch.org/get-started/locally/) and start the tools with `--device cuda:0`. If the
GPU cannot be used, MaDoc prints a warning and falls back to the CPU. To check your installation:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**Spelling correction (optional).** Install Ollama, then download the model once:

```bash
ollama pull qwen2.5:7b
```

**Groq (optional).** Create a `.env` file at the root of the project (it is listed in `.gitignore`):

```
GROQ_API_KEY=gsk_...
```

Do not move or rename the project folder after creating `.venv`. The launchers in `.venv\Scripts` store the
absolute path of the Python they were installed with. If the folder has moved, delete `.venv` and create it again.

## Configuration

All settings are environment variables. They can also be written in the `.env` file.

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | not set | Enables the translation stage. Without it, the French name fields keep the Arabic values. |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq model used for the translation. |
| `MADOC_SPELL_MODEL` | `qwen2.5:7b` | Ollama model used by the spelling checker. |
| `MADOC_OLLAMA_URL` | `http://127.0.0.1:11434` | Address of Ollama. Only this computer is accepted. |
| `MADOC_SPELL_ALLOW_REMOTE` | not set | Set to `1` to allow an Ollama server elsewhere on your own network. |
| `BIRTH_DICT_DIR` | `dictionaries/` | Folder containing the dictionaries, if they are kept somewhere else. |

## Using the web interface

demo link: https://5ea85740026fceb334.gradio.live

```bash
python src/app.py                                   # http://127.0.0.1:7860
python src/app.py --device cuda:0
python src/app.py --port 8080 --share               # public gradio.live link, see Data protection
python src/app.py --share --auth demo:password     # the same, with a login (or MADOC_AUTH=demo:password in .env)
python src/app.py --segmentation-model path/blla.mlmodel --recognition-model path/htr.mlmodel
```

Upload a scan (or paste it from the clipboard) and press **Transcribe**. Photos taken with a phone are turned
upright from their EXIF orientation before segmentation. The **Options** panel switches the NER models, the
Groq translation and the spelling correction on or off, and sets the confidence threshold under which a line is
flagged.

| Tab | Content |
|---|---|
| Image and lines | The scan with the polygon of every detected line, coloured by the recognition confidence (green, amber, red under the threshold), next to the list of lines. Hovering a line highlights its polygon and the other way round. |
| Transcription | The recognised text. It can be edited by hand; **Parse the corrected text again** reruns everything after recognition. |
| Spelling | The corrections proposed by the spelling checker, highlighted in the text. Click a correction to accept or refuse it, then **Apply and parse again**. See [Spelling correction](#spelling-correction). |
| JSON record | The structured record. |
| Validation | The validation report, followed by the spelling and parser notes. |
| Downloads | `record.json`, `transcription.txt` and `segmentation.xml`. |

The device and the model paths can only be set on the command line, so that visitors of a shared instance
cannot point kraken at other files on the server. Uploaded scans and the files produced by each run are
deleted after one hour.

## Using the command line

### Full pipeline

```bash
python src/pipeline.py scan.jpg                            # whole chain, record printed on stdout
python src/pipeline.py scan.jpg -o record.json --report report.json
python src/pipeline.py scan.xml --image scan.jpg           # start from an existing segmentation
python src/pipeline.py samples/1.txt                       # start from a transcription
python src/pipeline.py scan.jpg --spellcheck               # with the local spelling correction
python src/pipeline.py samples/1.txt --no-ner --no-translation   # fully offline, rules only
python src/pipeline.py scan.jpg --save-stages out/         # keep the output of every stage
```

| Option | Effect |
|---|---|
| `-o FILE` | Write the record to a file instead of stdout. |
| `--report FILE` | Also write the validation report as JSON. |
| `--save-stages DIR` | Write the output of every stage: `0a_segmentation.xml`, `0b_text.txt`, `0c_spellchecked.txt`, `1_ner.json`, `2_parsed.json`, `3_converted.json`, `4_mapped.json`, `5_translated.json`, `6_report.json`. |
| `--spellcheck` | Run the spelling checker between recognition and the NER stage. |
| `--no-ner` | Do not load the NER models; the parser works from the rules alone. |
| `--ner FILE` | Reuse a saved NER output instead of running the models. |
| `--no-translation` | Skip the Groq stage. |
| `--image FILE` | The scan of a PageXML input, when it is no longer where the XML says. |
| `--device` | kraken device: `cpu` (default), `cuda:0`, ... |
| `--segmentation-model`, `--recognition-model` | Use other kraken models. |
| `--strict` | Exit with code 1 when the report contains errors (useful in scripts). |

The record goes to stdout and every message (kraken progress, notes, the readable report) goes to stderr, so
`python src/pipeline.py scan.jpg > record.json` always produces a valid JSON file.

### Individual stages

```bash
python src/segmentation.py scan.jpg -o scan.xml
python src/recognition.py scan.xml -o scan.txt             # --raw keeps kraken's visual order
python src/spelling_checker.py scan.txt -o fixed.txt       # every change is listed on stderr
python src/classificator.py scan.txt -o ner.json
python src/parser.py scan.txt ner.json -o parsed.json --notes
python src/converter.py parsed.json -o converted.json
python src/mapper.py converted.json -o mapped.json
python src/translator.py mapped.json -o translated.json
python src/validator.py translated.json -o report.json     # exit code 1 when there is an error
```

## The output record

The record is a flat JSON object with a fixed set of fields. Arabic fields hold the text as written on the
certificate; fields ending in `FR` hold the French value. Example, from the anonymized sample `samples/1.txt`
(rules only, without translation):

```json
{
  "NUMACTE": "159",
  "DATENAISSFR": "2016-12-15",
  "DATENAISSAR": "1438-02-11",
  "HEURENAISS": 11,
  "MINNAISS": 0,
  "LIEUNAISS": "جماعة خريبڭة الحضرية اقليم خريبڭة",
  "PRENOMAR": "هشام",
  "PRENOM": "HICHAM",
  "NOMAR": "الإدريسي",
  "NOM": "IDRISSI",
  "SEXE": 1,
  "NATIONALITE": "مغربية",
  "NATIONALITEFR": "MAROCAINE",
  "PRENOMPEREAR": "نبيل الإدريسي",
  "NOMPEREAR": "بن عادل",
  "DATENAISSPEREFR": "1986-12-09",
  "PROFESSIONPERE": "أجير",
  "PROFESSIONPEREFR": "SALARIER",
  "PRENOMMEREAR": "ريم الشاوي",
  "NOMMEREAR": "بنت أنس",
  "PROFESSIONMERE": "ربة بيت",
  "PROFESSIONMEREFR": "FEMME AU FOYER",
  "ADRESSEPARENT": "بحي القدس الأول الرقم - 968 خريبڭة",
  "DATEDECLAFR": "2017-01-03",
  "DATEDECLAAR": "1438-03-01",
  "ANNEEDECLA": "2017",
  "EMPLOYECREANT": "فؤاد السعدي موظف جماعي",
  "BECNAME": "بالتفويض",
  "BECNAMEFR": "PAR DELEGATION"
}
```

(Some fields are left out above for brevity; the full list follows.)

| Group | Fields | Content |
|---|---|---|
| Act | `NUMACTE` | Act number. |
| Child | `PRENOMAR`, `PRENOM`, `NOMAR`, `NOM` | First and family name in Arabic, and in Latin letters as written on the certificate. |
| | `SEXE` | `1` male, `0` female. |
| | `DATENAISSFR`, `DATENAISSAR` | Gregorian and Hijri birth dates, `YYYY-MM-DD`. |
| | `HEURENAISS`, `MINNAISS` | Hour and minute of birth, integers. |
| | `LIEUNAISS`, `LIEUNAISSFR` | Place of birth. |
| | `NATIONALITE`, `NATIONALITEFR` | Nationality. |
| Father, mother | `PRENOMPEREAR`, `PRENOMMEREAR` (+ `FR`) | Everything written before `بن` / `بنت`: first name and family name. |
| | `NOMPEREAR`, `NOMMEREAR` (+ `FR`) | The lineage: `بن` / `بنت` and the grandfather's name, `Fils de` / `Fille de` in French. |
| | `DATENAISSPERE*`, `DATENAISSMERE*` | Birth date, or the year alone when the certificate gives only a year. |
| | `ADRESSENAISSPERE`, `ADRESSENAISSMERE` | Place of birth. |
| | `NATIONALITEPERE*`, `NATIONALITEMERE*`, `PROFESSIONPERE*`, `PROFESSIONMERE*` | Nationality and job, Arabic and French. |
| Declaration | `ADRESSEPARENT` | The parents' address. |
| | `TEXTEDECLA` | The declaration sentence (who declared the birth, age, job, address). |
| | `DATEDECLAFR`, `DATEDECLAAR`, `ANNEEDECLA` | Declaration dates and Gregorian year. |
| | `EMPLOYECREANT`, `EMPLOYECREANTFR` | The civil status officer who drew up the act. |
| | `BECNAME`, `BECNAMEFR` | Fixed: `بالتفويض` / `PAR DELEGATION`. |

## Stage details

### Segmentation and recognition

kraken finds the text lines with the baseline model and writes a PageXML file containing, for every line, its
polygon and baseline. The recognition model then reads every line. kraken writes Arabic in visual order (the
letters of a line come out reversed), so each line is passed through the Unicode bidi algorithm
(`python-bidi`) to get back the logical order that the rest of the pipeline expects. The web interface keeps the
confidence score of every line from the PageXML output.

kraken is started through the Python interpreter of the running program (not the `kraken.exe` launcher), and
with `--raise-on-error`, so that a failing image stops with kraken's real error message.

### Spelling correction

Recognition errors in the fixed words of the form (`موفق` for `موافق`, `ميلاي ية` for `ميلادية`) break the
rules of the parser, so the spelling checker is placed before it. It is optional and runs only on this computer.

The model receives the whole text and proposes a corrected version of every line. Its proposal is never applied
as it is. The two versions are compared word by word, and each change is judged separately:

- a change is accepted only if it replaces an unknown word with a word of the form, or with a number, month or
  time word from `date_time.json` (plus any word listed in `dictionaries/spelling_words.txt`);
- the original word must not already be a correct word, and at most one letter may differ (two for words of
  seven letters or more);
- the words that follow a name, place or address label (`الإسم الشخصي`, `من والده`, `بن`, `السيدة`, `جماعة`,
  `بحي`, `زنقة`, ...) are never changed, even when the label itself is misread or sits on the previous line;
- digits, Latin text and fragments of one or two letters are never changed;
- differences the parser does not see are ignored: punctuation, the space after a prefix (`ب جماعة` and
  `بجماعة`), `ڭ` and `ك`, and hamza on alif;
- an hour or minute may differ by up to three letters when the corrected phrase becomes one of the time
  dictionary (`التانس عشرة` → `التاسعة عشرة`).

Jobs are corrected towards `jobs.json` only. A misread job receives every job of the dictionary at the smallest
distance, with the model's choice listed first. When there is a single candidate it is applied; when there are
several (`صباخ` may be `صباغ`, painter, or `طباخ`, cook), nothing is applied until someone chooses. Jobs are
also read in context: when the declarant is a parent (`(والدها)`), the job written in the declaration is the
same person's job written a second time, and only the jobs that fit both readings are kept. Two readings that
agree are left as they are, since they most likely show a job that is missing from the dictionary.

In the **Spelling** tab of the web interface, applied changes are shown in green and the others in red, with
the proposed spelling as a badge. Clicking a change accepts or refuses it; for a job with several candidates,
each click moves to the next candidate. **Back to the rules' choice** resets every decision and **Apply and
parse again** rebuilds the record. The notes record which decisions were taken in the tab.

The job check also works when Ollama is not running, because it only needs the dictionary.

### Named entity recognition

Two MagBERT models are run on the text: the Arabic model finds persons, places and dates with their position
in the text, and the French model reads the Latin names. The parser uses these entities as a second, independent
reading of every field (see below). The stage is optional: with `--no-ner`, or when a model cannot be loaded,
the parser relies on its rules alone.

### Parsing

The parser cuts the text into four zones using the fixed labels of the form: the child (from `في يوم` to
`من والده`), the father (to `و والدته`), the mother (to `الساكنان`) and the administration (the address,
the declaration and the signature). Each field is then read within its zone. The rules tolerate the variants
found on real certificates: labels split over two lines (`والد` / `ها`), common misspellings, a nationality or
job printed in the wrong block, the child's Latin names printed inside a parent's block, and so on.

When NER entities are available, every field is also read from the entities of its zone, without using any
label. The two readings are then compared:

- if the rules found nothing, the NER value is used;
- for a date, if only one of the two converts to a valid date of the right calendar, that one is kept;
- if the rules swallowed a label word into a name or a place, the NER value is used;
- in every other disagreement the rules are kept and the difference is reported in the notes.

### Conversion and mapping

Dates are written out in words on the certificates (`ثالث و عشري شوال عام ألف و أربعمائة و أربعة و عشرين`).
The converter recognises the day, month and year phrases with `date_time.json`, in every spelling seen in the
documents, and produces `YYYY-MM-DD` (or `YYYY` when only a year is given). Hours and minutes are matched as
whole phrases against the time section of the dictionary. The sex becomes `1` or `0`.

The mapper looks up the jobs in `jobs.json` and the nationalities in `nationalities.json`, with tolerance for
small spelling mistakes. A job that is not in the dictionary is left for the translation stage.

### Translation

One Groq request per record translates the parents' names, the place of birth, the civil status officer and
any job missing from the dictionary. The model is constrained in several ways:

- official place names found in `cities.json` are imposed through a glossary;
- the child's family name in Latin letters is sent with the father's name, so both are spelled the same way;
- `بن` and `بنت` are always rendered as `Fils de` and `Fille de`;
- the officer's title is fixed (`FONCTIONAIRE ET OFFECIER DETAT CIVIL`);
- any answer that still contains Arabic script is rejected and the original value is kept.

Without `GROQ_API_KEY` this stage is skipped and the report marks the French fields as untranslated.

### Validation

The validator checks the structure and content of the record and returns a list of issues, each with a level:

- **error**: the record is wrong and must be corrected. This covers missing required fields, values of the
  wrong type or out of range (hour 0-23, minute 0-59, sex 0/1), invalid dates, a declaration dated before the
  birth, a parent born after the child, Latin letters in an Arabic field or Arabic in a French field, label
  words caught inside a value (a sign that a rule captured too much), and modified fixed values.
- **warning**: the value is plausible but should be reviewed. Examples: a Gregorian and Hijri year that do not
  match (registry staff sometimes enter them that way on purpose), a parent younger than 14 or older than 80 at
  the birth, a declaration made more than a year after the birth, a job or nationality missing from the
  dictionaries, or an unusually long value.

A record is valid when it has no error.

## Dictionaries

All dictionaries are JSON files in `dictionaries/` and can be edited by hand. Every lookup ignores diacritics,
hamza and the article `ال`, so a single entry covers the usual spelling variants.

| File | Format | Used for |
|---|---|---|
| `date_time.json` | `hijri_months`, `miladi_months`, `numbers`, `time`: value → list of written forms | Converting dates, hours and minutes. The spelling checker also uses it as vocabulary. |
| `jobs.json` | `"Arabic job": "FRENCH JOB"` | French jobs; the spelling checker's job candidates. |
| `nationalities.json` | `{"ar": [...], "fr": [...]}` per nationality | French nationalities. |
| `cities.json` | list of `{"name": French, "nameAr": Arabic, ...}` | Official French spelling of Moroccan places. |
| `places.json` (optional) | `"Arabic": "French"` | Extra places or corrections; overrides `cities.json`. |
| `spelling_words.txt` (optional) | words separated by spaces, `#` for comments | Extra words the spelling checker may correct towards. |

Adding a job that appears in your documents, for example `"حداد": "FORGERON"`, improves both the French
mapping and the spelling correction.

## Data protection

Birth certificates are personal data. The table shows what leaves the computer at each stage.

| Stage | Data sent | Destination | How to keep it local |
|---|---|---|---|
| Segmentation, recognition, spelling, parsing, conversion, mapping, validation | nothing | | |
| French NER model | the whole transcription | Hugging Face Space `TypicaAI/MagBERT-NER-Fr` | untick "NER models" or use `--no-ner` |
| Translation | parents' names, place of birth, officer's name, unknown jobs | Groq API | untick "Groq translation" or use `--no-translation` |

The spelling checker refuses any Ollama address other than this computer unless `MADOC_SPELL_ALLOW_REMOTE=1`
is set. Starting the web interface with `--share` makes it reachable from the internet through gradio.live; do
not use it with real documents.

The web interface deletes uploaded scans and the files of each run after one hour. Nothing is written to the
project folder.

The files in `samples/` are anonymized: names, act and house numbers were replaced and every year was shifted,
Gregorian and Hijri separately so the pairs stay consistent. Do not commit real scans or transcriptions;
`.gitignore` already excludes `*.jpg`, `*.xml` and `.env`.

## Tests and continuous integration

```bash
pip install -r requirements-ci.txt      # pytest, python-dotenv, python-bidi, pillow
python -m pytest -q
```

The tests need neither the models nor a network connection. kraken, Groq and Ollama are replaced by stand-ins
(the Ollama stand-in is a small local HTTP server that speaks the same API).

| File | What it covers |
|---|---|
| `tests/test_samples.py` | Every text in `samples/` goes through parser, converter, mapper and validator without any error. Add a `.txt` file to `samples/` to add a case. |
| `tests/test_units.py` | File encodings, kraken calls and error handling, GPU fallback, PageXML escaping, translator output, the web interface helpers. |
| `tests/test_spelling.py` | The rules of the spelling checker: accepted and refused changes, protected values, time phrases, job candidates and their context, the Spelling tab. |

GitHub Actions runs the same command on Python 3.10 and 3.12 for every push and pull request
(`.github/workflows/tests.yml`). The tests of the web interface are skipped when gradio is not installed, which
is the case in CI.

## Repository layout

```
MaDoc/
├── src/
│   ├── app.py               web interface (Gradio)
│   ├── pipeline.py          runs the whole chain
│   ├── segmentation.py      stage 0a: image → PageXML
│   ├── recognition.py       stage 0b: PageXML → text
│   ├── spelling_checker.py  stage 0c: local spelling correction
│   ├── classificator.py     stage 1: named entities
│   ├── parser.py            stage 2: text → record
│   ├── converter.py         stage 3: dates, time, sex
│   ├── mapper.py            stage 4: jobs, nationalities
│   ├── translator.py        stage 5: French names and places
│   ├── validator.py         stage 6: report
│   └── common.py            dictionaries, record template, Arabic normalisation, helpers
├── dictionaries/            date_time.json, jobs.json, nationalities.json, cities.json
├── models/                  blla.mlmodel (segmentation), htr.mlmodel (recognition)
├── samples/                 anonymized transcriptions used by the tests
├── tests/                   pytest suite
├── .github/workflows/       CI configuration
├── requirements.txt         full installation
└── requirements-ci.txt      what the tests need
```

## Known limitations

- The parser is built for the birth certificate form used by Moroccan civil status offices. Other documents
  (marriage, death certificates) would need their own zones and rules.
- A name misread by the recognition model is not corrected: the spelling checker deliberately leaves names
  alone, and the reading has to be fixed in the Transcription tab.
- The quality of the French transliteration of names depends on the Groq model. Place names and the father's
  family name are constrained; first names and grandfathers' names are not.
- The spelling checker with a 7B model takes a few minutes per certificate on a CPU.
- Hijri dates are only range-checked, not converted, so a Hijri day that does not exist in a given month is not
  detected.

## Troubleshooting

**`Fatal error in launcher: Unable to create process`.** The `.venv` was created in another folder. Delete
`.venv`, create it again and reinstall the requirements.

**`kraken segmentation failed: no output written`.** Read the kraken message printed just above it. The most
common cause used to be `--device cuda:0` with a CPU-only PyTorch; MaDoc falls back to the CPU in that case
and prints a warning.

**`dictionaries not found`.** Run the commands from the project root, or set `BIRTH_DICT_DIR` to the folder
that contains `date_time.json`.

**`spelling model skipped: Ollama not reachable`.** Start Ollama and check that the model is installed
(`ollama list`). The job check still runs without it.

**`Groq client unavailable`.** `GROQ_API_KEY` is missing from the environment or from `.env`. The record is
still produced, with the Arabic values in the French name fields.

**A field is wrong or empty.** Run the pipeline with `--save-stages out/` and look at the files in order:
`0b_text.txt` (what was read), `2_parsed.json` (what the rules found) and the notes printed on stderr. Most
problems come from a recognition error in a label, which can be corrected in the Transcription tab or by the
spelling checker.