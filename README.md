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

Every stage runs on the computer that processes the certificates. No document is sent to an online service (DIL).

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
  │  spelling_checker.py  local LLM + rules         text    → text with recognition mistakes corrected
  │  classificator.py     MagBERT NER models        text    → named entities (persons, places, dates)
  │  parser.py            rules + NER entities      text    → record with the raw Arabic values
  │  converter.py         date_time.json            record  → dates, hours, minutes and sex as numbers
  │  mapper.py            jobs / nationalities      record  → French jobs and nationalities
  │  translator.py        local LLM + dictionaries  record  → French names, birthplace, officer
  ▼  validator.py         checks                    record  → report of errors and warnings
record.json + report
```

The input decides where the chain starts. An image goes through every stage; a PageXML file that has already
been segmented starts at recognition; a `.txt` transcription starts at the spelling checker (when enabled) or
at the classificator. Each stage is also a command line tool that reads the output of the previous one, which
makes it easy to see which stage introduced a problem.

| Stage | Method |
|---|---|
| Segmentation | kraken `blla` baseline segmentation |
| Recognition | kraken HTR model, then Unicode bidi reordering |
| Spelling correction | a local model through Ollama (`gemma3:12b` by default), filtered by dictionary rules and checks of the whole certificate |
| Named entities | `TypicaAI/MagBERT-NER-AR` (Arabic) and `TypicaAI/magbert-ner` (French), with transformers |
| Parsing | regular expressions on the fixed labels of the form, checked against the entities |
| Conversion and mapping | dictionary lookups |
| Translation | dictionaries first (`names.json`, `cities.json`), the local model for the remaining words |
| Validation | field by field checks |

## Requirements

- Python 3.10 or newer. MaDoc is developed on Windows with Python 3.11 and runs on Linux; the CI uses Python
  3.10 and 3.12.
- Several GB of free disk space for PyTorch, kraken and the two NER models (about 1 GB for the models).
- Optional: an NVIDIA GPU with a CUDA build of PyTorch, to speed up kraken and the NER models.
- [Ollama](https://ollama.com) for the spelling correction and the French translation of names (about 8 GB of
  RAM or of GPU memory for a 7B model). Without it, both stages are skipped and the rest still works.
- A free [Hugging Face](https://huggingface.co) account, once, to download the French NER model (it is gated).

## Installation

```bash
git clone https://github.com/NiemaAM/madoc.git
cd madoc
python -m venv .venv
.venv\Scripts\activate            # Linux / macOS: source .venv/bin/activate
pip install -r requirements.txt
```

The kraken models (`models/blla.mlmodel`, `models/htr.mlmodel`) and the dictionaries (`dictionaries/`) are part
of the repository.

**NER models.** Both models are downloaded from Hugging Face once, then run on this computer. The French model
`TypicaAI/magbert-ner` is gated: open [its page](https://huggingface.co/TypicaAI/magbert-ner) while logged in and
accept the conditions, create a *Read* token in Settings → Access Tokens, then:

```bash
hf auth login                               # paste the token (or put HF_TOKEN=hf_... in .env)
python src/classificator.py --download      # downloads both models, prints "ready" for each
```

Once both are ready, `HF_HUB_OFFLINE=1` in `.env` makes sure nothing is downloaded or checked online again.

**Local language model.** Install Ollama, then download the model once:

```bash
ollama pull gemma3:12b
```

The same model is used for the spelling correction and for the translation. Another model can be chosen in
`.env` (see [Configuration](#configuration)), for example `gemma3:12b` or `qwen2.5:14b` on a GPU with enough
memory.

**GPU.** On Windows, `pip install torch` installs a CPU-only build. To use an NVIDIA GPU, reinstall the same
version of PyTorch from its CUDA index (see [pytorch.org](https://pytorch.org/get-started/locally/)) and start
the tools with `--device cuda:0`. If the GPU cannot be used, MaDoc prints a warning and falls back to the CPU.
The NER models use the GPU automatically when PyTorch sees one, and Ollama does too (`ollama ps` shows where the
model runs). To check your installation:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Do not move or rename the project folder after creating `.venv`. The launchers in `.venv\Scripts` store the
absolute path of the Python they were installed with. If the folder has moved, delete `.venv` and create it again.

## Configuration

All settings are environment variables. They can also be written in a `.env` file at the root of the project,
which is read by every tool (it is listed in `.gitignore`).

| Variable | Default | Purpose |
|---|---|---|
| `MADOC_SPELL_MODEL` | `gemma3:12b` | Ollama model used by the spelling checker. |
| `MADOC_TRANSLATE_MODEL` | the value of `MADOC_SPELL_MODEL` | Ollama model used for the translation. |
| `MADOC_OLLAMA_URL` | `http://127.0.0.1:11434` | Address of Ollama (spelling and translation). Only this computer is accepted. |
| `MADOC_SPELL_ALLOW_REMOTE` | not set | Set to `1` to allow an Ollama server elsewhere on your own network. |
| `HF_TOKEN` | not set | Hugging Face token, needed once to download the gated French NER model. |
| `HF_HUB_OFFLINE` | not set | Set to `1` once the NER models are downloaded: no more contact with Hugging Face. |
| `HF_HUB_DISABLE_SYMLINKS_WARNING` | not set | Set to `1` on Windows to hide a harmless warning of the Hugging Face cache. |
| `MADOC_FRENCH_NER_SPACE` | not set | Set to `1` to fall back on the online Hugging Face Space when the local French model is missing. This sends the text online. |
| `MADOC_AUTH` | not set | `user:password` (several separated by commas): the web interface asks for a login. |
| `BIRTH_DICT_DIR` | `dictionaries/` | Folder containing the dictionaries, if they are kept somewhere else. |

## Using the web interface

```bash
python src/app.py                                   # http://127.0.0.1:7860
python src/app.py --device cuda:0
python src/app.py --host 0.0.0.0                    # reachable from the other computers of the network
python src/app.py --share --auth demo:password      # public gradio.live link with a login, see Data protection
python src/app.py --segmentation-model path/blla.mlmodel --recognition-model path/htr.mlmodel
python src/app.py --no-pwa                          # not installable as an application
```

The screen is split in two, with a menu bar on top and a console at the bottom.

```
┌ Files │ Save │ Options ─────────────────────────────────────────────────── MaDoc  ? ┐
│ Image │ Segmentation          − + ⤓ │ Text │ JSON              ↶ ↷  S  E  ⤓          │
│                                     │                                               │
│   the scan, or the scan with the    │   the transcription line by line, or the      │
│   line polygons                     │   record as a table                           │
│                                     │                                               │
└─────────────────────────────────────┴───────────────────────────────────────────────┘
  console: validation errors and warnings, low-confidence lines, spelling and parser notes
```

| Control | What it does |
|---|---|
| **Files** | Two choices. **Import** a scan of your computer (jpg, png, tif), or **Choose from samples** one of the example scans of `samples/` (hidden when the folder has no image). The scan is processed at once: segmentation, recognition, spelling check, parsing, translation and validation. Phone photos are turned upright from their EXIF orientation first. |
| **Save** | Downloads a zip of four files, all named after the scan: the original scan, its segmentation as PageXML, the transcription and the record (`1.jpg`, `1.xml`, `1.txt`, `1.json` in `1.zip`). The PageXML names its image by the file name alone (`imageFilename="1.jpg"`), not by its path on the server. |
| **Options** | Spelling correction, Arabic NER, French NER, translation and the confidence threshold. **Run again with these options** processes the scan again from the HTR transcription; the previous versions of the text stay in the history. |
| **?** | A short help. |
| **Image / Segmentation** | The scan, and the scan with the polygon of every detected line coloured by the recognition confidence (green, amber, red under the threshold). **+ / −** zoom, scroll to move. **⤓** downloads the scan in the Image tab, the PageXML in the Segmentation tab. |
| **Text** | The transcription, one numbered line per text line with its confidence. **↶ / ↷** undo and redo every change of the text. **S** shows the spelling review. **E** opens the text for editing; pressing it again checks the spelling of the edited text and parses it again. **⤓** downloads the text. |
| **JSON** | The record as a table, one row per field with the Arabic and the French value; rows with a validation issue are coloured red (error) or amber (warning), and their tooltip gives the message. **{ } raw** switches to the raw JSON, **⤓** downloads it. |
| **Console** | The validation summary and issues, the lines under the confidence threshold, then the spelling and parser notes and the log of the run. |

Hovering a line anywhere highlights the same line everywhere: a polygon on the scan, a line of the text, the
rows of the record read from that line and its console entry. After a hand edit that changes the number of
lines, the text no longer matches the polygons; the console says so and the polygons are no longer linked.

**The spelling review (S).** Every correction is highlighted in the text with the proposed spelling as a badge:
green when it is applied, red when the rules refused it, orange when the certificate disagrees with itself (two
readings of the same address, a date that the Hijri date contradicts, a word written by hand). Clicking a
correction applies or removes it; when several words are possible (two jobs, the day before or after a date),
each click shows the next one. An orange mark without a badge (`≠ AYOUB`) has no correction to offer.
**Back to the rules' choice** resets every decision, **Refuse all** removes them all, and **Apply** parses the
text again with the chosen corrections.

**Hand edits.** When the text is edited by hand (E), the spelling check runs again on the edited text. Only the
lines that changed are sent to the model; the words typed by hand are never corrected by themselves (an orange
proposal, applied by a click), and a correction refused earlier in the review stays refused. The edit and the
corrections of the rules are two versions in the history, so ↶ gives back the text exactly as it was typed.

**Languages.** The interface is in English, French and Arabic. It follows the browser's language, and the
language can be changed in **Settings** at the bottom of the page (gradio does not keep that choice after a
reload). Buttons, tabs, options, help, the record table and the console are translated; the detailed messages
of the validator, the parser and the spelling checker stay in English. The texts are in `src/ui_text.py`.

**Installing MaDoc as an application.** The interface is a Progressive Web App: Chrome and Edge offer to install
it (the install icon in the address bar), and it then opens in its own window. The first click on **Files**
asks the permission to show notifications; when a scan finishes processing while MaDoc is in the background,
a notification says the record is ready. Notifications need `localhost` or HTTPS (a plain `http://` address of
another computer is not allowed by browsers).

**Sharing.** `--share` gives a temporary public address (`https://….gradio.live`) that works from anywhere.
`--auth user:password`, or `MADOC_AUTH` in `.env` to keep the password out of the shell history, makes the page
ask for a login; MaDoc prints a warning when `--share` is used without one. One scan is processed at a time: other
users wait in line.

The device and the model paths can only be set on the command line, so that visitors of a shared instance
cannot point kraken at other files on the server.

## Using the command line

### Full pipeline

```bash
python src/pipeline.py scan.jpg                            # whole chain, record printed on stdout
python src/pipeline.py scan.jpg -o record.json --report report.json
python src/pipeline.py scan.xml --image scan.jpg           # start from an existing segmentation
python src/pipeline.py samples/1.txt                       # start from a transcription
python src/pipeline.py scan.jpg --spellcheck               # with the spelling correction
python src/pipeline.py samples/1.txt --no-ner --no-translation   # rules only, no model loaded
python src/pipeline.py scan.jpg --save-stages out/         # keep the output of every stage
```

| Option | Effect |
|---|---|
| `-o FILE` | Write the record to a file instead of stdout. |
| `--report FILE` | Also write the validation report as JSON. |
| `--save-stages DIR` | Write the output of every stage: `0a_segmentation.xml`, `0b_text.txt`, `0c_spellchecked.txt`, `1_ner.json`, `2_parsed.json`, `3_converted.json`, `4_mapped.json`, `5_translated.json`, `6_report.json`. |
| `--spellcheck` | Run the spelling checker between recognition and the NER stage (the rules' choices are applied). |
| `--no-ner` | Do not load the NER models; the parser works from the rules alone. |
| `--ner FILE` | Reuse a saved NER output instead of running the models. |
| `--no-translation` | Skip the translation stage. |
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
python src/spelling_checker.py scan.txt -o fixed.txt       # every change is listed on stderr; --model MODEL
python src/classificator.py scan.txt -o ner.json           # --no-french, --no-arabic, --download
python src/parser.py scan.txt ner.json -o parsed.json --notes
python src/converter.py parsed.json -o converted.json
python src/mapper.py converted.json -o mapped.json
python src/translator.py mapped.json -o translated.json
python src/validator.py translated.json -o report.json     # exit code 1 when there is an error
```

## The output record

The record is a flat JSON object with a fixed set of fields. Arabic fields hold the text as written on the
certificate; fields ending in `FR` hold the French value. Example, from the anonymized sample `samples/1.txt`
(rules only, without translation, so the French name fields are empty):

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
  "DATENAISSPEREAR": "1407-03-03",
  "PROFESSIONPERE": "أجير",
  "PROFESSIONPEREFR": "SALARIER",
  "PRENOMMEREAR": "ريم الشاوي",
  "NOMMEREAR": "بنت أنس",
  "ADRESSENAISSMERE": "جماعة امرتريك اقليم سطات",
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
| | `HEURENAISS`, `MINNAISS` | Hour and minute of birth, integers; empty when the certificate leaves them blank. |
| | `LIEUNAISS`, `LIEUNAISSFR` | Place of birth. |
| | `NATIONALITE`, `NATIONALITEFR` | Nationality. When the certificate does not give it and both parents have the same one, the child gets it too (the notes say so). |
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
rules of the parser, so the spelling checker is placed before it. It works in three layers: the local model
proposes, the rules judge every proposal, and checks of the whole certificate add the corrections that need
more than one line to be seen.

**The model.** The text is sent line by line, with each line numbered, and the answer is matched by number, so
a line the model splits, joins or forgets never cancels the corrections of the others. The model's answer for a
line is kept: after a hand edit, only the changed lines are sent again. The model may restore a fixed word of the
form that the recognition lost (`الدقيقة`, `موافق`, `سنة`...), never a name, a place or a number.

**The rules.** The model's version is compared word by word with the original, and each change is judged alone:

- a change is accepted only if it replaces an unknown word with a word of the form, or with a number, month or
  time word from `date_time.json` (plus any word listed in `dictionaries/spelling_words.txt`);
- the original word must not already be a correct word, and at most one letter may differ (two for words of
  seven letters or more); a change of words that are already correct (`ثالث` → `الثالث عشر`) is not even shown;
- the words that follow a name, place or address label (`الإسم الشخصي`, `من والده`, `بن`, `السيدة`, `جماعة`,
  `بحي`, `زنقة`, ...) are never changed by the model, even when the label itself is misread or sits on the
  previous line;
- digits, Latin text and fragments of one or two letters are never changed;
- differences the parser does not see are ignored: punctuation, the space after a prefix (`ب جماعة` and
  `بجماعة`), `ڭ` and `ك`, and hamza on alif;
- a number or month word may be replaced by another one when the whole record read with the correction is
  better (an impossible date becomes possible, or the Gregorian and Hijri dates come to agree): `سنة ألفين و
  ثمانين` (2080, next to the Hijri year 1428) becomes `ثمانية` (2008). The validator of the record is the
  judge, so a correct number is never changed on a document without date errors.

A refused proposal of the model never blocks a dictionary: when the model proposes `من شعبان` for `ششوال`,
`date_time.json` still proposes `شوال`.

**Jobs** are corrected towards `jobs.json` only. A misread job receives every job of the dictionary at the
smallest distance, with the model's choice listed first. When there is a single candidate it is applied; when
there are several (`صباخ` may be `صباغ`, painter, or `طباخ`, cook), nothing is applied until someone chooses.
When the declarant is a parent (`(والدها)`), the job written in the declaration is the same person's job written
a second time, and only the jobs that fit both readings are kept. The job check works without Ollama.

**The child's names** are written twice, in Arabic (`الإسم الشخصي`, `الإسم العائلي`) and in Latin letters
(`PRENOM`, `NOM`). The checker compares the consonants of the two readings, following the usual French spellings
of Moroccan names (`ش` = CH, `خ` = KH, `غ` = GH, `و` = OU, a leading `ال` written or not), so `قاسم` / `KACIM`
and `الإدريسي` / `IDRISSI` agree while `أيرب` / `AYOUB` does not. When they disagree:

1. the same name written elsewhere in the certificate (usually the father's family name) that agrees with the
   Latin reading is applied: `الودعيري` / OUDGHIRI becomes `الودغيري` when the father is `خالد الودغيري`;
2. otherwise a first name of `names.json` one letter away that agrees with the Latin reading is applied
   (`إدريش` / IDRISS → `ادريس`), or proposed in orange when two letters differ; when the Arabic reading is itself
   a first name of `names.json`, the Latin reading is the one corrected;
3. otherwise the local model transliterates each reading into the other script, and the side that needs the
   fewest letters changed gets a proposal, which is never applied automatically;
4. without any proposal, the name is marked `≠ AYOUB` in orange and the notes say what the model answered.

**The certificate read as a whole.** Most fields are written twice or depend on each other, and the checker
uses it:

| Check | Example | Applied |
|---|---|---|
| A misread number or month word (`date_time.json`), judged by the record | `ألفين و خمسن عشرة` → `خمسة` (2015, as the Hijri 1436); two misread words of the same date are judged together | yes |
| The minute's label lost: the split where both the hour and the minute are time phrases | `الرابعة` / `وة : الثلاثين` → `و الدقيقة` | yes, when the record reads the hour and the minute |
| A word of the form or of an address one letter away | `عمرهار` → `عمرها`, `الرق` → `الرقم`, `محج المستتشفى` → `المستشفى` | yes |
| A label glued to the word before it | `العربجننسيته` → `العرب جنسيته`; on the mother's line, `العمرانيبست` → `العمراني بنت` | yes |
| A place of `cities.json` one letter away, after a place label | `اقليم برشبد` → `برشيد`, `اليوسفة` → `اليوسفية` | when the place is written elsewhere in the certificate; otherwise proposed |
| A first name of `names.json` with one letter misread, after a name label | `بنت حالد` → `خالد`, `عبد العزير` → `العزيز` | when the name is written elsewhere; otherwise proposed |
| An unreadable Gregorian date next to a readable Hijri date | `رامع بير` → `رابع يناير` (the Hijri 26 Dhu al-Hijja 1428), also when the year is on the next line | when the date of the Hijri one is close to what was read (the day closest to the reading first); otherwise proposed in orange, with the day before / after as other options |
| A Gregorian year that differs from the Hijri date's while the day and month are the same | `ثاني مارس ... تسعة و ستين` next to 10 Dhu al-Qa'da 1385 (2 March 1966) | proposed in orange (registry staff sometimes write another year on purpose) |
| The declarant is a parent: the same person's name in the header and in the declaration | `عرلاب الزروالي` / `السيدة غزلان الزروالي (والدتها)` | the reading that is a first name of `names.json`, or that agrees with the child's Latin family name, when two letters at most differ; otherwise proposed in orange |
| The house number of the two addresses (the parents' and the declarant's) | `الرقم 591` / `الرقم 519` | proposed in orange on both (digits are never changed by themselves) |
| The street or neighbourhood name of the two addresses | `بتجزئة الفتح` / `ب تجزئة الفيقة` | the reading listed in `spelling_words.txt` or written elsewhere in the certificate; otherwise proposed in orange on both |

The proposed dates are written as on the certificates and read back by the converter before being proposed.
When several places are one letter away (`أسي`: `آسفي` or `أسا`), the one written elsewhere, then the one in the
same region as the other places of the line, comes first, and a place of another region is not proposed. The
second word of a compound place (`اولاد فارس`, `بني ملال`, `سيدي رحال`) is never replaced by a single city, and a
name or place read the same way twice is left alone. The name after `حي`, `زنقة`, `شارع`, ... is only corrected
towards the words of `spelling_words.txt`, where the neighbourhoods and streets of your documents can be listed.

### Named entity recognition

Two MagBERT models are run on the text: the Arabic model finds persons, places and dates with their position
in the text, and the French model reads the Latin names. The French model is only given the lines written in
Latin letters (`PRENOM ...`, `NOM ...`); a name in capitals alone on a line is hard for it, so a line is also
tried in title case and inside a short French sentence, and the positions are mapped back to the text. The parser
uses these entities as a second, independent reading of every field (see below). The stage is optional: with
`--no-ner`, or when a model cannot be loaded, the parser relies on its rules alone. The entities of a text are
kept, so an undo or an edit elsewhere does not run the models again.

### Parsing

The parser cuts the text into four zones using the fixed labels of the form: the child (from `في يوم` to
`من والده`), the father (to `و والدته`), the mother (to `الساكنان`) and the administration (the address,
the declaration and the signature). Each field is then read within its zone. The rules tolerate the variants
found on real certificates: labels split over two lines (`والد` / `ها`), common misspellings, a nationality or
job printed in the wrong block, the child's Latin names printed inside a parent's block, a label with nothing
after it (`و الدقيقة` alone gives an empty minute, not the next line), and so on. A masculine `جنسيته` read in
the mother's block is the mother's nationality when the father already has his own.

When NER entities are available, every field is also read from the entities of its zone, without using any
label. The two readings are then compared:

- if the rules found nothing, the NER value is used;
- for a date, if only one of the two converts to a valid date of the right calendar, that one is kept;
- if the rules swallowed a label word into a name or a place, the NER value is used;
- in every other disagreement the rules are kept and the difference is reported in the notes.

### Conversion and mapping

Dates are written out in words on the certificates (`ثالث و عشري شوال عام ألف و أربعمائة و أربعة و عشرين`).
The converter recognises the day, month and year phrases with `date_time.json`, in every spelling seen in the
documents, and produces `YYYY-MM-DD` (or `YYYY` when only a year is given). An unknown word after `و` inside a
number makes the date unreadable rather than silently wrong (`ألفين و ثلاثن` is not 2000). Hours and minutes are
matched as whole phrases against the time section of the dictionary. The sex becomes `1` or `0`.

The mapper looks up the jobs in `jobs.json` and the nationalities in `nationalities.json`, with tolerance for
small spelling mistakes. A job that is not in the dictionary is left for the translation stage.

### Translation

The French names are built by the code, word by word, and only what the dictionaries do not know goes to the
local model, in one request per record:

- the lineage word is written by the code (`بن` = `Fils de`, `بنت` = `Fille de`);
- first names come from `names.json`, and the father's family name from the child's Latin name;
- the remaining words are sent to the model; a reading whose consonants do not match the Arabic is reported on
  stderr;
- official place names of `cities.json` and the usual address words (`مستشفى` = `Hôpital`, `جماعة` = `Commune`,
  `محج` = `Boulevard` ...) are imposed through a glossary, and the model is told to write French, never English;
- the officer's title is fixed (`FONCTIONAIRE ET OFFECIER DETAT CIVIL`); a title written by the model is dropped;
- any answer that still contains Arabic script is rejected and the original value is kept.

The items are numbered and the answer is matched by number. The answer for a record is kept, so parsing the same
record again does not call the model. Without Ollama, or when the model is not pulled, this stage is skipped and
the report marks the French fields as untranslated.

### Validation

The validator checks the structure and content of the record and returns a list of issues, each with a level:

- **error**: the record is wrong and must be corrected. This covers missing required fields, values of the
  wrong type or out of range (hour 0-23, minute 0-59, sex 0/1), invalid dates, a declaration dated before the
  birth, a parent born after the child, Latin letters in an Arabic field or Arabic in a French field, label
  words caught inside a value (a sign that a rule captured too much), and modified fixed values.
- **warning**: the value is plausible but should be reviewed. Examples: a Gregorian and Hijri year that do not
  match (registry staff sometimes enter them that way on purpose), an hour or minute not written, a parent
  younger than 14 or older than 80 at the birth, a declaration made more than a year after the birth, a job or
  nationality missing from the dictionaries, or an unusually long value.

A record is valid when it has no error.

## Dictionaries

All dictionaries are JSON files in `dictionaries/` and can be edited by hand. Every lookup ignores diacritics,
hamza and the article `ال`, so a single entry covers the usual spelling variants.

| File | Format | Used for |
|---|---|---|
| `date_time.json` | `hijri_months`, `miladi_months`, `numbers`, `time`: value → list of written forms, the usual one first | Converting dates, hours and minutes. The spelling checker also uses it as vocabulary. |
| `jobs.json` | `"Arabic job": "FRENCH JOB"` | French jobs; the spelling checker's job candidates. |
| `nationalities.json` | `{"ar": [...], "fr": [...]}` per nationality | French nationalities. |
| `cities.json` | list of `{"name": French, "nameAr": Arabic, "region": ...}` | Official French spelling of Moroccan places; places and regions for the spelling checker. |
| `names.json` | `"Arabic first name": "French spelling"` (one to three words) | French first names, imposed on the translation; the spelling checker's first names. |
| `places.json` (optional) | `"Arabic": "French"` | Extra places or corrections; overrides `cities.json`. |
| `spelling_words.txt` (optional) | words separated by spaces, `#` for comments | Extra words the spelling checker may correct towards, including the names of your neighbourhoods and streets (`المقاومة` fixes `حي المعاومة`, `بحي القاومة`). |

Adding a job that appears in your documents, for example `"حداد": "FORGERON"`, improves both the French
mapping and the spelling correction. Adding a first name to `names.json` (`"عزيز": "Aziz"`) fixes its French
spelling for good, whatever the model would have written.

## Data protection

Birth certificates are personal data. Every stage runs on the computer that processes them:

| Stage | Data sent | Destination |
|---|---|---|
| Segmentation, recognition, spelling, NER (both models), parsing, conversion, mapping, translation, validation | nothing | |
| French NER, only with `MADOC_FRENCH_NER_SPACE=1` and no local model | the whole transcription | Hugging Face Space `TypicaAI/MagBERT-NER-Fr` |
| Web interface started with `--share` | everything uploaded and displayed | gradio's relay server (gradio.live) |

The spelling checker and the translator refuse any Ollama address other than this computer unless
`MADOC_SPELL_ALLOW_REMOTE=1` is set. A `--share` link is reachable from the internet: use it with a login
(`--auth`) and only with anonymized or blank documents. To give access to the other computers of a lab instead,
start the interface with `--host 0.0.0.0`: nothing leaves the local network.

The web interface keeps the files of a run (the scan, the downloads) in a temporary folder and deletes them one
hour after their last use; gradio's own copy of an upload is deleted after one hour too. Nothing is written to
the project folder.

The files in `samples/` are anonymized: names, act and house numbers were replaced and every year was shifted,
Gregorian and Hijri separately so the pairs stay consistent. The tests only use these samples (see below). Do not
commit real scans or transcriptions; `.gitignore` already excludes `*.jpg`, `*.xml` and `.env`.

## Tests and continuous integration

```bash
pip install -r requirements-ci.txt      # pytest, python-dotenv, python-bidi, pillow
python -m pytest -q
```

The tests need neither the models nor a network connection. kraken, the NER models and Ollama are replaced by
stand-ins (the Ollama stand-in is a small local HTTP server that speaks the same API).

| File | What it covers |
|---|---|
| `tests/documents.py` | The documents of the tests: every one is a sample of `samples/` with some words replaced by the reading mistakes being tested. No test contains a certificate of its own. |
| `tests/test_samples.py` | Every text in `samples/` goes through parser, converter, mapper and validator without any error. Add a `.txt` file to `samples/` to add a case. |
| `tests/test_units.py` | File encodings, kraken calls and error handling, GPU fallback, PageXML parsing, the parser's empty values and nationalities, the converter, the local NER and the translator, the web interface (files of a run, history, hand edits, login). |
| `tests/test_spelling.py` | The spelling checker: accepted and refused changes, protected values, time phrases, jobs and their context, names, dates and addresses checked across the certificate, the spelling review. |

GitHub Actions runs the same command on Python 3.10 and 3.12 for every push and pull request
(`.github/workflows/tests.yml`). The tests of the web interface are skipped when gradio is not installed, which
is the case in CI.

## Repository layout

```
madoc/
├── src/
│   ├── app.py               web interface (Gradio)
│   ├── ui_text.py           texts of the interface in English, French and Arabic
│   ├── pipeline.py          runs the whole chain
│   ├── segmentation.py      stage 0a: image → PageXML
│   ├── recognition.py       stage 0b: PageXML → text
│   ├── spelling_checker.py  stage 0c: spelling correction (local model + rules)
│   ├── classificator.py     stage 1: named entities (local models)
│   ├── parser.py            stage 2: text → record
│   ├── converter.py         stage 3: dates, time, sex
│   ├── mapper.py            stage 4: jobs, nationalities
│   ├── translator.py        stage 5: French names and places (local model)
│   ├── validator.py         stage 6: report
│   └── common.py            settings (.env), dictionaries, record template, Arabic normalisation, helpers
├── dictionaries/            date_time.json, jobs.json, nationalities.json, cities.json, names.json
├── models/                  blla.mlmodel (segmentation), htr.mlmodel (recognition)
├── samples/                 anonymized transcriptions used by the tests, example scans of the Samples button
├── logo/                    the logos: the wide one (with the name) in the top bar and on the login page,
│                            its version for the dark theme (a name with "dark": madoc_dark.png), the square
│                            one as the icon of the browser tab and of the installed application
├── tests/                   pytest suite and its documents
├── .github/workflows/       CI configuration
├── requirements.txt         full installation
└── requirements-ci.txt      what the tests need
```

## Known limitations

- The parser is built for the birth certificate form used by Moroccan civil status offices. Other documents
  (marriage, death certificates) would need their own zones and rules.
- Names are only corrected when the certificate itself supports the correction (the Latin reading, the same
  name written elsewhere, `names.json`); otherwise they are proposed or flagged, and the reading has to be fixed
  by hand.
- The quality of the French transliteration of the words missing from `names.json` depends on the local model:
  a 7B model is weaker than a large online one on rare Moroccan family names. A larger model can be set with
  `MADOC_TRANSLATE_MODEL`.
- Hijri and Gregorian dates are compared with the arithmetic (tabular) Hijri calendar, which can differ from
  the calendar observed in Morocco by a day or two: date proposals therefore offer the day before and after.
- The spelling checker with a 7B model takes a few minutes per certificate on a CPU, a few seconds on a GPU.

## Troubleshooting

**`Fatal error in launcher: Unable to create process`.** The `.venv` was created in another folder. Delete
`.venv`, create it again and reinstall the requirements.

**`kraken segmentation failed: no output written`.** Read the kraken message printed just above it. The most
common cause used to be `--device cuda:0` with a CPU-only PyTorch; MaDoc falls back to the CPU in that case
and prints a warning.

**`dictionaries not found`.** Run the commands from the project root, or set `BIRTH_DICT_DIR` to the folder
that contains `date_time.json`.

**`NER unavailable ... cannot be loaded`.** The model is not downloaded on this computer, or `HF_HUB_OFFLINE=1`
is set before the first download, or no Hugging Face token is set (`hf auth login`, or `HF_TOKEN` in `.env`).
Run `python src/classificator.py --download` and read its message. `huggingface-hub>=1.5.0,<2.0 is required`
means `huggingface_hub` was upgraded too far: `pip install "huggingface_hub>=1.5,<2"`. A tokenizer error about
SentencePiece or tiktoken means `pip install sentencepiece protobuf`.

**`spelling model skipped: Ollama not reachable`** or **`Translation model unavailable`.** Start Ollama and
check that the model is installed (`ollama list`); the message gives the `ollama pull` command when the model is
missing. The record is still produced: the job check runs without Ollama, and the French name fields keep the
Arabic values.

**A field is wrong or empty.** Run the pipeline with `--save-stages out/` and look at the files in order:
`0b_text.txt` (what was read), `2_parsed.json` (what the rules found) and the notes printed on stderr. Most
problems come from a recognition error in a label, which can be corrected in the Text tab (E) or by the spelling
checker.