"""Stage 2 - parser: raw text + NER entities -> JSON object (the template of common.py).

    python src/parser.py samples/6.txt ner.json               # ner.json comes from classificator.py
    python src/parser.py samples/6.txt ner.json -o parsed.json --notes
    python src/parser.py samples/6.txt                        # without NER (rules only)

Lines read out of order
  0. The HTR does not always read the lines in the order of the form. restore_order() recognizes every line by
     its label and puts it back in its part (child / father / mother / address / declaration) and in the order
     of the form; a date or a nationality line goes to its parent by the year of the Hijri date next to it and
     by its gender ("جنسيته" / "جنسيتها"). A text already in order is not changed. The NER offsets move with
     their lines.

How the two sources are combined
  1. The rules (regex) cut the text into zones (child / father / mother / administration) with the
     fixed labels of the form ("من والده", "و والدته", "الساكنان", ...) and read every field in its zone.
  2. The NER entities (PERSON / GPE / DATE, with their position in the text) are assigned to the same
     zones and read as a SECOND extractor: for every field they give a candidate value without using any
     label, so a typo in a label ("الاسم الشحصي", "ميلاي ية") does not matter to them.
  3. reconcile() compares the two values of each field:
       - the rules found nothing                        -> the NER value
       - a date: only one of them is a valid date        -> the valid one
       - a name / place with a label word swallowed      -> the NER value
       - anything else that differs                      -> the rules value is kept and the difference noted
  4. Every decision is returned as a note (parse_with_notes / --notes), so a doubtful record can be reviewed.
Without NER the output is the output of the rules alone.
"""

import argparse
import copy
import re
import sys
from bisect import bisect_left
from difflib import SequenceMatcher
from dataclasses import dataclass
from pathlib import Path

from common import (
    ARABIC_DIACRITICS,
    CHAR_MAP,
    DIGIT_MAP,
    INVISIBLE_CHARS,
    TEMPLATE,
    ar_key,
    extract,
    extract_any,
    get_jobs_lookup,
    get_nationality_lookup,
    get_place_first_words,
    normalize_text,
    read_json_file,
    read_text_file,
    write_json,
)
from converter import replace_aliases

# place words that can be glued to the preposition ب (e.g. بجماعة)
PLACE_WORDS = (
    r"(?:جماعة|مدينة|دوار|قرية|دائرة|[اإ]قليم|عمالة|مستشفى|المستشفى|حي|مركز|المركز|"
    r"قيادة|باشوية|ولاية|جهة|مقاطعة|منزل|مصحة|الجماعة|المدينة|الدوار|الدائرة|ال)"
)
BIRTH_PREP = r"(?:ب\s+|ب(?=" + PLACE_WORDS + r"))"
# "موافق" is sometimes written "موفق"
# birth verb, also with the typo "والدت" ("ولد" / "ولدت")
BORN_VERB = r"(?:ولد|والد)ت?"
# markers of the parent blocks; the suffix may be split over two lines ("والد" + "ها")
FATHER_MARK = r"من\s+والد\s*(?:ه|ها)"
# "و" is often missing or on the line before ("والدته" alone at the start of a line); "(والدته)" of the declaration
# is not a marker
MOTHER_MARK = r"(?<!\(\s)(?:و\s*)?والدت\s*(?:ه|ها)"


# ---------------------------------------------------------------------------
# text cleaning
# ---------------------------------------------------------------------------
# characters that are replaced by a space
PUNCT_TO_SPACE = {":", "،", "؛", "٪", "﹕", "："}


def flatten(raw):
    """Cleaned text on one line, and for every kept character its index in the raw text.

    Cleaning: Arabic-Indic digits -> ASCII, invisible characters, diacritics and "." removed, ":" "،" "؛" "٪"
    and every run of whitespace (new lines too) -> one space. The indexes let the NER offsets (which are
    positions in the raw text) be moved to the cleaned text the rules work on.
    """
    chars, index = [], []
    space_at = None  # raw index of the first whitespace of the current run
    for i, ch in enumerate(raw):
        if INVISIBLE_CHARS.match(ch) or ARABIC_DIACRITICS.match(ch) or ch == ".":
            continue
        if ch.isspace() or ch in PUNCT_TO_SPACE:
            if chars and space_at is None:
                space_at = i
            continue
        if space_at is not None:
            chars.append(" ")
            index.append(space_at)
            space_at = None
        chars.append(ch.translate(DIGIT_MAP))
        index.append(i)
    return "".join(chars), index


# ---------------------------------------------------------------------------
# rules (regex)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Zone:
    """A part of the cleaned text (child / father / mother / administration) and where it is."""
    text: str = ""
    start: int = 0
    end: int = 0


def find_zone(patterns, text):
    """First pattern whose group 1 is not empty -> Zone (empty Zone when nothing matches)."""
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if match and match.group(1).strip():
            group = match.group(1)
            start = match.start(1) + len(group) - len(group.lstrip())
            stripped = group.strip()
            return Zone(normalize_text(stripped), start, start + len(stripped))
    return Zone()


# one Latin word (French names may contain accents, apostrophes and hyphens)
LATIN_WORD = re.compile(r"^[A-Za-zÀ-ÿ'’\-]+$")


def latin_field(label, text):
    """Read the French value written after PRENOM / NOM, wherever the scan put it."""
    match = re.search(r"(?<![A-Za-z])" + label + r"\s+", text)
    if not match:
        return ""
    words = []
    for token in text[match.end():].split():
        if not LATIN_WORD.match(token) or token in ("NOM", "PRENOM"):
            break
        words.append(token)
    return " ".join(words)


def strip_latin_labels(block):
    """Remove 'PRENOM xxx' / 'NOM xxx' (child's French names) that landed in a parent block."""
    for label in ("PRENOM", "NOM"):
        value = latin_field(label, block)
        if value:
            block = re.sub(
                r"(?<![A-Za-z])" + label + r"\s+" + re.escape(value), " ", block, count=1
            )
    return normalize_text(block)


# words that start an address (used to spot an address piece printed in the wrong place)
ADDRESS_START = (
    r"(?:ال)?(?:زنقة|رقم|شارع|درب|حي|دوار|مدينة|تجزئة|[إا]قامة|عمارة|شطر|مجموعة|زاوية)(?:\s|$)"
)


def strip_attached_b(place):
    """'بخريبڭة' -> 'خريبڭة' when the rest is a known city (the ب is only the preposition)."""
    if not place:
        return place
    first, _, rest = place.partition(" ")
    firsts = get_place_first_words()
    if (
        len(first) > 1 and first.startswith("ب")
        and ar_key(first) not in firsts and ar_key(first[1:]) in firsts
    ):
        return normalize_text(first[1:] + " " + rest)
    return place


# words that end a profession when the profession is not in the jobs dictionary
JOB_STOP = (
    r"(?=\s*(?<!\S)(?:المولود|جنسيت|في\s|(?:ال)?موا?فق|حرفت|مهنت|الساكن|الزنقة|الرقم|زنقة|رقم|"
    r"شارع|درب|دوار|مدينة|بناء|توقيع|و\s*والد|والدت\s*(?:ه|ها)(?!\S))|\s*$)"
)


def extract_job(label, text):
    """Find the profession after a label, wherever the scan put it in the block."""
    tail = extract(label + r"\s+(.*)", text)
    if not tail:
        return ""
    tokens = tail.split()
    lookup = get_jobs_lookup()
    # 1. longest profession known by jobs.json
    for n in range(min(5, len(tokens)), 0, -1):
        if ar_key(" ".join(tokens[:n])) in lookup:
            return " ".join(tokens[:n])
    # 2. otherwise read up to the next known label / address word
    return extract(r"^(.*?)" + JOB_STOP, tail)   # empty when the next label follows ("حرفته : و والدته")


def trim_nationality(value):
    """'مغربية المولو دة بجماعة ...' -> 'مغربية': a nationality is a short known word, cut at the longest one."""
    tokens = value.split()
    lookup = get_nationality_lookup()
    for n in range(min(3, len(tokens)), 0, -1):
        if ar_key(" ".join(tokens[:n])) in lookup:
            return " ".join(tokens[:n])
    return value


def parse_parent(block, nat_label, born_label, job_label, core=""):
    """Parse a father or mother block (labels differ only by gender)."""
    lineage = r"(?:ابن|بن|بنت|ابنة)"
    stop = r"(?=\s*(?<!\S)جنسيت|\s*(?<!\S)المولود|\s*(?<!\S)في\s|\s*(?<!\S)(?:ال)?موا?فق|\s*(?<!\S)حرفت|\s*(?<!\S)مهنت|\s*$)"
    return {
        # Everything before بن/بنت (or before the nationality when there is no lineage word)
        "first": extract_any([
            r"^(.*?)(?=\s*(?<!\S)" + lineage + r"(?:\s|$))",
            r"^(.*?)(?=\s*(?<!\S)جنسيت|\s*(?<!\S)المولود|\s*$)",
        ], block),
        # بن/بنت and what comes after
        "last": extract(
            r"(?<!\S)(" + lineage + r"\s+.+?)" + stop, block
        ),
        "nationality": extract(
            nat_label + r"\s+(.*?)(?=\s*(?<!\S)المولود|\s*(?<!\S)في\s|\s*(?<!\S)(?:ال)?موا?فق|\s*(?<!\S)حرفت|\s*(?<!\S)مهنت|\s*$)", block
        ),
        "birthplace": extract(
            born_label + r"\s*(?:" + BIRTH_PREP + r")?(.*?)(?=\s*(?<!\S)جنسيت|\s*(?<!\S)في\s|\s*(?<!\S)(?:ال)?موا?فق|\s*(?<!\S)حرفت|\s*(?<!\S)مهنت|\s*$)", block
        ),
        "date_ar": extract(r"(?<!\S)في\s+(.*?)(?=\s*(?<!\S)هجر)", block),
        "date_fr": extract(r"(?:ال)?موا?فق\s*(.*?)(?=\s*(?<!\S)ميلا)", block),
        # the job is searched in the block first, then in the whole parents zone
        # (some scans put "حرفته" / "حرفتها" far from its parent block)
        "job": extract_job(job_label, block) or extract_job(job_label, core),
    }


# ---------------------------------------------------------------------------
# NER entities
# ---------------------------------------------------------------------------
PERSON, GPE, DATE = "PERSON", "GPE", "DATE"
LABEL_ALIASES = {
    "PERSON": PERSON, "PER": PERSON, "PERS": PERSON,
    "GPE": GPE, "LOC": GPE, "LOCATION": GPE,
    "DATE": DATE,
}


def label_kind(label):
    """'B-PER' / 'PERSON' / 'person' -> 'PERSON' (unknown labels are kept upper cased)."""
    label = re.sub(r"^[BIES]-", "", str(label or "").strip().upper())
    return LABEL_ALIASES.get(label, label)


@dataclass(frozen=True)
class Entity:
    """A NER entity located in the cleaned text."""
    kind: str
    start: int
    end: int
    text: str


def _squash(value):
    """Text without spaces, diacritics and invisible characters (for comparisons)."""
    return re.sub(r"\s+", "", ARABIC_DIACRITICS.sub("", INVISIBLE_CHARS.sub("", value)))


def _similar(a, b):
    """True when two strings are the same text, or nearly (the model rebuilds words from word pieces)."""
    a, b = _squash(a), _squash(b)
    return a == b or SequenceMatcher(None, a, b).ratio() >= 0.7


# how far after the previous entity an entity searched by its text may be (characters)
SEARCH_WINDOW = 1000


def locate_entity(raw, word, cursor):
    """Find an entity string in the raw text, after `cursor`: (start, end) or None.

    The model's string is not always a substring of the document (the word pieces are decoded again,
    spaces / letters can change), so when the exact string is not found the first word is searched and
    as many words are taken.
    """
    position = raw.find(word, cursor)
    # an entity that is far after the previous one is a repeat of the word elsewhere in the document
    # (the real one is missing): do not jump there, every following entity would be lost
    if 0 <= position - cursor <= SEARCH_WINDOW:
        return position, position + len(word)
    tokens = word.split()
    if tokens and len(tokens[0]) >= 3:
        position = raw.find(tokens[0], cursor)
        if 0 <= position - cursor <= 400:
            match = re.compile(r"\S+(?:\s+\S+){%d}" % (len(tokens) - 1)).match(raw, position)
            if match:
                return match.start(), match.end()
    return None


def build_entities(ner, raw, flat, index, notes):
    """Arabic NER items -> Entity list positioned in the cleaned text.

    Uses "start"/"end" (positions in the raw text) when they are present and consistent, otherwise
    the entity string is searched in the raw text, in order (a NER json without offsets still works).
    """
    items = [i for i in (ner or {}).get("arabic_ner") or [] if i]
    entities, cursor, searched = [], 0, 0
    for item in items:
        word = str(item.get("entity", item.get("word", ""))).strip()
        kind = label_kind(item.get("label", item.get("entity_group")))
        if not word:
            continue
        start, end = item.get("start"), item.get("end")
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(raw)
                and _similar(raw[start:end], word)):
            found = locate_entity(raw, word, cursor)
            if found is None:
                notes.append(f"NER entity {word!r} ({kind}) not found in the text, ignored")
                continue
            start, end = found
            searched += 1
        cursor = max(cursor, end)
        # raw positions -> positions in the cleaned text
        flat_start, flat_end = bisect_left(index, start), bisect_left(index, end)
        while flat_start < flat_end and flat[flat_start] == " ":
            flat_start += 1
        while flat_end > flat_start and flat[flat_end - 1] == " ":
            flat_end -= 1
        if flat_end <= flat_start:
            continue
        # the model sometimes stops in the middle of a word ("ميلاي"): take the whole word
        while flat_end < len(flat) and flat[flat_end] != " ":
            flat_end += 1
        entities.append(Entity(kind, flat_start, flat_end, flat[flat_start:flat_end]))
    if items and searched == len(items):
        notes.append("NER json has no usable offsets: entities were located by searching their text")
    return sorted(entities, key=lambda e: (e.start, e.end))


def french_person_tokens(ner):
    """PERSON tokens of the French model, in text order."""
    return [
        str(item.get("entity", "")).strip()
        for item in (ner or {}).get("french_ner") or []
        if label_kind(item.get("label")) == PERSON and str(item.get("entity", "")).strip()
    ]


# words that can sit between two entities of the same name / place / date
LINEAGE_WORDS = {ar_key(w) for w in ("بن", "ابن", "بنت", "ابنة")}
PLACE_LEFT = {ar_key(w) for w in (
    "جماعة", "مدينة", "دوار", "قرية", "دائرة", "اقليم", "إقليم", "أقليم", "عمالة", "حي", "مركز",
    "قيادة", "باشوية", "مقاطعة", "مشيخة", "مدشر", "ولاية", "جهة",
)}
PLACE_RIGHT = {ar_key(w) for w in ("الحضرية", "القروية", "حضرية", "قروية")}
PLACE_JOINERS = PLACE_LEFT | PLACE_RIGHT
DATE_FILLERS = {ar_key(w) for w in ("في", "يوم", "عام", "موافق", "موفق", "الموافق", "لسنة")}
# the closing word of a date, compared after CHAR_MAP (ة -> ه): "هجرية" / "ميلادية"
TERMINATORS = (("hijri", "هجريه"), ("miladi", "ميلاديه"))


def entities_in(zone, entities, kind=None):
    """Entities that start inside the zone (optionally of one kind)."""
    if not zone.text:
        return []
    return [e for e in entities
            if zone.start <= e.start < zone.end and (kind is None or e.kind == kind)]


def group_runs(ents, flat, joiners=frozenset()):
    """Merge entities separated only by whitespace or by joiner words -> [(start, end)]."""
    runs = []
    for ent in sorted(ents, key=lambda e: e.start):
        if runs:
            gap = flat[runs[-1][1]:ent.start] if ent.start > runs[-1][1] else ""
            if all(ar_key(token) in joiners for token in gap.split()):
                runs[-1] = (runs[-1][0], max(runs[-1][1], ent.end))
                continue
        runs.append((ent.start, ent.end))
    return runs


def widen_place(start, end, flat):
    """Add the administrative words around a place: 'جماعة' + خريبڭة + 'الحضرية'."""
    while True:
        previous = flat[:start].rstrip().rsplit(" ", 1)[-1]
        if previous and ar_key(previous) in PLACE_LEFT:
            start = flat.rfind(previous, 0, start)
        elif previous.startswith("ب") and ar_key(previous[1:]) in PLACE_LEFT:
            start = flat.rfind(previous, 0, start) + 1  # glued preposition: "بجماعة"
            break
        else:
            break
    while True:
        match = re.match(r"\s+(\S+)", flat[end:])
        if match and ar_key(match.group(1)) in PLACE_RIGHT:
            end += match.end()
        else:
            break
    return start, end


def first_place(ents, flat, before=None, after=None):
    """First run of GPE entities (before / after a position) -> text with its administrative words."""
    runs = group_runs([e for e in ents if e.kind == GPE], flat, PLACE_JOINERS)
    for start, end in runs:
        if (before is not None and start >= before) or (after is not None and start < after):
            continue
        start, end = widen_place(start, end, flat)
        return flat[start:end]
    return ""


def is_date_filler(token):
    """'في', 'يوم', 'عام', 'موافق', 'لسنة' ... (the long ones also with a typo: 'مواقق')."""
    key = ar_key(token)
    return key in DATE_FILLERS or (
        len(key) >= 4 and any(len(f) >= 4 and SequenceMatcher(None, key, f).ratio() >= 0.8 for f in DATE_FILLERS)
    )


def strip_date_fillers(phrase):
    """Remove the label words at the start of a date ('في يوم', 'عام', 'موافق', 'لسنة')."""
    tokens = phrase.split()
    while tokens and is_date_filler(tokens[0]):
        tokens.pop(0)
    return " ".join(tokens)


def terminator_at(tokens, i):
    """Is there a closing word "هجرية" / "ميلادية" at tokens[i], with typos? -> (kind, words used) or None.

    The word may be misspelt ("هجريه", "ميلادبة") or cut in two by the scan ("ميلاي" + "ية"), so tokens[i]
    alone and tokens[i] + tokens[i+1] glued are compared with the two words (difflib ratio >= 0.8).
    """
    best = None
    for count in (1, 2):
        if i + count > len(tokens):
            break
        word = "".join(tokens[i:i + count]).translate(CHAR_MAP)
        for kind, canonical in TERMINATORS:
            if word[:1] != canonical[:1]:
                continue
            ratio = SequenceMatcher(None, word, canonical).ratio()
            if ratio >= 0.8 and (best is None or ratio > best[0]):
                best = (ratio, kind, count)
        if best and best[0] >= 0.95:
            break  # a (nearly) exact single word: do not glue the next word to it
    return (best[1], best[2]) if best else None


def date_phrases(ents, flat):
    """DATE entities -> [("hijri"|"miladi", phrase)] in text order.

    The model cuts a date in many small entities ("ثالث و عشري" + "شوال" + "ألف و أربعمائة" ...), so the
    text from the first to the last DATE entity is taken and cut after every closing word
    ("هجرية" / "ميلادية", typos accepted, see terminator_at).
    """
    ents = sorted(ents, key=lambda e: e.start)
    if not ents:
        return []
    start, end = ents[0].start, max(e.end for e in ents)
    # the closing word may be untagged or cut in two: look at the next two words
    following = list(re.finditer(r"\S+", flat[end:end + 40]))
    hit = terminator_at([m.group() for m in following[:2]], 0)
    if hit:
        end += following[hit[1] - 1].end()
    span = flat[start:end]
    words = list(re.finditer(r"\S+", span))
    texts = [m.group() for m in words]
    phrases, last, i = [], 0, 0
    while i < len(words):
        hit = terminator_at(texts, i)
        if not hit:
            i += 1
            continue
        kind, count = hit
        phrase = strip_date_fillers(span[last:words[i].start()])
        if phrase:
            phrases.append((kind, phrase))
        last = words[i + count - 1].end()
        i += count
    return phrases


def split_lineage(text):
    """'خالد الودغيري بن رضا' -> ('خالد الودغيري', 'بن رضا')."""
    tokens = text.split()
    for i, token in enumerate(tokens):
        if i > 0 and ar_key(token) in LINEAGE_WORDS:
            return " ".join(tokens[:i]), " ".join(tokens[i:])
    return text, ""


def agrees(value, candidates):
    """True when the value of the rules is one of (or inside / contains) the candidates of the model."""
    key = ar_key(value)
    return any(key in ar_key(c) or ar_key(c) in key for c in candidates if c)


def longer_than(value, candidate):
    """True when `value` starts with `candidate` (word by word) and has more words after it."""
    a, b = ar_key(value).split(), ar_key(candidate).split()
    return bool(b) and len(a) > len(b) and a[:len(b)] == b


# words that belong to the labels of the form: if they are inside a name / a place, a label was swallowed
LEAK_STEMS = tuple(ar_key(w) for w in ("جنسيت", "المولود", "حرفت", "مهنت", "موافق", "موفق", "الساكن", "توقيع"))
EMPLOYEE_JOB = re.compile(r"\s+(?:ال)?موظف[ةه]?\b.*$")
DATE_FIELDS = {
    "DATENAISSAR", "DATENAISSFR", "DATENAISSPEREAR", "DATENAISSPEREFR",
    "DATENAISSMEREAR", "DATENAISSMEREFR", "DATEDECLAAR", "DATEDECLAFR",
}


def has_label_word(text):
    """True when a word of the text starts like a label of the form ("جنسيتها", "المولودة", ...)."""
    return any(token.startswith(stem) for token in ar_key(text).split() for stem in LEAK_STEMS)


def date_is_valid(text, hijri):
    """The written date converts to a real date (or a year) of the right calendar."""
    match = re.fullmatch(r"(\d{4})(?:-(\d{2})-(\d{2}))?", str(replace_aliases(text)))
    if not match:
        return False
    year = int(match[1])
    return 1200 <= year <= 1500 if hijri else 1800 <= year <= 2100


def same_date(a, b):
    """Same converted date, or one of them is only the year of the other (the NER may miss day / month)."""
    a, b = str(replace_aliases(a)), str(replace_aliases(b))
    return a == b or a[:4] == b[:4] and (len(a) == 4 or len(b) == 4)


def pick_name(names, rules_value, exclude):
    """The NER name that matches the value of the rules, else the first one that is not in `exclude`."""
    if rules_value:
        match = next((n for n in names if agrees(rules_value, [n])), None)
        if match:
            return match
    return next((n for n in names if ar_key(n) not in exclude), "")


def ner_candidates(parsed, zones, entities, flat):
    """Second extractor: the value of every field according to the NER entities -> {field: text}.

    Nothing is written here; reconcile() compares these candidates with the values read by the rules.
    The entities are used by zone (child / father / mother / administration) and by position, never by
    label, so a typo in a label does not matter.
    """
    child, father, mother, admin = zones
    out = {}

    # --- child -------------------------------------------------------------------------------
    ents = entities_in(child, entities)
    person_runs = group_runs([e for e in ents if e.kind == PERSON], flat)
    # the model sometimes glues the sex word to the name ("سارة أنثى")
    names = [n for n in (re.sub(r"(?<!\S)(?:ذكر|[أا]نثى)(?!\S)", " ", flat[s:e]).strip() for s, e in person_runs) if n]
    names = [normalize_text(n) for n in names]
    first = pick_name(names, parsed["PRENOMAR"], {ar_key(parsed["NOMAR"])} - {""})
    out["PRENOMAR"] = first
    out["NOMAR"] = pick_name(names, parsed["NOMAR"], ({ar_key(parsed["PRENOMAR"] or first)} - {""}))
    # the birth place is written before the names
    first_person = person_runs[0][0] if person_runs else None
    out["LIEUNAISS"] = first_place(ents, flat, before=first_person)
    phrases = date_phrases([e for e in ents if e.kind == DATE], flat)
    out["DATENAISSAR"] = next((p for k, p in phrases if k == "hijri"), "")
    out["DATENAISSFR"] = next((p for k, p in phrases if k == "miladi"), "")

    # --- parents -----------------------------------------------------------------------------
    for zone, prefix in ((father, "PERE"), (mother, "MERE")):
        ents = entities_in(zone, entities)
        runs = group_runs([e for e in ents if e.kind == PERSON], flat, LINEAGE_WORDS)
        first_name, lineage, after = "", "", None
        if runs:
            first_name, lineage = split_lineage(flat[runs[0][0]:runs[0][1]])
            after = runs[0][1]
        out[f"PRENOM{prefix}AR"] = first_name
        out[f"NOM{prefix}AR"] = lineage
        out[f"ADRESSENAISS{prefix}"] = strip_attached_b(first_place(ents, flat, after=after))
        phrases = date_phrases([e for e in ents if e.kind == DATE], flat)
        out[f"DATENAISS{prefix}AR"] = next((p for k, p in phrases if k == "hijri"), "")
        out[f"DATENAISS{prefix}FR"] = next((p for k, p in phrases if k == "miladi"), "")

    # --- administration ----------------------------------------------------------------------
    ents = entities_in(admin, entities)
    # the declarant's age / address come before the declaration: only look after the last "الساكن"
    sakin = list(re.finditer(r"(?<!\S)الساكن\w*", flat[admin.start:admin.end]))
    limit = admin.start + sakin[-1].end() if sakin else admin.start
    date_ents = [e for e in ents if e.kind == DATE and e.start >= limit]
    phrases = date_phrases(date_ents, flat)
    out["DATEDECLAAR"] = next((p for k, p in reversed(phrases) if k == "hijri"), "")
    out["DATEDECLAFR"] = next((p for k, p in reversed(phrases) if k == "miladi"), "")
    # the employee signs after the declaration dates, the declarant (the parent) is named before them
    runs = group_runs([e for e in ents if e.kind == PERSON], flat)
    last_date = max((e.end for e in date_ents), default=0)
    employee = [flat[s:e] for s, e in runs if s >= last_date] or [flat[s:e] for s, e in runs]
    out["EMPLOYECREANT"] = employee[-1] if employee else ""
    return out


def reconcile(parsed, candidates, notes):
    """Compare the value of the rules with the NER candidate of each field and decide.

    - rules empty                      -> NER value
    - dates: only one of them converts to a valid date (right calendar) -> the valid one;
             both valid but different -> rules kept, difference noted
    - other fields: same text or one inside the other -> rules; rules = NER + extra words that are label
             words ("جنسيتها" ...) -> the label was swallowed, NER value; other differences -> noted only
    Every decision except "they agree" is added to `notes`.
    """
    for field, candidate in candidates.items():
        candidate = normalize_text(candidate) if candidate else ""
        value = parsed[field]
        if not candidate:
            continue
        if not value:
            parsed[field] = candidate
            notes.append(f"{field}: empty after the rules, filled from NER: {candidate!r}")
        elif field in DATE_FIELDS:
            hijri = field.endswith("AR")
            rules_ok, ner_ok = date_is_valid(value, hijri), date_is_valid(candidate, hijri)
            if ner_ok and not rules_ok:
                parsed[field] = candidate
                notes.append(f"{field}: rules read {value!r}, not a valid date; replaced by NER: {candidate!r}")
            elif rules_ok and ner_ok and not same_date(value, candidate):
                notes.append(f"{field}: rules read {value!r}, NER reads {candidate!r} (both valid, rules kept)")
            elif not rules_ok and not ner_ok:
                notes.append(f"{field}: neither the rules ({value!r}) nor NER ({candidate!r}) give a valid date")
        else:
            # the employee is written with his job ("فؤاد السعدي موظف جماعي"): compare the name only
            compared = EMPLOYEE_JOB.sub("", value) if field == "EMPLOYECREANT" else value
            if ar_key(compared) == ar_key(candidate):
                continue
            if longer_than(compared, candidate):
                tail = " ".join(compared.split()[len(candidate.split()):])
                if has_label_word(tail):
                    parsed[field] = candidate
                    notes.append(f"{field}: rules read {value!r}, a label was swallowed ({tail!r}); NER value kept: {candidate!r}")
                else:
                    notes.append(f"{field}: rules read {value!r}, NER stops at {candidate!r} (label swallowed?)")
            elif not agrees(compared, [candidate]):
                notes.append(f"{field}: rules read {value!r}, NER says {candidate!r}")


def fill_french_names(parsed, french, notes):
    """PRENOM / NOM from the French PERSON tokens when the labels were not read (heuristic order)."""
    prenom, nom = parsed["PRENOM"], parsed["NOM"]
    if not french or (prenom and nom):
        return
    if not prenom and not nom:
        # PRENOM first, NOM after: the Arabic first name tells how many words are the PRENOM
        count = max(1, len(parsed["PRENOMAR"].split()))
        prenom, nom = " ".join(french[:count]), " ".join(french[count:])
    elif not prenom:
        prenom = " ".join(t for t in french if t not in nom.split())
    else:
        nom = " ".join(t for t in french if t not in prenom.split())
    for field, value in (("PRENOM", prenom), ("NOM", nom)):
        if value and not parsed[field]:
            parsed[field] = value
            notes.append(f"{field}: empty after the rules, filled from French NER (heuristic order): {value!r}")


def apply_ner(parsed, zones, entities, french, flat, notes):
    """NER as a second extractor: candidates for every field, then arbitration with the rules."""
    reconcile(parsed, ner_candidates(parsed, zones, entities, flat), notes)
    fill_french_names(parsed, french, notes)


def find_zones(txt):
    """Cleaned text -> (child, father, mother, administration) zones, cut with the fixed labels of the form."""
    # 1. extract child info from the first في يوم to من والدها/والده
    child_zone = find_zone([
        r"(?:^|\s)(في\s+يوم\s+.*?)(?=\s*(?<!\S)" + FATHER_MARK + r"(?:\s|$))",
        r"(?:^|\s)(في\s+يوم\s+.*?)(?=\s*(?<!\S)من\s+والد|\s*(?<!\S)" + MOTHER_MARK + r"(?:\s|$)|\s*(?<!\S)الساكن|\s*$)",
    ], txt)
    # 2. extract father info from من والدها/والده to و والدته/و والدتها
    father_zone = find_zone([
        FATHER_MARK + r"\s+(.*?)(?=\s*(?<!\S)" + MOTHER_MARK + r"(?:\s|$)|\s*(?<!\S)الساكن|\s*$)"
    ], txt)
    # 3. extract mother info from و والدته/و والدتها to الساكن/الساكنان
    mother_zone = find_zone([
        r"(?:^|\s)" + MOTHER_MARK + r"\s+(.*?)(?=\s*(?<!\S)الساكن|\s*(?<!\S)بناء[اأ]?\s+على|\s*$)"
    ], txt)
    # 4. extract administrative info from الساكن/الساكنان to توقيع المصرح
    administrative_zone = find_zone([
        r"(?:^|\s)(الساكن\w*\s+.*?)(?=\s*(?<!\S)توقيع\s*المصرح|\s*$)",
        r"(?:^|\s)(بناء[اأ]?\s+على.*?)(?=\s*(?<!\S)توقيع\s*المصرح|\s*$)",
    ], txt)
    return child_zone, father_zone, mother_zone, administrative_zone


# ---------------------------------------------------------------------------
# line order: the HTR does not always read the lines in the order of the form
# ---------------------------------------------------------------------------
CHILD, FATHER, MOTHER, ADDRESS, DECLARATION = range(5)
# the kind of a line, from its first words (tried in this order): (kind, pattern, zone, rank in the zone).
# zone None: a line every zone has (date, nationality), its zone comes from where it is (see restore_order).
LINE_KINDS = [
    ("number", r"^رسم\s+(?:ال)?ولادة", CHILD, 0),
    ("day", r"^في\s+يوم", CHILD, 1),
    ("written", r"^و\s*حر\w{0,3}\s+في(?:\s|$)", DECLARATION, 2),
    ("by", r"^من\s+طرفنا", DECLARATION, 4),
    ("read", r"^و\s*بعد\s+تلاوة", DECLARATION, 5),
    ("delegation", r"^بالتفويض", DECLARATION, 6),
    ("signature", r"^توقيع", DECLARATION, 7),
    ("declaration", r"^بناء[اأ]?\s+على", DECLARATION, 0),
    ("father", r"^من\s+والد\s*(?:ه|ها)(?:\s|$)", FATHER, 0),
    ("mother", r"^(?:و\s*)?والدت\s*(?:ه|ها)(?:\s|$)", MOTHER, 0),
    ("hour", r"^على\s+الساعة", CHILD, 4),
    ("minute", r"^(?:و\s*)?الدقيقة", CHILD, 5),
    ("born", r"^(?:ولد|ولدت|والدت?)(?:\s|$)", CHILD, 6),
    ("first name", r"^ال[إأا]سم\s+الشخصي", CHILD, 7),
    ("prenom", r"^PRENOM(?:\s|$)", CHILD, 7),
    ("sex", r"^(?:ذكر|[أا]نثى)$", CHILD, 7),
    ("family name", r"^ال[إأا]سم\s+العائلي", CHILD, 8),
    ("nom", r"^NOM(?:\s|$)", CHILD, 8),
    # "حرفته أجير جنسيته مغربية الساكن ..." is the declaration going on, not the father's job
    ("declared", r"^(?:سنة\s+)?(?:حرفت|مهنت)\S*\s.*(?:جنسيت|الساكن)", DECLARATION, 1),
    ("father job", r"^(?:حرفته|مهنته)(?:\s|$)", FATHER, 4),
    ("mother job", r"^(?:حرفتها|مهنتها)(?:\s|$)", MOTHER, 4),
    ("father born", r"^المولود(?:\s|$)", FATHER, 1),
    ("mother born", r"^المولودة(?:\s|$)", MOTHER, 1),
    ("address", r"^الساكن", ADDRESS, 0),
    # the end of the address on a line of its own ("الشطر الأول الرقم: 57", "زنقة رام الله الرقم : 31")
    ("address end", r"^(?:ال)?(?:زنقة|رقم|شارع|درب|تجزئة|[إا]قامة|عمارة|شطر|مجموعة)(?:\s|$)", ADDRESS, 1),
    ("nationality", r"^جنسيت", None, None),
    ("gregorian", r"^(?:ال)?موا?فق(?:\s|$)", None, None),
    ("hijri", r"^(?:في|عام)(?:\s|$)", None, None),
]
LINE_PATTERNS = [(kind, re.compile(pattern, re.IGNORECASE), zone, rank)
                 for kind, pattern, zone, rank in LINE_KINDS]
# the lines every zone has: how many of them, and their rank in the zone
SHARED = {
    "nationality": {CHILD: (1, 7), FATHER: (1, 1), MOTHER: (1, 1)},
    "gregorian": {CHILD: (1, 3), FATHER: (1, 3), MOTHER: (1, 3), DECLARATION: (1, 3)},
    "hijri": {CHILD: (1, 2), FATHER: (1, 2), MOTHER: (1, 2)},
}


def line_kind(line):
    """'من والده نبيل ...' -> ('father', FATHER, 0); a line without a label of the form -> (None, None, None)."""
    flat = flatten(line)[0]
    for kind, pattern, zone, rank in LINE_PATTERNS:
        if pattern.search(flat):
            return kind, zone, rank
    return None, None, None


# the end of a date on a line of its own ("ألفين و ستة ميلادية"): it goes with the date line next to it
DATE_ENDS = (("ميلا", ("gregorian",)), ("هجر", ("hijri", "day", "written")))


def date_owners(kinds, lines):
    """{line: date line} for the unlabeled lines that end a date, joined to the nearest date line of that kind
    (at most two lines away, the one before first)."""
    owners = {}
    for i, line in enumerate(lines):
        if kinds[i] is not None:
            continue
        flat = flatten(line)[0]
        for word, owner_kinds in DATE_ENDS:
            if re.search(r"(?<!\S)" + word, flat):
                near = [j for j in (i - 1, i + 1, i - 2, i + 2) if 0 <= j < len(lines) and kinds[j] in owner_kinds]
                if near:
                    owners[i] = near[0]
                break
    return owners


def nationality_gender(line):
    """'m' for "جنسيته", 'f' for "جنسيتها", None when the label is damaged."""
    flat = flatten(line)[0]
    return "f" if re.match(r"جنسيتها", flat) else "m" if re.match(r"جنسيته(?!ا)", flat) else None


def child_gender(found, lines):
    """'f' / 'm' from the sex line of the child (None when there is none)."""
    for (kind, _, _), line in zip(found, lines):
        if kind == "sex":
            return "f" if re.search(r"[أا]نثى", line) else "m"
    return None


def date_year(text, hijri):
    """The year of a date line as a number ("عام ألف و أربعمائة ..." / "سنة ألفين و ستة ...") or None."""
    flat = flatten(text)[0]
    match = re.search(r"(?<!\S)(?:عام|ل?سنة)\s+(.+?)(?=\s+(?:هجر|ميلا)|$)", flat)
    if not match:
        return None
    from converter import get_number_keys, phrase_to_number
    year = phrase_to_number(match.group(1), get_number_keys())
    if year is None:
        return None
    if hijri:
        return year * 0.970224 + 621.5744 if 1250 <= year <= 1500 else None
    return year if 1850 <= year <= 2100 else None


def pair_dates(groups, lines):
    """A Hijri date and the Gregorian date of the same year go together: when they were put in two parts (one of
    the lines was read far from its place), the moved one joins the part that misses it."""
    hijri = [g for g in groups if g[3] in ("hijri", "written")]
    gregorian = [g for g in groups if g[3] == "gregorian"]
    years = {id(g): date_year(" ".join(lines[i] for i in g[2]), g[3] != "gregorian") for g in hijri + gregorian}
    pairs = []
    for h in hijri:
        for g in gregorian:
            if years[id(h)] is not None and years[id(g)] is not None and abs(years[id(h)] - years[id(g)]) <= 1.2:
                pairs.append((abs(h[2][0] - g[2][0]), h, g))
    used, chosen = set(), []
    for _, h, g in sorted(pairs, key=lambda p: (p[0], p[1][2][0])):
        if id(h) not in used and id(g) not in used:
            used.update((id(h), id(g)))
            chosen.append((h, g))
    # the years cannot tell (a year misread): a Gregorian date goes with the nearest Hijri one before it that has
    # none, else the nearest after it
    for g in gregorian:
        if id(g) in used:
            continue
        free = [h for h in hijri if id(h) not in used]
        before = [h for h in free if h[2][0] < g[2][0]]
        after = [h for h in free if h[2][0] > g[2][0]]
        h = max(before, key=lambda x: x[2][0]) if before else min(after, key=lambda x: x[2][0]) if after else None
        if h is not None:
            used.update((id(h), id(g)))
            chosen.append((h, g))

    # the parents' dates: the first pair is the father's, the second the mother's (whatever part each line was
    # read in)
    parents = sorted(((h, g) for h, g in chosen if h[3] == "hijri" and h[0] != CHILD),
                     key=lambda pair: min(pair[0][2][0], pair[1][2][0]))
    if len(parents) == 2:
        for zone, (h, g) in zip((FATHER, MOTHER), parents):
            h[0], h[1] = zone, SHARED["hijri"][zone][1]
            g[0], g[1] = zone, SHARED["gregorian"][zone][1]

    def has(kinds, zone, but):
        return any(x is not but and x[0] == zone and x[3] in kinds for x in groups)

    # a move can free the place another pair needs: again until nothing moves
    for _ in range(len(chosen)):
        changed = False
        for h, g in chosen:
            if h[0] == g[0]:
                continue
            if not has(("gregorian",), h[0], g) and h[0] in SHARED["gregorian"]:
                g[0], g[1] = h[0], SHARED["gregorian"][h[0]][1]
                changed = True
            elif not has(("hijri", "written"), g[0], h) and g[0] in SHARED["hijri"]:
                h[0], h[1] = g[0], SHARED["hijri"][g[0]][1]
                changed = True
        if not changed:
            break


def restore_order(raw):
    """Put the lines of the form back in its order: (text, moved lines, {old line: new line}).

    Every line is recognized by its label: a label of one part of the form (child, father, mother, address,
    declaration) takes the line to that part; a line without a label stays with the line before it (the value
    goes on). A date or a nationality line belongs to the part it is in, or to the next one when that part has it
    already (a nationality also by its gender, a Gregorian date by the year of the Hijri one). In each part the
    lines are sorted by the order of the form; lines whose order does not matter keep theirs, so a text in the
    right order is not changed."""
    lines = raw.split("\n")
    found = [line_kind(line) for line in lines]
    kinds = [kind for kind, _, _ in found]
    owners = date_owners(kinds, lines)
    sex = child_gender(found, lines)
    groups = []          # [zone, rank, [line indexes], kind]
    group_of = {}        # line -> its group
    current = CHILD      # the furthest part reached
    shared_count = {}
    top_rank = {}        # zone -> highest rank of a labeled line in it

    def next_kind(i):
        return next((kinds[j] for j in range(i + 1, len(lines)) if kinds[j]), None)

    def child_nationality_late(i, gender):
        """The child's nationality read after "من والده": a boy's comes with the father's (two "جنسيته"), a
        girl's "جنسيتها" before the father's "جنسيته" (after it, it is the mother's read early)."""
        later = []
        for j in range(i + 1, len(lines)):
            if kinds[j] in ("mother", "address", "declaration"):
                break
            if kinds[j] == "nationality":
                later.append(nationality_gender(lines[j]))
        return "m" in later

    for i, line in enumerate(lines):
        kind, zone, rank = found[i]
        if i in owners:                      # the end of a date: with its date line (placed below)
            continue
        if kind is None and groups:          # no label: the value of the line before goes on
            groups[-1][2].append(i)
            group_of[i] = groups[-1]
            continue
        if kind in SHARED:
            places = SHARED[kind]
            gender = nationality_gender(line) if kind == "nationality" else None

            def free(z):
                if z not in places or shared_count.get((kind, z), 0) >= places[z][0]:
                    return False
                if kind != "nationality":
                    return True
                if (z, gender) in ((FATHER, "f"), (MOTHER, "m")):
                    return False             # "جنسيتها" is not the father's, "جنسيته" not the mother's
                # a masculine nationality after the child's family name, just before the father's lines, is his
                return not (z == CHILD and top_rank.get(CHILD, 0) > places[z][1]
                            and next_kind(i) in ("father", "father born", "father job") and gender == "m")

            first = current
            # the child's nationality read after "من والده": two of them before the mother, the child has none
            if (kind == "nationality" and current == FATHER and gender and gender == sex
                    and not shared_count.get((kind, CHILD)) and not shared_count.get((kind, FATHER))
                    and child_nationality_late(i, gender)):
                first = CHILD
            zone = next((z for z in range(first, DECLARATION + 1) if free(z)), None)
            if zone is None:                 # more of them than the form has: leave it where it is
                zone = current
                rank = groups[-1][1] if groups else 0
            else:
                shared_count[(kind, zone)] = shared_count.get((kind, zone), 0) + 1
                rank = places[zone][1]
        elif zone is None:                   # an unlabeled first line
            zone, rank = CHILD, 0
        if kind not in SHARED:               # only the labels of one part tell how far the text is
            current = max(current, zone)
        top_rank[zone] = max(top_rank.get(zone, 0), rank)
        groups.append([zone, rank, [i], kind])
        group_of[i] = groups[-1]
    for line, owner in owners.items():
        members = group_of.get(owner, groups[-1] if groups else None)
        if members is None:
            groups.append([CHILD, 0, [line], None])
        else:
            members[2].append(line)
            members[2].sort(key=lambda k: (k != owner, k))   # the date line first, then its end
    pair_dates(groups, lines)
    order = [i for _, _, members, _ in sorted(groups, key=lambda g: (g[0], g[1])) for i in members]
    new_line = {old: new for new, old in enumerate(order)}
    return "\n".join(lines[i] for i in order), moved_count(order, lines), new_line


def moved_count(order, lines):
    """How many (non-empty) lines were moved: those outside the longest run kept in their order."""
    order = [i for i in order if lines[i].strip()]
    tails = []                               # longest increasing subsequence, patience sorting
    for i in order:
        k = bisect_left(tails, i)
        tails[k:k + 1] = [i]
    return len(order) - len(tails)



def move_ner(ner, raw, new_line):
    """The NER offsets of the raw text moved with their lines (an entity over two lines that are no longer
    together loses its offsets: it is then found by its text)."""
    if not ner:
        return ner
    starts = [0]
    for line in raw.split("\n"):
        starts.append(starts[-1] + len(line) + 1)
    lengths = [starts[i + 1] - starts[i] - 1 for i in range(len(starts) - 1)]
    new_starts = [0] * len(lengths)
    position = 0
    for old in sorted(new_line, key=new_line.get):
        new_starts[old] = position
        position += lengths[old] + 1
    out = dict(ner)
    for key in ("arabic_ner", "french_ner"):
        items = []
        for item in ner.get(key) or []:
            item = dict(item)
            start, end = item.get("start"), item.get("end")
            if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(raw):
                line = bisect_left(starts, start + 1) - 1
                last = bisect_left(starts, end) - 1
                if line == last or all(new_line[k + 1] == new_line[k] + 1 for k in range(line, last)):
                    shift = new_starts[line] - starts[line]
                    item["start"], item["end"] = start + shift, end + shift
                else:
                    item.pop("start"), item.pop("end")
            items.append(item)
        out[key] = items
    return out


# 1. parse the raw txt into a json format
def parse_with_notes(raw_text, ner=None):
    """raw text + NER json (or None) -> (parsed JSON object, notes about what the NER changed)."""
    notes = []
    # the lines in the order of the form (the NER offsets move with them)
    ordered, moved, new_line = restore_order(raw_text)
    if moved:
        notes.append(f"{moved} line(s) were not in the order of the form: read in that order")
        ner = move_ner(ner, raw_text, new_line)
        raw_text = ordered
    # copy the json template
    parsed = copy.deepcopy(TEMPLATE)
    # 0. remove all punctuations and lines spaces (the position of every kept character is remembered)
    txt, index = flatten(raw_text)
    entities = build_entities(ner, raw_text, txt, index, notes)
    french = french_person_tokens(ner)
    child_zone, father_zone, mother_zone, administrative_zone = find_zones(txt)
    child = child_zone.text
    # the child's French labels can be printed inside the parent blocks
    father = strip_latin_labels(father_zone.text)
    mother = strip_latin_labels(mother_zone.text)
    administrative = administrative_zone.text
    # 5. structure the info extracted
    # Child
    parsed["NUMACTE"] = extract(r"رسم\s+(?:ال)?ولادة\s+رقم\s*(\d+)", txt)
    parsed["DATENAISSAR"] = extract(
        r"في\s+يوم\s+(.*?)(?=\s*(?<!\S)(?:ال)?موا?فق|\s*(?<!\S)على\s+الساعة)", child
    )
    parsed["DATENAISSFR"] = extract(r"(?:ال)?موا?فق\s*(.*?)(?=\s*(?<!\S)ميلا)", child)
    parsed["HEURENAISS"] = extract(
        r"على\s+الساعة\s+(.*?)(?=\s*(?<!\S)و\s*الدقيقة|\s*(?<!\S)" + BORN_VERB + r"\s)", child
    )
    parsed["MINNAISS"] = extract(
        r"و\s*الدقيقة\s*(.*?)(?=\s*(?<!\S)على\s+الساعة|\s*(?<!\S)" + BORN_VERB + r"\s|\s*(?<!\S)ال[إأا]سم|\s*$)", child
    )
    parsed["LIEUNAISS"] = extract_any([
        r"(?<!\S)" + BORN_VERB + r"\s+" + BIRTH_PREP + r"(.*?)(?=\s*(?<!\S)(?:ذكر|[أا]نثى)(?:\s|$)|\s*(?<!\S)و\s*الدقيقة|\s*(?<!\S)ال[إأا]سم\s+الشخصي|\s*(?<!\S)PRENOM\b|\s*(?<!\S)NOM\b|\s*(?<!\S)جنسيت)",
        r"(?<!\S)" + BORN_VERB + r"\s+(.*?)(?=\s*(?<!\S)(?:ذكر|[أا]نثى)(?:\s|$)|\s*(?<!\S)و\s*الدقيقة|\s*(?<!\S)ال[إأا]سم\s+الشخصي|\s*(?<!\S)PRENOM\b|\s*(?<!\S)NOM\b|\s*(?<!\S)جنسيت)",
    ], child)
    if parsed["LIEUNAISS"] in ("ب", "في"):
        parsed["LIEUNAISS"] = ""   # "ولد ب:" with nothing after it: the preposition is not a place
    parsed["PRENOMAR"] = extract(
        r"ال[إأا]سم\s+الشخصي\s+(.*?)(?=\s*(?<!\S)ذكر(?:\s|$)|\s*(?<!\S)[أا]نثى|\s*(?<!\S)PRENOM\b|\s*(?<!\S)جنسيت|\s*(?<!\S)ال[إأا]سم\s+العائلي|\s*$)",
        child,
    )
    parsed["SEXE"] = extract(r"(?<!\S)(ذكر|[أا]نثى)(?!\S)", child)
    # fallback: the verb tells the gender (ولدت = female, ولد = male)
    if not parsed["SEXE"] and child:
        if re.search(r"(?<!\S)(?:ولدت|والدت)(?!\S)", child):
            parsed["SEXE"] = "أنثى"
        elif re.search(r"(?<!\S)(?:ولد|والد)(?!\S)", child):
            parsed["SEXE"] = "ذكر"
    parsed["PRENOM"] = latin_field("PRENOM", txt) or extract(
        r"\bPRENOM\s+(.*?)(?=\s*(?<!\S)ذكر(?:\s|$)|\s*(?<!\S)[أا]نثى|\s*(?<!\S)جنسيت|\s*(?<!\S)ال[إأا]سم\s+العائلي|\s*(?<!\S)NOM\b|\s*$)", child
    )
    # the nationality label agrees with the child's gender (جنسيتها = girl, جنسيته = boy);
    # for a girl, "جنسيته" in the child zone is the father's nationality
    sex_key = ar_key(parsed["SEXE"])
    if sex_key == "انثي":
        child_nat_label = r"جنسيتها"
    elif sex_key == "ذكر":
        child_nat_label = r"جنسيته"
    else:
        child_nat_label = r"جنسيت(?:ه|ها)"
    nat_stop = (
        r"(?=\s*(?<!\S)ال[إأا]سم\s+العائلي|\s*(?<!\S)NOM\b|\s*(?<!\S)PRENOM\b|\s*(?<!\S)ذكر(?:\s|$)|\s*(?<!\S)[أا]نثى|\s*(?<!\S)جنسيت|\s*$)"
    )
    parsed["NATIONALITE"] = extract(child_nat_label + r"\s+(.+?)" + nat_stop, child)
    parsed["NOMAR"] = extract(
        r"ال[إأا]سم\s+العائلي\s+(.*?)(?=\s*(?<!\S)NOM\b|\s*(?<!\S)PRENOM\b|\s*(?<!\S)جنسيت|\s*$)", child
    )
    parsed["NOM"] = latin_field("NOM", txt) or extract(r"(?<![A-Za-z])NOM\s+(.*?)(?=\s*(?<!\S)جنسيت|\s*$)", child)
    # Father
    # zones from each parent marker to "بناء على" (used to find misplaced professions)
    father_core = extract(
        FATHER_MARK + r"\s+(.*?)(?=\s*(?<!\S)بناء[اأ]?\s+على|\s*$)", txt
    )
    mother_core = extract(
        r"(?:^|\s)" + MOTHER_MARK + r"\s+(.*?)(?=\s*(?<!\S)بناء[اأ]?\s+على|\s*$)", txt
    )
    father_info = parse_parent(father, r"جنسيته", r"المولود(?:ة)?", r"(?:حرفته|مهنته)", father_core)
    if sex_key == "انثي" and not father_info["nationality"]:
        # the father's "جنسيته" was printed in the child zone
        father_info["nationality"] = extract(r"جنسيته\s+(.+?)" + nat_stop, child)
    # a "جنسيتها" printed in the father zone belongs to the girl (if hers is missing) or to the mother
    feminine_nationality = extract(
        r"جنسيتها\s+(.*?)(?=\s*(?<!\S)المولود|\s*(?<!\S)في\s|\s*(?<!\S)حرفت|\s*(?<!\S)مهنت|\s*$)", father
    )
    used_for_child = False
    if sex_key == "انثي" and not parsed["NATIONALITE"] and feminine_nationality:
        parsed["NATIONALITE"] = feminine_nationality
        used_for_child = True
    parsed["PRENOMPEREAR"] = father_info["first"]  # Everything before بن/بنت
    parsed["NOMPEREAR"] = father_info["last"]  # بن/بنت and what comes after
    parsed["NATIONALITEPERE"] = father_info["nationality"]
    parsed["ADRESSENAISSPERE"] = strip_attached_b(father_info["birthplace"])
    parsed["DATENAISSPEREAR"] = father_info["date_ar"]
    parsed["DATENAISSPEREFR"] = father_info["date_fr"]
    parsed["PROFESSIONPERE"] = father_info["job"]
    # mother
    mother_info = parse_parent(mother, r"جنسيتها", r"المولودة", r"(?:حرفتها|مهنتها)", mother_core)
    if not mother_info["nationality"] and feminine_nationality and not used_for_child:
        mother_info["nationality"] = feminine_nationality
    parsed["PRENOMMEREAR"] = mother_info["first"]  # Everything before بن/بنت
    parsed["NOMMEREAR"] = mother_info["last"]  # بن/بنت and what comes after
    parsed["NATIONALITEMERE"] = mother_info["nationality"]
    # a masculine "جنسيته" in the mother zone: the HTR misread "جنسيتها", or the father's nationality was printed
    # there. It is the mother's when the father already has his own, otherwise it is the father's.
    if not parsed["NATIONALITEMERE"]:
        masculine = extract(
            r"(?<!\S)جنسيته\s+(.*?)(?=\s*(?<!\S)المولود|\s*(?<!\S)في\s|\s*(?<!\S)(?:ال)?موا?فق|\s*(?<!\S)حرفت|\s*(?<!\S)مهنت|\s*(?<!\S)جنسيت|\s*$)", mother
        )
        if masculine and parsed["NATIONALITEPERE"]:
            parsed["NATIONALITEMERE"] = masculine
            notes.append(f"NATIONALITEMERE: read after 'جنسيته' (masculine label) in the mother zone: {masculine!r}")
        elif masculine:
            parsed["NATIONALITEPERE"] = masculine
            notes.append(f"NATIONALITEPERE: read after 'جنسيته' printed in the mother zone: {masculine!r}")
    parsed["ADRESSENAISSMERE"] = strip_attached_b(mother_info["birthplace"])
    parsed["DATENAISSMEREAR"] = mother_info["date_ar"]
    parsed["DATENAISSMEREFR"] = mother_info["date_fr"]
    parsed["PROFESSIONMERE"] = mother_info["job"]
    # a nationality is a short known word: a damaged label must not let it swallow the rest of the line
    for field in ("NATIONALITE", "NATIONALITEPERE", "NATIONALITEMERE"):
        trimmed = trim_nationality(parsed[field])
        if trimmed != parsed[field]:
            notes.append(f"{field}: {parsed[field]!r} cut to the known nationality {trimmed!r}")
            parsed[field] = trimmed
    # some certificates do not write the child's nationality: when both parents have the same one, it is the child's
    if (not parsed["NATIONALITE"] and parsed["NATIONALITEPERE"]
            and ar_key(parsed["NATIONALITEPERE"]) == ar_key(parsed["NATIONALITEMERE"])):
        parsed["NATIONALITE"] = parsed["NATIONALITEPERE"]
        notes.append(f"NATIONALITE: not written, both parents are {parsed['NATIONALITEPERE']!r}: same for the child")
    # administartion
    parsed["ADRESSEPARENT"] = extract(
        r"الساكن\w*\s+(.*?)(?=\s*(?<!\S)بناء[اأ]?\s+على|\s*$)", administrative
    )
    # an address piece printed after the mother's profession (before "الساكنان") is the end of the address
    if mother_info["job"]:
        address_tail = extract(
            r"(?:حرفتها|مهنتها)\s+" + re.escape(mother_info["job"]) + r"\s+(.+)$", mother
        )
        if address_tail and re.match(ADDRESS_START, address_tail):
            parsed["ADRESSEPARENT"] = normalize_text(f"{parsed['ADRESSEPARENT']} {address_tail}")
    # a profession that the scan inserted inside the address is removed from it
    for label, job in (
        (r"(?:حرفتها|مهنتها)", parsed["PROFESSIONMERE"]),
        (r"(?:حرفته|مهنته)", parsed["PROFESSIONPERE"]),
    ):
        if job and parsed["ADRESSEPARENT"]:
            parsed["ADRESSEPARENT"] = normalize_text(
                re.sub(label + r"\s+" + re.escape(job) + r"(?=\s|$)", " ", parsed["ADRESSEPARENT"])
            )
    parsed["TEXTEDECLA"] = extract(
        r"(بناء[اأ]?\s+على.+?)(?=\s*(?<!\S)و\s*حر\w{0,3}\s+في|\s*$)", administrative
    )
    # the address is written twice (after "الساكنان" and inside the declaration): a difference is a typo
    # in one of them, the first one is kept and the difference is noted
    declared_address = extract(r"(?<!\S)الساكن\w*\s+((?:(?!الساكن).)+)$", parsed["TEXTEDECLA"])
    if declared_address and parsed["ADRESSEPARENT"] and ar_key(declared_address) != ar_key(parsed["ADRESSEPARENT"]):
        notes.append(
            f"ADRESSEPARENT: the header says {parsed['ADRESSEPARENT']!r}, the declaration says {declared_address!r}"
        )
    # the declaration dates are read after "وحرر في" so they are never mixed with other dates
    declaration_tail = extract(r"و\s*حر\w{0,3}\s+في\s+(.*)", administrative)
    parsed["DATEDECLAAR"] = extract(r"^(.*?)(?=\s*(?<!\S)هجر)", declaration_tail)
    parsed["DATEDECLAFR"] = extract(r"(?:ال)?موا?فق\s*(.*?)(?=\s*(?<!\S)ميلا)", declaration_tail)
    parsed["EMPLOYECREANT"] = extract(
        r"من\s+طرفنا\s+نحن\s+(.*?)(?=\s*(?<!\S)و\s*بعد\s+تلاوة|\s*(?<!\S)و\s*ضابط|\s*$)", administrative
    )
    # 6. NER as a second extractor: fill what the rules could not read, replace what they read wrongly,
    #    note every difference
    if entities or french:
        apply_ner(
            parsed, (child_zone, father_zone, mother_zone, administrative_zone),
            entities, french, txt, notes,
        )
    # the declaration year is the year of the Gregorian declaration date
    parsed["ANNEEDECLA"] = extract(r"سنة\s+(.+)$", parsed["DATEDECLAFR"])
    # return the parsed Json object
    return parsed, notes


def parse(raw_text, ner=None):
    """raw text + NER json (or None) -> parsed JSON object."""
    return parse_with_notes(raw_text, ner)[0]


def main():
    parser = argparse.ArgumentParser(
        description="Parse a Moroccan birth certificate text with the help of the NER entities."
    )
    parser.add_argument("input_file", type=Path, help="Path to the input .txt file")
    parser.add_argument("ner_file", type=Path, nargs="?", help="JSON written by classificator.py (optional)")
    parser.add_argument("-o", "--output", type=Path, help="write the JSON here instead of stdout")
    parser.add_argument("--notes", action="store_true", help="print on stderr what the NER changed / disputed")
    args = parser.parse_args()
    if not args.input_file.is_file():
        parser.error(f"Input file not found: {args.input_file}")
    if args.ner_file and not args.ner_file.is_file():
        parser.error(f"NER file not found: {args.ner_file}")
    ner = read_json_file(args.ner_file) if args.ner_file else None
    parsed, notes = parse_with_notes(read_text_file(args.input_file), ner)
    if args.notes:
        for note in notes:
            print(f"[parser] {note}", file=sys.stderr)
    write_json(parsed, args.output)


if __name__ == "__main__":
    main()