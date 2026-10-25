"""Stage 0c - spelling checker: recognized text -> same text with the HTR spelling mistakes corrected.

    python src/spelling_checker.py scan.txt                  # prints the corrected text, the changes on stderr
    python src/spelling_checker.py scan.txt -o fixed.txt
    python src/spelling_checker.py scan.txt --model qwen2.5:14b

The documents are personal data: the model runs on this computer with Ollama (https://ollama.com), nothing is
sent to an online API. The Ollama address must be this computer (localhost); MADOC_SPELL_ALLOW_REMOTE=1 allows
another address (a server of your own network, never a public service).

    ollama pull qwen2.5:7b            # once
    (Ollama runs in the background; MADOC_SPELL_MODEL / MADOC_OLLAMA_URL change the model / address)

How "only spelling" is enforced
  The model proposes a corrected version of every line. Its proposal is never taken as it is: the lines are
  compared word by word and every change is checked on its own (accept_change). A change is kept only when
    - the original words are NOT known words (a correct word is never replaced),
    - every new word IS a known word of the form: its fixed words (labels), the numbers / months / hours of
      date_time.json and dictionaries/spelling_words.txt (your own list),
    - at most 1 letter differs (2 for words of 7 letters or more): a spelling fix, not another word,
    - the words are not a value of the form: a name (after "الإسم الشخصي", "من والده", "بن", "السيدة", "نحن"),
      a place or an address (after "ب", "جماعة", "بحي", "زنقة", "المولود" ...) or a job (after "حرفته" ...),
      up to the next word of the form, also across line breaks and when the label itself is misspelt,
    - the original word has at least 3 letters (a lost fragment like "ها" is left alone),
    - no digit and no Latin letter is touched, and at most 3 words are involved.
  Names, numbers, jobs, nationalities and places are never modified: a misread name stays as the HTR read it
  (the NER, the validator and the person reviewing the record see the real reading); the jobs, nationalities
  and places already accept typos later (mapper.py fuzzy lookup), and their dictionaries hold many words
  close to personal names ("نادل", "ولد" ...), which is why they are not used here.
  Every kept and every refused change is returned as a note. Without Ollama the text is returned unchanged.
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from common import (
    ARABIC_DIACRITICS,
    BASE_DIR,
    INVISIBLE_CHARS,
    ar_key,
    dates_data,
    jobs_data,
    read_text_file,
)

OLLAMA_URL = os.environ.get("MADOC_OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = os.environ.get("MADOC_SPELL_MODEL", "qwen2.5:7b")
TIMEOUT = 600   # seconds: a 7B model on a CPU needs a few minutes for a whole certificate
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}

# the fixed words of the birth certificate form (the dictionaries hold the numbers, months, jobs, places ...)
FORM_WORDS = """
رسم ولادة رقم في يوم عام سنة لسنة هجرية ميلادية موافق الموافق على الساعة الدقيقة ولد ولدت
جماعة مدينة اقليم إقليم دائرة عمالة قيادة باشوية مقاطعة دوار قرية مركز الحضرية القروية
حي زنقة شارع درب تجزئة إقامة عمارة الشطر الرقم محج المستشفى مستشفى مصحة
الإسم الاسم الشخصي العائلي ذكر أنثى جنسيته جنسيتها مغربية مغربي
من والده والدها والدته والدتها والد والدت المولود المولودة بن بنت ابن ابنة
حرفته حرفتها مهنته مهنتها ربة البيت بيت الساكنان الساكن الساكنة
بناء بناءا ما صرح صرحت به السيد السيدة تحت عدد عمره عمرها وحرر حرر
طرفنا نحن موظف موظفة جماعي جماعية بعد تلاوة الرسم المصرح وقعه وقعناه معنا وحدنا ضابط الحالة المدنية
بالتفويض توقيع عشري تاني تالت
"""
# "والد" / "والدت" are kept as they are: the scan often cuts "والد" + "ها" over two lines, the parser expects it
# one-letter prefixes glued to a word: و (and), ب (in), ل (to), ف
PREFIXES = "وبلف"
MAX_WORDS = 3          # a change may involve at most this many words on each side
# words after which a value of the form is written (a name, a place / an address, a job): the following words
# are protected up to the next word of the form (compared without a glued و / ب and without the article)
VALUE_LABELS = {
    # names
    "الشخصي", "العائلي", "والده", "والدها", "والدته", "والدتها", "والد", "والدت",
    "بن", "بنت", "ابن", "ابنة", "السيد", "السيدة", "نحن",
    # places and addresses
    "ب", "جماعة", "مدينة", "اقليم", "إقليم", "أقليم", "دائرة", "عمالة", "قيادة", "باشوية", "مقاطعة", "دوار",
    "قرية", "مركز", "حي", "زنقة", "شارع", "درب", "تجزئة", "إقامة", "اقامة", "عمارة", "الشطر", "محج",
    "المولود", "المولودة", "الساكن", "الساكنان", "الساكنة",
}
# a job is a value too, but it is checked against jobs.json (a closed list): it can be corrected
JOB_LABELS = {"حرفته", "حرفتها", "مهنته", "مهنتها"}
VALUE_LABELS |= JOB_LABELS
MIN_LETTERS = 3        # shorter words (lost fragments, prepositions) are never corrected
PUNCTUATION = ".:،؛٪()-«»\"'"
# letters the parser treats as the same (dictionary keys): a change between them is not a correction
SAME_LETTERS = str.maketrans({"ڭ": "ك", "گ": "ك", "أ": "ا", "إ": "ا", "آ": "ا"})
SPLIT_PUNCTUATION = re.compile("[" + re.escape(PUNCTUATION) + "]+")

ARABIC_WORD = re.compile("[؀-ۿ]")
DIGIT_OR_LATIN = re.compile(r"[0-9A-Za-z٠-٩۰-۹]")


def log(*args):
    print(*args, file=sys.stderr)


# ---------------------------------------------------------------------------------------------------------
# vocabulary: the words a correction may produce
# ---------------------------------------------------------------------------------------------------------
def form_vocabulary():
    words = set()
    for word in FORM_WORDS.split():
        words.update({word, word[2:] if word.startswith("ال") and len(word) > 3 else "ال" + word})
    return frozenset(words)


def surface(token):
    """A word as written, without diacritics / tatweel / invisible characters.

    Unlike ar_key the letters are NOT unified: "جنسيتة" and "جنسيته" are different words here, because the
    parser needs the right letter ("ة" / "ه" / "ى" / hamza are exactly what the HTR confuses).
    """
    return ARABIC_DIACRITICS.sub("", INVISIBLE_CHARS.sub("", token)).strip()


FORM_VOCABULARY = form_vocabulary()


@lru_cache(maxsize=None)
def get_vocabulary():
    """Every word a correction may produce, as written: the form words, date_time.json (numbers, months,
    hours) and dictionaries/spelling_words.txt, with and without the article "ال"."""
    words = set()

    def add(text):
        if isinstance(text, str):
            for word in surface(text).split():
                words.add(word)
                words.add(word[2:] if word.startswith("ال") and len(word) > 3 else "ال" + word)

    def walk(obj):
        if isinstance(obj, dict):
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)
        else:
            add(obj)

    add(FORM_WORDS)
    walk(dates_data)
    # your own words (other form words, neighbourhoods, streets ...), several per line, "#" starts a comment
    extra = Path(BASE_DIR) / "spelling_words.txt"
    if extra.is_file():
        for line in extra.read_text(encoding="utf-8").splitlines():
            add(line.split("#", 1)[0])
    words.discard("")
    return frozenset(words)


def is_known(token, vocabulary=None):
    """A known word, also with a glued prefix ("بجماعة", "والدته", "لسنة")."""
    vocabulary = vocabulary if vocabulary is not None else get_vocabulary()
    word = surface(token).strip(PUNCTUATION)
    if not word:
        return False
    if word in vocabulary:
        return True
    return len(word) > 2 and word[0] in PREFIXES and word[1:] in vocabulary


# ---------------------------------------------------------------------------------------------------------
# guard: which proposed changes are spelling corrections
# ---------------------------------------------------------------------------------------------------------
def edit_distance(a, b):
    """Number of letters to insert, delete or replace to turn a into b (Levenshtein)."""
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_a != char_b)))
        previous = current
    return previous[-1]


@lru_cache(maxsize=4096)
def label_kind(word):
    """"job" after a job label, "value" after a name / place / address label, else None. The label is also
    recognised with a glued و / ب, the article, or one wrong letter ("الشحصي", "حرفتة")."""
    forms = {word, word[2:] if word.startswith("ال") else "ال" + word}
    if len(word) > 1 and word[0] in PREFIXES:
        forms.add(word[1:])

    def near(labels):
        return bool(forms & labels) or len(word) >= 5 and any(
            len(label) >= 5 and abs(len(label) - len(form)) <= 1 and edit_distance(form, label) <= 1
            for form in forms for label in labels
        )

    if near(JOB_LABELS):
        return "job"
    if near(VALUE_LABELS):
        return "value"
    return None


def protected_positions(tokens, vocabulary=None, inside=None):
    """The words that are a value: after a value label, up to the next word of the form.

    -> ({index: "job" | "value"}, state at the end of the line). `inside` carries the state (None, "job" or
    "value") from the previous line, a label is often at the end of a line and its value on the next one.
    """
    protected = {}
    for i, token in enumerate(tokens):
        word = surface(token).strip(PUNCTUATION)
        if not word:
            continue
        kind = label_kind(word)
        if kind:
            inside = kind
            continue
        if DIGIT_OR_LATIN.search(word) or is_known(word, vocabulary):
            inside = None
            continue
        if inside:
            protected[i] = inside
    return protected, inside


# ---------------------------------------------------------------------------------------------------------
# jobs: corrected only towards jobs.json
# ---------------------------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def get_jobs():
    """The jobs of jobs.json as written (diacritics removed) -> their words."""
    return {surface(job): tuple(surface(job).split()) for job in jobs_data if surface(job)}


@lru_cache(maxsize=None)
def job_keys():
    """The jobs as the parser compares them (unit_key of every word) -> True."""
    return {" ".join(map(unit_key, words)): True for words in get_jobs().values()}


def job_distance(old_words, job_words):
    old, new = "".join(map(unit_key, old_words)), "".join(map(unit_key, job_words))
    return edit_distance(old, new), len(new)


def job_limit(length, relaxed=False):
    """Letters that may differ between a reading and a job: 1 (2 for 7 letters or more); one more when the
    same job is read elsewhere in the document (relaxed)."""
    return (1 if length < 7 else 2) + (1 if relaxed else 0)


def exact_job(words):
    """The job of jobs.json that the start of `words` already is (ڭ / hamza differences do not count), or None."""
    keys = job_keys()
    for n in range(min(len(words), 5), 0, -1):
        if keys.get(" ".join(map(unit_key, words[:n]))):
            return " ".join(words[:n])
    return None


def job_candidates(words, relaxed=False):
    """Jobs of jobs.json that `words` (the start of a job slot) may be a misreading of.

    -> (number of words of the slot used, [jobs, closest first]) or (0, []). A job already written right gives
    nothing. Every job at the best distance is returned ("صباخ" -> "صباغ" painter and "طباخ" cook): the
    person chooses, or the other mention of the same job in the document decides (job_changes).
    """
    if exact_job(words):
        return 0, []
    jobs = get_jobs()
    best = (0, [], None)
    for n in range(min(len(words), 5), 0, -1):
        part = words[:n]
        if len("".join(part)) < MIN_LETTERS:
            continue
        found = []
        for job, job_words in jobs.items():
            if abs(len(job_words) - n) > 1:
                continue
            distance, length = job_distance(part, job_words)
            if distance <= job_limit(length, relaxed):
                found.append((distance, job))
        if found:
            distance = min(d for d, _ in found)
            if best[2] is None or distance < best[2]:
                best = (n, sorted(job for d, job in found if d == distance), distance)
    return best[0], best[1]


def distance_to_job(words, job):
    """(distance, words used) of the closest start of `words` to this job."""
    job_words = get_jobs().get(surface(job), tuple(job.split()))
    best = None
    for n in range(1, min(len(words), 5) + 1):
        distance, length = job_distance(words[:n], job_words)
        if best is None or distance < best[0]:
            best = (distance, n, length)
    return best


def job_slots(units, protected):
    """[(start, end)] of the job values of a line (consecutive units protected as "job")."""
    slots, start = [], None
    for i in range(len(units) + 1):
        is_job = protected.get(i) == "job"
        if is_job and start is None:
            start = i
        elif not is_job and start is not None:
            slots.append((start, i))
            start = None
    return slots


# the declarant is a parent: "(والده)" / "(والدها)" = the father, "(والدته)" / "(والدتها)" = the mother
DECLARANT_PARENT = re.compile(r"\(\s*(والده|والدها|والدته|والدتها)\s*\)")
DECLARATION_START = re.compile(r"^بناء[اأ]?$")


def find_job_slots(lines, protected_by_line):
    """Every job value of the document: {"line", "start", "end", "words", "gender" ("m" after حرفته / مهنته,
    "f" after حرفتها / مهنتها), "declaration" (True when it is in the declaration, after "بناء على")}."""
    slots, previous_label, in_declaration = [], None, False
    for line_index, line in enumerate(lines):
        units = line_units(line)
        protected = protected_by_line.get(line_index, {})
        starts = {start: end for start, end in job_slots(units, protected)}
        for i, unit in enumerate(units):
            word = surface(unit["text"])
            if DECLARATION_START.match(word):
                in_declaration = True
            if label_kind(word) == "job":
                previous_label = word
            if i in starts:
                label = surface(units[i - 1]["text"]) if i > 0 and label_kind(surface(units[i - 1]["text"])) else previous_label
                slots.append({
                    "line": line_index, "start": i, "end": starts[i],
                    "words": [u["text"] for u in units[i:starts[i]]],
                    "gender": "f" if (label or "").endswith("ها") else "m",
                    "declaration": in_declaration,
                })
    return slots


def job_change(slot, options, accepted, reason, context=""):
    count = max(distance_to_job(slot["words"], options[0])[1], 1) if options else len(slot["words"])
    return {
        "line": slot["line"], "start": slot["start"], "end": slot["start"] + count, "old": slot["words"][:count],
        "new": options[0].split(), "options": [o.split() for o in options], "accepted": accepted,
        "reason": reason, "source": "jobs.json", "context": context,
    }


def model_jobs_for(slot, model_changes, relaxed=False):
    """The jobs the model proposed inside a slot that are close enough to what is written."""
    jobs = []
    for change in model_changes:
        if change["line"] == slot["line"] and slot["start"] <= change["start"] < slot["end"] and change["new"]:
            job = " ".join(change["new"])
            if surface(job) in get_jobs():
                distance, _, length = distance_to_job(slot["words"], job)
                if distance <= job_limit(length, relaxed):
                    jobs.append(job)
    return jobs


def job_changes(text, lines, protected_by_line, model_changes):
    """Changes for the misread jobs of the whole document (the model's changes inside a job are replaced).

    The same person's job is written twice when the declarant is a parent: in the parent's block and in the
    declaration ("... (والدها) ... حرفته جاغ"). Those mentions are read together:
      - one of them is a job of jobs.json: it is proposed for the other one (2 letters may differ),
      - none is: the jobs possible for every mention (2 letters) are intersected, one left -> applied,
      - otherwise every mention is handled alone (1 letter, several jobs -> the person chooses).
    The model's job comes first among equal choices, but it is never taken without being close to the reading.
    """
    slots = find_job_slots(lines, protected_by_line)
    handled, changes, ids = set(), [], {}
    for k, slot in enumerate(slots):
        ids[k] = slot
    relation = DECLARANT_PARENT.search(text.replace("\n", " "))
    groups = []
    if relation:
        gender = "m" if relation.group(1) in ("والده", "والدها") else "f"
        group = [k for k, slot in ids.items() if slot["gender"] == gender]
        if len(group) >= 2 and any(ids[k]["declaration"] for k in group) and any(not ids[k]["declaration"] for k in group):
            groups.append(group)
    for group in groups:
        readings = {" ".join(map(unit_key, ids[k]["words"])) for k in group}
        if len(readings) == 1 and not exact_job(ids[group[0]]["words"]):
            # every mention reads the same word: it is right, only missing from jobs.json ("حداد")
            handled.update(group)
            continue
        exact = {k: exact_job(ids[k]["words"]) for k in group}
        known = {job for job in exact.values() if job}
        where = lambda k: f"line {ids[k]['line'] + 1}"
        if len(known) == 1:
            job = known.pop()
            source = next(k for k in group if exact[k])
            for k in group:
                if exact[k]:
                    continue
                distance, _, length = distance_to_job(ids[k]["words"], job)
                close = distance <= job_limit(length, relaxed=True)
                changes.append(job_change(ids[k], [job], close, "" if close else
                                          f"too different from {job!r} read on {where(source)}",
                                          f"same job as {where(source)}"))
            handled.update(group)
        elif not known:
            possible = []
            for k in group:
                _, jobs = job_candidates(ids[k]["words"], relaxed=True)
                possible.append(set(jobs) | set(model_jobs_for(ids[k], model_changes, relaxed=True)))
            common = set.intersection(*possible) if possible else set()
            if common:
                preferred = [j for k in group for j in model_jobs_for(ids[k], model_changes, True) if j in common]
                options = list(dict.fromkeys(preferred + sorted(common)))
                single = len(options) == 1
                others = ", ".join(where(k) for k in group)
                for k in group:
                    changes.append(job_change(ids[k], options, single,
                                              "" if single else "several jobs are possible: " + " / ".join(options),
                                              f"the same job is read on {others}"))
                handled.update(group)
    for k, slot in ids.items():
        if k in handled:
            continue
        _, candidates = job_candidates(slot["words"])
        options = list(dict.fromkeys(model_jobs_for(slot, model_changes) + candidates))
        if options:
            single = len(options) == 1
            changes.append(job_change(slot, options, single,
                                      "" if single else "several jobs are possible: " + " / ".join(options)))
            handled.add(k)
    # a job written right is never changed: the model's proposals there are dropped, not shown
    for k, slot in ids.items():
        if exact_job(slot["words"]):
            handled.add(k)
    # the model's own changes inside every job are replaced by these (or refused by line_changes)
    for k in handled:
        slot = ids[k]
        for change in [c for c in model_changes
                       if c["line"] == slot["line"] and slot["start"] <= c["start"] < slot["end"]]:
            model_changes.remove(change)
    return changes


def accept_change(old_tokens, new_tokens, vocabulary=None, max_distance=None):
    """(True, "") when replacing old_tokens by new_tokens is a spelling correction, else (False, reason).

    max_distance: letters that may differ (default 1, 2 for words of 7 letters or more).
    """
    if not old_tokens or not new_tokens:
        return False, "a word was added or removed"
    if len(old_tokens) > MAX_WORDS or len(new_tokens) > MAX_WORDS:
        return False, "too many words changed at once"
    if any(DIGIT_OR_LATIN.search(t) for t in old_tokens + new_tokens):
        return False, "digits and Latin letters are never changed"
    if not all(ARABIC_WORD.search(t) for t in new_tokens):
        return False, "the new text is not Arabic words"
    if all(is_known(t, vocabulary) for t in old_tokens):
        return False, "the original words are already correct words"
    if len("".join(surface(t).strip(PUNCTUATION) for t in old_tokens)) < MIN_LETTERS:
        return False, "too short to be corrected safely"
    unknown = [t for t in new_tokens if not is_known(t, vocabulary)]
    if unknown:
        return False, f"{' '.join(unknown)!r} is not a known word (names are never changed)"
    old = "".join(surface(t).strip(PUNCTUATION) for t in old_tokens)
    new = "".join(surface(t).strip(PUNCTUATION) for t in new_tokens)
    allowed = max_distance if max_distance is not None else (1 if len(new) < 7 else 2)
    if edit_distance(old, new) > allowed:
        return False, "too different to be a spelling mistake"
    return True, ""


# ---------------------------------------------------------------------------------------------------------
# comparing a line with the model's version
# ---------------------------------------------------------------------------------------------------------
def line_units(line):
    """The words of a line without their punctuation: [{"text", "token", "lead", "trail"}].

    "جنسيته:مغربية." gives two units ("جنسيته", "مغربية"); lead / trail are the punctuation written before /
    after the unit inside its token, so the line can be written again exactly (write_units); begin / finish are
    its position in the line, punctuation included.
    """
    units = []
    for token_index, token_match in enumerate(re.finditer(r"\S+", line)):
        token, offset = token_match.group(), token_match.start()
        pieces = [m for m in re.finditer(r"[^" + re.escape(PUNCTUATION) + r"]+", token)]
        for k, match in enumerate(pieces):
            lead = token[:match.start()] if k == 0 else ""
            trail = token[match.end():pieces[k + 1].start()] if k + 1 < len(pieces) else token[match.end():]
            units.append({
                "text": match.group(),
                "token": token_index,
                "lead": lead,
                "trail": trail,
                # where the unit is written in the line, with its punctuation (for display)
                "begin": offset + match.start() - len(lead),
                "finish": offset + match.end() + len(trail),
            })
    return units


def unit_key(text):
    """What the parser sees of a word: no diacritics, ڭ = ك, أ / إ / آ = ا."""
    return surface(text).translate(SAME_LETTERS)


def write_units(line, units, replacements):
    """The line with units[start:end] replaced by new words; the words not replaced keep their exact writing.

    replacements: {start: (end, [new words])}, indexes in units = line_units(line). Only the tokens that
    contain a replaced unit are written again (with their punctuation); the others are copied as they are.
    """
    pieces = {}            # token index -> written pieces of its units
    i = 0
    while i < len(units):
        unit = units[i]
        if i in replacements:
            end, words = replacements[i]
            last = units[end - 1]
            text = " ".join(words)
            pieces.setdefault(unit["token"], []).append(unit["lead"] + text + last["trail"] if text else "")
            for k in range(i + 1, end):   # tokens fully inside the replacement disappear
                pieces.setdefault(units[k]["token"], [])
            i = end
            continue
        pieces.setdefault(unit["token"], []).append(unit["lead"] + unit["text"] + unit["trail"])
        i += 1
    touched = {units[k]["token"] for start, (end, _) in replacements.items() for k in range(start, end)}
    out = []
    for k, token in enumerate(line.split()):
        if k in touched:
            written = " ".join(p for p in pieces.get(k, []) if p)
            if written:
                out.append(written)
        else:
            out.append(token)
    return " ".join(out)


def align_block(old, new, vocabulary=None):
    """Pair the words of a changed block: [(old words, new words)].

    The word diff groups neighbouring changes ("ثلاثة ميلاي ية" -> "أربعة ميلادية"); they are paired again so
    each change is judged alone. A pair has 1 to MAX_WORDS words on each side; a word can also be removed
    (new side empty) or added (old side empty). A correct word is never glued to another one in a pair
    ("التانس عشرة" -> "التاسعة" is "التانس" -> "التاسعة" + "عشرة" removed, not one change of two words).
    Dynamic programming on the spelling distance.
    """
    def cost(o, n):
        if not o or not n:
            return float(len(o) + len(n))
        if len(o) > 1 and any(is_known(w, vocabulary) and len(surface(w)) >= MIN_LETTERS for w in o):
            return None   # a correct word inside a merge
        ratio = SequenceMatcher(None, unit_key("".join(o)), unit_key("".join(n))).ratio()
        return (1 - ratio) * max(len(o), len(n))

    best = {(0, 0): (0.0, [])}
    for i in range(len(old) + 1):
        for j in range(len(new) + 1):
            if (i, j) not in best:
                continue
            total, pairs = best[(i, j)]
            for a in range(0, MAX_WORDS + 1):
                for b in range(0, MAX_WORDS + 1):
                    if (a == 0 and b == 0) or i + a > len(old) or j + b > len(new):
                        continue
                    o, n = old[i:i + a], new[j:j + b]
                    step = cost(o, n)
                    if step is None:
                        continue
                    key = (i + a, j + b)
                    if key not in best or total + step < best[key][0]:
                        best[key] = (total + step, pairs + [(o, n)])
    found = best.get((len(old), len(new)))
    return found[1] if found else [(old, new)]


# a time phrase: after "الساعة" / "الدقيقة", up to the next label of the form
TIME_PHRASE = re.compile(r"(?:الساعة|الدقيقة)\s*(.+?)(?=\s+(?:و\s*الدقيقة|على\s+الساعة|و?لد|و?الد)|$)")


def normalize_spaces(text):
    return re.sub(r"\s+", " ", text).strip()


def time_values(text):
    """The hours / minutes of a line that the time dictionary knows ([] when none or not readable)."""
    from converter import time_value
    values = []
    for match in TIME_PHRASE.finditer(normalize_spaces(SPLIT_PUNCTUATION.sub(" ", text))):
        value = time_value(match.group(1))
        values.append(value if isinstance(value, int) else None)
    return values


def spacing_only(old, new, vocabulary=None):
    """Only the spaces differ and the original words are correct ("ب جماعة" / "بجماعة"): the parser reads
    both the same, so it is not a correction. A word cut by the HTR ("ميلاي ية") is not correct: kept."""
    return bool(old) and bool(new) and "".join(map(unit_key, old)) == "".join(map(unit_key, new)) and all(
        is_known(w, vocabulary) or (len(w) == 1 and w in PREFIXES) for w in old
    )


def line_changes(original, proposed, line_index, vocabulary=None, protected=None):
    """Every change the model proposes in one line, judged by the rules -> list of changes.

    A change: {"line": index of the line, "start" / "end": its units in line_units(original), "old": [words],
    "new": [words] (empty = the model removed the words), "accepted": decision of the rules, "reason": why the
    rules refused it ("" if accepted)}. What the parser cannot see is not a change: punctuation, spaces
    around a prefix ("ب جماعة" = "بجماعة"), ڭ / ك, hamza on alif.
    protected: indexes of the units that must not change (default: protected_positions of this line alone).
    """
    old_units, new_units = line_units(original), line_units(proposed)
    old_keys = [unit_key(u["text"]) for u in old_units]
    new_keys = [unit_key(u["text"]) for u in new_units]
    if old_keys == new_keys:
        return []
    if protected is None:
        protected = protected_positions([u["text"] for u in old_units], vocabulary)[0]
    changes = []
    matcher = SequenceMatcher(None, old_keys, new_keys, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        old = [u["text"] for u in old_units[i1:i2]]
        new = [u["text"] for u in new_units[j1:j2]]
        if spacing_only(old, new, vocabulary):
            continue
        position = i1
        for old_part, new_part in align_block(old, new, vocabulary):
            start, end = position, position + len(old_part)
            position = end
            if not old_part:
                # words added by the model: only words of the printed form ("الدقيقة"), written before the next
                # word and applied when the record needs them (judge_by_record)
                if (new_part and start < len(old_units) and all(surface(w) in FORM_VOCABULARY for w in new_part)
                        and not any(protected.get(k) for k in (start - 1, start))):
                    following = old_units[start]["text"]
                    changes.append({"line": line_index, "start": start, "end": start + 1, "old": [following],
                                    "new": list(new_part) + [following], "accepted": False, "source": "added",
                                    "context": "a word of the form added by the model",
                                    "reason": "not applied: the record did not need it"})
                continue
            if spacing_only(old_part, new_part, vocabulary):
                continue
            if not new_part:
                accepted, reason = False, "the model removed a word"
            elif any(protected.get(k) == "value" for k in range(start, end)):
                accepted, reason = False, "a name, place or address is never changed"
            elif any(protected.get(k) == "job" for k in range(start, end)):
                accepted, reason = False, "not a job of jobs.json"
            else:
                accepted, reason = accept_change(old_part, new_part, vocabulary)
                if not accepted and reason.startswith("too different"):
                    # a misread hour / minute: accepted when the time phrase becomes one the dictionary knows
                    candidate = write_units(original, old_units, {start: (end, new_part)})
                    before, after = time_values(original), time_values(candidate)
                    if len(before) == len(after) and any(b is None and a is not None
                                                         for b, a in zip(before, after)):
                        accepted, reason = accept_change(old_part, new_part, vocabulary, max_distance=3)
            changes.append({"line": line_index, "start": start, "end": end, "old": old_part, "new": new_part,
                            "accepted": accepted, "reason": reason})
    return changes


def describe(change):
    """One readable line for a change ("line 4: 'موفق' -> 'موافق'", or the refusal and its reason)."""
    before = " ".join(change["old"]) or "(nothing)"
    after = " ".join(change["new"]) or "(removed)"
    if change.get("flag"):
        return (f"line {change['line'] + 1}: {' '.join(change['old'])!r} {change['reason']} "
                "(no correction found, check the scan)")
    if change.get("source") in ("names", "split"):
        source = f" ({change['context']})"
    elif change.get("source") == "date_time.json":
        source = f" (date_time.json{', ' + change['context'] if change.get('context') else ''})"
    elif change.get("source") == "jobs.json":
        source = " (jobs.json" + (", " + change["context"] if change.get("context") else "") + ")"
    else:
        source = f" ({change['context']})" if change.get("context") else ""
    if change["accepted"]:
        return f"line {change['line'] + 1}: {before!r} -> {after!r}{source}"
    if change.get("check"):
        return f"line {change['line'] + 1}: check {before!r} -> {after!r}{source} ({change['reason']})"
    return f"line {change['line'] + 1}: refused {before!r} -> {after!r}{source} ({change['reason']})"


def apply_changes(text, changes, accepted=None):
    """The text with the chosen changes applied; the other lines (and words) stay exactly as they were.

    accepted: set of the indexes (in `changes`) to apply; default = the decision of the rules.
    """
    if accepted is None:
        accepted = {i for i, change in enumerate(changes) if change["accepted"]}
    lines = text.split("\n")
    by_line = {}
    for i in sorted(accepted):
        change = changes[i]
        if change["old"]:
            by_line.setdefault(change["line"], {})[change["start"]] = (change["end"], change["new"])
    for line_index, replacements in by_line.items():
        lines[line_index] = write_units(lines[line_index], line_units(lines[line_index]), replacements)
    return "\n".join(lines)


def merge_line(original, proposed, line_number, notes, vocabulary=None, protected=None):   # used by the tests
    """The original line with only the changes accepted by the rules applied (notes: every change)."""
    changes = line_changes(original, proposed, 0, vocabulary, protected)
    for change in changes:
        notes.append(describe({**change, "line": line_number - 1}))
    return apply_changes(original, changes) if changes else original


# ---------------------------------------------------------------------------------------------------------
# local model (Ollama)
# ---------------------------------------------------------------------------------------------------------
PROMPT = """You correct the spelling of Moroccan Arabic birth certificates read by a handwriting recognition model.
The user sends a JSON object {"lines": [{"id": 1, "text": "..."}, ...]}, one object per line of the document.

Correct ONLY spelling mistakes made by the reading model:
- a missing, extra, doubled or wrong letter in a word ("البيتت" -> "البيت", "موفق" -> "موافق"),
- a word cut in two ("ميلاي ية" -> "ميلادية"),
- a misread profession after "حرفته" / "حرفتها" / "مهنته" / "مهنتها", only when the written word is not a real
  profession: write the profession it is most probably a misreading of (one or two letters differ),
- a word of the printed form lost or cut by the reading model, when the sentence of the form needs it: the
  minute's label in "على الساعة الواحدة و العشرين و الدقيقة الصفر" ("وة: الصفر" -> "و الدقيقة: الصفر"),
  "موافق", "سنة", "عام", "هجرية", "ميلادية". Only these fixed words of the form, never a name, a place or a number.
NEVER replace a correct word by another correct word: a correct profession by another profession, a correct
number by another number. NEVER change: personal names, family names, place names you are not sure of, numbers, digits, Latin text
(PRENOM, NOM and the French names), punctuation, the order of the words. Never remove a word, never rephrase,
never translate, never complete missing information (a name, a number, a date). When unsure, keep the word.

Return ONLY a JSON object {"lines": [{"id": 1, "text": "..."}, ...]}: one object per input line, with the same
"id". Never split a line in two and never join two lines, even when a word seems to belong to the next line."""


def ollama_url(path):
    """Address of the Ollama API; refuses anything that is not this computer unless explicitly allowed."""
    host = urlparse(OLLAMA_URL).hostname or ""
    if host not in LOCAL_HOSTS and os.environ.get("MADOC_SPELL_ALLOW_REMOTE") != "1":
        raise PermissionError(
            f"MADOC_OLLAMA_URL={OLLAMA_URL!r} is not this computer: the documents must not leave it "
            "(set MADOC_SPELL_ALLOW_REMOTE=1 only for a server of your own network)"
        )
    return OLLAMA_URL.rstrip("/") + path


def ollama_json(system, payload, model=None):
    """One request to the local model: system instructions + a JSON payload -> the JSON object it answers."""
    body = {
        "model": model or MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_ctx": 8192},
    }
    request = urllib.request.Request(
        ollama_url("/api/chat"),
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        answer = json.loads(response.read().decode("utf-8"))
    content = answer.get("message", {}).get("content", "")
    result = json.loads(content)
    if not isinstance(result, dict):
        raise ValueError("the model did not return a JSON object")
    return result


def ask_model(lines, model=None):
    """Lines -> the model's corrected lines, one per input line (see align_answer), or raise."""
    payload = {"lines": [{"id": n, "text": line} for n, line in enumerate(lines, 1)]}
    return align_answer(lines, ollama_json(PROMPT, payload, model).get("lines"))


def align_answer(lines, answer):
    """The model's answer -> exactly one proposed line per input line.

    The lines are numbered, so the answer is matched by "id": a line the model split in two (two objects with
    the same id) is joined again, a line it forgot stays as it was, an unknown id is ignored. An answer of plain
    strings is matched by position when the count is right, otherwise by similarity (a split line is joined
    again). One misbehaving line never costs the corrections of the others.
    """
    if not isinstance(answer, list):
        raise ValueError("the model did not return a list of lines")
    if all(isinstance(entry, dict) for entry in answer):
        by_id = {}
        for entry in answer:
            try:
                number = int(entry.get("id"))
            except (TypeError, ValueError):
                continue
            text = entry.get("text")
            if isinstance(text, str) and 1 <= number <= len(lines):
                by_id.setdefault(number, []).append(text)
        if not by_id:
            raise ValueError("the model returned no usable line")
        return [" ".join(by_id[n]) if n in by_id else line for n, line in enumerate(lines, 1)]
    texts = [entry for entry in answer if isinstance(entry, str)]
    if len(texts) == len(lines):
        return texts
    return align_by_similarity(lines, texts)


def align_by_similarity(lines, texts):
    """Proposed lines whose count differs from the input: each input line takes the proposed line(s) most like
    it, in order (two consecutive proposed lines are joined when together they match better)."""
    def ratio(a, b):
        return SequenceMatcher(None, a, b).ratio()

    out, j = [], 0
    for line in lines:
        if j >= len(texts):
            out.append(line)
            continue
        best, take = ratio(line, texts[j]), 1
        if j + 1 < len(texts) and ratio(line, texts[j] + " " + texts[j + 1]) > best:
            best, take = ratio(line, texts[j] + " " + texts[j + 1]), 2
        if best < 0.5:   # nothing alike: the line is kept
            out.append(line)
            continue
        out.append(" ".join(texts[j:j + take]))
        j += take
    return out


# ---------------------------------------------------------------------------------------------------------
# the child's names: the Arabic and the Latin reading must be the same name
# ---------------------------------------------------------------------------------------------------------
NAME_PROMPT = """You transliterate Moroccan personal names between Arabic script and the Latin spelling used by
Moroccan civil registers (French conventions: ش = CH, خ = KH, غ = GH, و = OU, ج = J; the article ال is often
not written: الإدريسي = IDRISSI).

The user sends {"names": [{"kind": "...", "arabic": "...", "latin": "..."}]}: two readings of the same name.
For each name return:
- "arabic_of_latin": the usual Arabic spelling of the Latin reading,
- "latin_of_arabic": the usual registry Latin spelling (upper case) of the Arabic reading.
Transliterate only: do not correct, do not choose, do not translate the meaning.

Return ONLY {"names": [{"arabic_of_latin": "...", "latin_of_arabic": "..."}]}, one object per input name."""

# Arabic letter -> consonant class shared with the French spelling (vowels, alif, waw, ya, ain, ta marbuta: none)
ARABIC_SOUNDS = {
    "ب": "B", "ت": "T", "ث": "T", "ج": "J", "ح": "H", "خ": "K", "د": "D", "ذ": "D", "ر": "R", "ز": "Z",
    "س": "S", "ش": "C", "ص": "S", "ض": "D", "ط": "T", "ظ": "D", "غ": "K", "ف": "F", "ق": "K", "ك": "K",
    "ڭ": "K", "گ": "K", "ل": "L", "م": "M", "ن": "N", "ه": "H", "پ": "B", "ڤ": "F",
}
# G and K are one class: ق is written K, Q or G in French spellings, غ GH, ڭ G
# French spellings of Moroccan names: two letters for one sound, then one letter -> the same classes
LATIN_PAIRS = [("CH", "C"), ("KH", "K"), ("GH", "K"), ("TH", "T"), ("DH", "D"), ("DJ", "J"), ("OU", ""),
               ("QU", "K")]
LATIN_SOUNDS = {"B": "B", "T": "T", "J": "J", "G": "K", "H": "H", "D": "D", "R": "R", "Z": "Z", "S": "S",
                "F": "F", "Q": "K", "K": "K", "L": "L", "M": "M", "N": "N", "P": "B", "V": "F", "X": "KS"}


def arabic_skeleton(name, drop_articles=()):
    """'الإدريسي' -> 'LDRS': the consonants of an Arabic name (doubled letters dropped).

    drop_articles: indexes of the words whose leading "ال" is not written in French (الإدريسي = IDRISSI,
    عبد الرحيم = ABDERRAHIM); it is not always an article ("الياس" = ILYASSE), see arabic_skeletons."""
    out = []
    for k, word in enumerate(surface(name).split()):
        if k in drop_articles and word.startswith("ال") and len(word) > 3:
            word = word[2:]
        out.extend(ARABIC_SOUNDS.get(char, "") for char in word)
    return re.sub(r"(.)\1+", r"\1", "".join(out))


def arabic_skeletons(name):
    """Every skeleton of the name, with each leading "ال" written or not."""
    from itertools import combinations
    starts = [k for k, word in enumerate(surface(name).split()) if word.startswith("ال") and len(word) > 3]
    return {arabic_skeleton(name, set(drop)) for n in range(len(starts) + 1) for drop in combinations(starts, n)}


def latin_skeleton(name):
    """'EL IDRISSI' -> 'DRS': the consonants of a French spelling, with the classes of arabic_skeleton."""
    name = re.sub(r"[^A-Z ]", "", unit_key_latin(name))
    name = re.sub(r"^(?:EL|AL) +", "", name).replace(" ", "")
    out, i = [], 0
    while i < len(name):
        pair = next((two for two, sound in LATIN_PAIRS if name.startswith(two, i)), None)
        if pair:
            out.append(dict(LATIN_PAIRS)[pair])
            i += 2
            continue
        char = name[i]
        if char == "C":   # C before E / I sounds S, otherwise K
            out.append("S" if name[i + 1:i + 2] in ("E", "I") else "K")
        else:
            out.append(LATIN_SOUNDS.get(char, ""))
        i += 1
    return re.sub(r"(.)\1+", r"\1", "".join(out))


def unit_key_latin(name):
    import unicodedata
    name = unicodedata.normalize("NFKD", name)
    return "".join(c for c in name if not unicodedata.combining(c)).upper()


def names_agree(arabic, latin):
    """True when the Arabic and the Latin reading can be the same name ('بلال' / 'BILAL', 'قاسم' / 'KACIM')."""
    skeletons, b = arabic_skeletons(arabic), latin_skeleton(latin)
    if not b or not all(skeletons):
        return True   # nothing to compare (a name of vowels only): no alarm
    if b in skeletons:
        return True
    # long names only: one consonant may differ (a short name has no letter to spare: BIRAL is not بلال)
    return len(b) >= 5 and max(SequenceMatcher(None, a, b).ratio() for a in skeletons) >= 0.85


def name_span(lines, line_index, start, take):
    """Units after a label, on its line or (when the label ends the line) on the next line: (line, start, end)."""
    for index in (line_index, line_index + 1):
        if index >= len(lines):
            break
        units = line_units(lines[index])
        first = start if index == line_index else 0
        end = first
        while end < len(units) and take(units[end]["text"]):
            end += 1
        if end > first:
            return index, first, end
        if index == line_index and first < len(units):
            break   # something else follows the label on its line: no name there
    return None


def child_names(lines, vocabulary=None):
    """The child's names as read: {"first name" / "family name": {"arabic": (line, start, end), "latin": ...}}.

    Arabic: the words after "الإسم الشخصي" / "الإسم العائلي" (also misspelt); Latin: after "PRENOM" / "NOM".
    These labels are only written for the child (the scan may print "NOM ..." after the father's line).
    """
    def arabic_name_word(text):
        return bool(ARABIC_WORD.search(text)) and not is_known(text, vocabulary) and not label_kind(surface(text))

    def latin_name_word(text):
        return bool(re.fullmatch(r"[A-Za-zÀ-ÿ'’\-]+", text)) and text.upper() not in ("PRENOM", "NOM")

    found = {}
    for line_index, line in enumerate(lines):
        units = line_units(line)
        words = [surface(u["text"]) for u in units]
        for i, word in enumerate(words):
            kind = None
            if word.upper() == "PRENOM":
                kind, side, take = "first name", "latin", latin_name_word
            elif word.upper() == "NOM":
                kind, side, take = "family name", "latin", latin_name_word
            elif edit_distance(word, "الشخصي") <= 1:
                kind, side, take = "first name", "arabic", arabic_name_word
            elif edit_distance(word, "العائلي") <= 1:
                kind, side, take = "family name", "arabic", arabic_name_word
            if kind and side not in found.get(kind, {}):
                span = name_span(lines, line_index, i + 1, take)
                if span:
                    found.setdefault(kind, {})[side] = span
    return {kind: sides for kind, sides in found.items() if len(sides) == 2}


def span_words(lines, span):
    line, start, end = span
    return [u["text"] for u in line_units(lines[line])[start:end]]


def name_answers(answer, count):
    """The model's answer -> [{"arabic_of_latin", "latin_of_arabic"}] (tolerates a bare list or a single object)."""
    if isinstance(answer, dict) and isinstance(answer.get("names"), list):
        answer = answer["names"]
    elif isinstance(answer, dict) and ("arabic_of_latin" in answer or "latin_of_arabic" in answer):
        answer = [answer]
    if not isinstance(answer, list):
        return [{}] * count
    out = [a if isinstance(a, dict) else {} for a in answer][:count]
    return out + [{}] * (count - len(out))


def name_candidates(arabic, latin, entry):
    """The corrections the transliterations allow: [(distance, side, new text)], closest first.

    "arabic": the Arabic reading replaced by the Arabic spelling of the Latin reading ("ملال" -> "بلال");
    "latin": the Latin reading replaced by the Latin spelling of the Arabic reading. A side is changed only by
    2 letters at most, with the same number of words, and the two readings must then agree.
    """
    candidates = []
    new_arabic = str(entry.get("arabic_of_latin") or "").strip()
    if (new_arabic and surface(new_arabic) != surface(arabic) and ARABIC_WORD.search(new_arabic)
            and not DIGIT_OR_LATIN.search(new_arabic) and len(new_arabic.split()) == len(arabic.split())
            and names_agree(new_arabic, latin)):
        distance = edit_distance(surface(arabic).replace(" ", ""), surface(new_arabic).replace(" ", ""))
        if distance <= 2:
            candidates.append((distance, "arabic", new_arabic))
    new_latin = re.sub(r"\s+", " ", unit_key_latin(str(entry.get("latin_of_arabic") or ""))).strip()
    if (new_latin and new_latin != unit_key_latin(latin) and re.fullmatch(r"[A-Z'’\- ]+", new_latin)
            and len(new_latin.split()) == len(latin.split()) and names_agree(arabic, new_latin)):
        distance = edit_distance(unit_key_latin(latin).replace(" ", ""), new_latin.replace(" ", ""))
        if distance <= 2:
            candidates.append((distance, "latin", new_latin))
    return sorted(candidates)


def document_name(lines, span, arabic, latin):
    """(name, line) written elsewhere in the text that agrees with the Latin reading and differs from the Arabic
    reading by one letter per word, plus one ("تفاص" / TIFAF: the father is "رشيد تفاف"; "لحيس" / LAHNICHE:
    "لحنيش"), or None."""
    size = len(arabic.split())
    target = unit_key(arabic.replace(" ", ""))
    found = set()
    for line_index, line in enumerate(lines):
        words = [u["text"] for u in line_units(line)]
        for k in range(len(words) - size + 1):
            if (line_index, k) == (span[0], span[1]):
                continue
            candidate = " ".join(words[k:k + size])
            key = unit_key(candidate.replace(" ", ""))
            if (key != target and ARABIC_WORD.search(candidate) and not DIGIT_OR_LATIN.search(candidate)
                    and not any(is_known(w) for w in words[k:k + size])
                    and edit_distance(key, target) <= size + 1 and names_agree(candidate, latin)):
                found.add((surface(candidate), line_index))
    names = {name for name, _ in found}
    if len(names) != 1:
        return None   # nothing, or several different names: the model / the person decides
    return min(found, key=lambda item: item[1])


def name_changes(lines, model_changes, notes, model=None, use_model=True):
    """Proposals for the child's names whose Arabic and Latin readings disagree (never applied by default).

    The local model only transliterates (the Latin reading into Arabic, the Arabic reading into Latin); the
    side that needs the fewest letters changed to agree with the other one gets the proposal (both on a
    tie). Without a usable proposal, the Arabic name is flagged in the Spelling tab ("≠ BILAL") and the notes
    say what the model answered. The model's own changes on these names are replaced by this check.
    """
    names = child_names(lines)
    disagreeing = []
    for kind, sides in names.items():
        arabic = " ".join(span_words(lines, sides["arabic"]))
        latin = " ".join(span_words(lines, sides["latin"]))
        if not names_agree(arabic, latin):
            disagreeing.append((kind, sides, arabic, latin))
    if not disagreeing:
        return []
    # the same name written elsewhere in the certificate (the father's family name, the declarant ...) that
    # agrees with the Latin reading and is one letter from the Arabic one: applied, no model needed
    resolved = {}
    for n, (kind, sides, arabic, latin) in enumerate(disagreeing):
        found = document_name(lines, sides["arabic"], arabic, latin)
        if found:
            resolved[n] = found
            continue
        # a first name of names.json one letter away that agrees with the Latin reading ("ببيل" / NABIL)
        close = [w for w in close_entries(arabic, get_names()) if len(ar_key(w)) == len(ar_key(arabic))
                 and names_agree(w, latin)] if len(arabic.split()) == 1 else []
        if len(close) == 1:
            resolved[n] = (close[0], None)
            continue
        # two letters away: proposed in orange, never applied by the rules
        close = [w for w in close_entries(arabic, get_names(), limit=2) if names_agree(w, latin)] \
            if len(arabic.split()) == 1 else []
        if 1 <= len(close) <= 3:
            resolved[n] = (close, "check")
    unresolved = [n for n in range(len(disagreeing)) if n not in resolved]
    entries = [{}] * len(disagreeing)
    if use_model and unresolved:
        try:
            asked = [disagreeing[n] for n in unresolved]
            answer = ollama_json(NAME_PROMPT, {"names": [{"kind": k, "arabic": a, "latin": l}
                                                         for k, _, a, l in asked]}, model)
            for n, entry in zip(unresolved, name_answers(answer, len(asked))):
                entries[n] = entry
        except Exception as e:   # not running, refused address, bad answer: the mismatch is only flagged
            notes.append(f"names check without the model ({type(e).__name__}: {e})")
    changes = []
    for n, ((kind, sides, arabic, latin), entry) in enumerate(zip(disagreeing, entries)):
        for side in ("arabic", "latin"):
            line, start, end = sides[side]
            for change in [c for c in model_changes if c["line"] == line and start <= c["start"] < end]:
                model_changes.remove(change)
        if n in resolved and resolved[n][1] == "check":
            options, _ = resolved[n]
            line, start, end = sides["arabic"]
            changes.append({"line": line, "start": start, "end": end, "old": arabic.split(), "new": [options[0]],
                            "accepted": False, "check": True, "source": "names",
                            **({"options": [[w] for w in options]} if len(options) > 1 else {}),
                            "context": f"{kind}: first names of names.json that agree with {latin}",
                            "reason": f"two letters differ from {arabic}: click to use it"})
            continue
        if n in resolved:
            new, at = resolved[n]
            line, start, end = sides["arabic"]
            where = f"written so on line {at + 1}" if at is not None else "a first name of names.json"
            changes.append({"line": line, "start": start, "end": end, "old": arabic.split(), "new": new.split(),
                            "accepted": True, "source": "names", "reason": "",
                            "context": f"{kind}: {new} / {latin}, {where}"})
            continue
        candidates = name_candidates(arabic, latin, entry)
        best = [c for c in candidates if c[0] == candidates[0][0]] if candidates else []
        if not best:
            told = f"; the model answered {entry}" if entry else ""
            notes.append(f"child's {kind}: the Arabic {arabic!r} and the Latin {latin!r} do not match, no "
                         f"correction found (check the scan){told}")
            line, start, end = sides["arabic"]
            changes.append({"line": line, "start": start, "end": end, "old": arabic.split(), "new": arabic.split(),
                            "accepted": False, "flag": True, "source": "names",
                            "reason": f"does not match the Latin {latin}", "context": f"≠ {latin}"})
            continue
        for _, side, new in best:
            old = arabic if side == "arabic" else latin
            line, start, end = sides[side]
            pair = f"{new} / {latin}" if side == "arabic" else f"{arabic} / {new}"
            changes.append({"line": line, "start": start, "end": end, "old": old.split(), "new": new.split(),
                            "accepted": False, "source": "names",
                            "reason": f"the Arabic and the Latin {kind} did not match ({arabic} / {latin})",
                            "context": f"{kind}: {pair}"})
    return changes


def propose(text, model=None):
    """Text -> (changes proposed by the model and judged by the rules, notes when the check was skipped).

    Nothing is applied here: correct_text applies the decisions of the rules, the web interface lets a person
    accept or refuse every change (apply_changes).
    """
    notes = []
    lines = text.split("\n")
    # empty lines are not sent (kept in place)
    positions = [i for i, line in enumerate(lines) if line.strip()]
    if not positions:
        return [], notes
    proposed = None
    try:
        proposed = ask_model([lines[i] for i in positions], model)
    except PermissionError as e:
        notes.append(f"spelling model skipped: {e}")
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        notes.append(f"spelling model skipped: Ollama not reachable at {OLLAMA_URL} ({e})")
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        notes.append(f"spelling model skipped: unusable answer from the model ({e})")
    model_ok = proposed is not None
    if proposed is None:
        # without the model, the jobs are still checked against jobs.json
        proposed = [lines[i] for i in positions]
    vocabulary = get_vocabulary()
    # the values (names, places, jobs) are found on the original text, line after line (a label can end a line)
    changes, inside, protected_by_line = [], None, {}
    for i, new_line in zip(positions, proposed):
        protected, inside = protected_positions([u["text"] for u in line_units(lines[i])], vocabulary, inside)
        protected_by_line[i] = protected
        changes.extend(line_changes(lines[i], new_line, i, vocabulary, protected))
    # the jobs are read in the context of the whole document (a parent's job is written twice)
    changes.extend(job_changes(text, lines, protected_by_line, changes))
    # the child's names: the Arabic and the Latin reading are compared (proposals, never applied by default)
    changes.extend(name_changes(lines, changes, notes, model, use_model=model_ok))
    # a label glued to the previous word is split again; misread date words get the dictionary's word
    changes.extend(glued_labels(lines, changes))
    changes.extend(date_word_suggestions(lines, protected_by_line, changes))
    # inside names and places: the words of an address, cities.json, names.json, the rest of the certificate
    changes.extend(value_word_suggestions(lines, protected_by_line, changes))
    changes.extend(minute_label(lines, changes))
    changes.extend(form_word_suggestions(lines, protected_by_line, changes))
    changes.extend(declarant_names(lines, changes))
    changes = sorted(changes, key=lambda c: (c["line"], c["start"]))
    # numbers: judged by the whole record (a misread year makes a date impossible)
    date_fixes(text, changes)
    joint_date_fixes(text, changes)
    judge_by_record(text, changes, ("time", "added"))
    # a refused model change of words that are already correct is noise ("ثالث" -> "الثالث عشر"): notes only
    for change in [c for c in changes if model_change(c) and not c["accepted"]
                   and c["reason"] == "the original words are already correct words"]:
        notes.append("ignored: " + describe(change))
        changes.remove(change)
    # the certificate read as a whole: a Gregorian date the Hijri one contradicts, an address written twice
    changes.extend(date_flags(text, changes))
    changes.extend(address_flags(lines, changes))
    return without_overlaps(changes, notes), notes


def without_overlaps(changes, notes):
    """The changes sorted, two changes never on the same words: the review could not show both and applying
    both would keep only one. The applied one wins, then a check / flag, then a refused proposal."""
    rank = {id(c): (0 if c["accepted"] else 1 if c.get("check") or c.get("flag") else 2) for c in changes}
    kept, taken = [], set()
    for change in sorted(changes, key=lambda c: (rank[id(c)], c["line"], c["start"])):
        span = {(change["line"], k) for k in range(change["start"], max(change["end"], change["start"] + 1))}
        if span & taken:
            notes.append("dropped (another change on these words): " + describe(change))
            continue
        taken |= span
        kept.append(change)
    return sorted(kept, key=lambda c: (c["line"], c["start"]))


@lru_cache(maxsize=None)
def number_words():
    """The number words of date_time.json, as written (with and without the article)."""
    words = set()
    for aliases in dates_data.get("numbers", {}).values():
        for alias in aliases if isinstance(aliases, list) else []:
            for word in surface(str(alias)).split():
                words.update({word, word[2:] if word.startswith("ال") else "ال" + word})
    return frozenset(words)


@lru_cache(maxsize=None)
def month_words():
    """The month words of date_time.json, as written (one word each: "ربيع الثاني" is two words)."""
    words = set()
    for section in ("hijri_months", "miladi_months"):
        for aliases in dates_data.get(section, {}).values():
            for alias in aliases if isinstance(aliases, list) else []:
                if len(surface(str(alias)).split()) == 1 and len(surface(str(alias))) >= 4:
                    words.add(surface(str(alias)))
    return frozenset(words)


@lru_cache(maxsize=None)
def date_words():
    """Number and month words of date_time.json, as written (with and without the article)."""
    words = set(number_words())
    for section in ("hijri_months", "miladi_months"):
        for aliases in dates_data.get(section, {}).values():
            for alias in aliases if isinstance(aliases, list) else []:
                for word in surface(str(alias)).split():
                    words.update({word, word[2:] if word.startswith("ال") else "ال" + word})
    return frozenset(w for w in words if len(w) >= 3)


def date_word_suggestions(lines, protected_by_line, changes):
    """A misread number or month word ("ثالن") gets the words of date_time.json one letter away ("ثالث",
    "ثامن") as a proposal, with or without the model. It is applied only when it makes a date of the record
    valid, and with several words only when one of them makes the Hijri and Gregorian dates agree to the day
    (date_fixes); otherwise it stays a refused proposal (the person can choose)."""
    words = date_words()
    out = []
    for line_index, protected in protected_by_line.items():
        for k, unit in enumerate(line_units(lines[line_index])):
            word = surface(unit["text"])
            if (k in protected or len(word) < 3 or not ARABIC_WORD.search(word)
                    or DIGIT_OR_LATIN.search(word) or is_known(word) or covered(changes, line_index, k)):
                continue
            key = unit_key(word)
            close = one_per_key(w for w in words if abs(len(w) - len(word)) <= 1 and edit_distance(key, unit_key(w)) <= 1)
            if not close and len(word) >= 4:
                # a month is a long word of a short list: two letters may be misread ("دجرة" -> "دجنبر")
                close = one_per_key(w for w in month_words() if abs(len(w) - len(word)) <= 2
                                    and edit_distance(key, unit_key(w)) <= 2)
            if 1 <= len(close) <= 6:
                # the word with the same last letter first ("ثمانة": "ثمانية" before "ثمان" / "ثماني")
                close.sort(key=lambda w: (edit_distance(key, unit_key(w)), w[-1:] != word[-1:],
                                          abs(len(w) - len(word)), w))
                drop_refused_model_changes(changes, line_index, k)
                out.append({"line": line_index, "start": k, "end": k + 1, "old": [unit["text"]], "new": [close[0]],
                            "options": [[w] for w in close], "accepted": False,
                            "reason": "not applied: no date of the record needed it" if len(close) == 1 else
                                      "several words are possible: " + " / ".join(close),
                            "source": "date_time.json"})
    return out


def model_change(change):
    """A change proposed by the spelling model itself (not by a dictionary or a check of the document)."""
    return not change.get("source") and not change.get("flag")


def covered(changes, line_index, k):
    """True when unit k of the line is already handled: an accepted change, or a change of a dictionary or a
    check. A refused change of the model does not count: a dictionary word can replace it ("ششوال": the
    model proposed "من شعبان", date_time.json has "شوال")."""
    return any(c["line"] == line_index and c["start"] <= k < max(c["end"], c["start"] + 1)
               and (c["accepted"] or not model_change(c)) for c in changes)


def drop_refused_model_changes(changes, line_index, k):
    """Remove the refused model changes on unit k (a dictionary or document proposal replaces them)."""
    for change in [c for c in changes if c["line"] == line_index and c["start"] <= k < max(c["end"], c["start"] + 1)
                   and model_change(c) and not c["accepted"]]:
        changes.remove(change)


def one_per_key(words):
    """The words, one per spelling the parser reads the same ("ألفين" / "الفين"), the one with a hamza first."""
    best = {}
    for word in sorted(set(words)):
        key = unit_key(word)
        if key not in best or (word != key and best[key] == key):
            best[key] = word
    return sorted(best.values())


# ---------------------------------------------------------------------------------------------------------
# value slots: the common words of an address, the places of cities.json, the first names of names.json
# ---------------------------------------------------------------------------------------------------------
# words of an address that are not names: a misread one may be corrected inside a place
ADDRESS_WORDS = frozenset(surface(w) for w in """
جماعة الجماعة مدينة اقليم إقليم دائرة عمالة قيادة باشوية مقاطعة دوار قرية مركز الحضرية القروية حي زنقة الزنقة
شارع درب تجزئة إقامة عمارة الشطر الرقم محج المستشفى مستشفى مصحة المصحة مستوصف المستوصف الإقليمي
""".split())
# labels after which a place of cities.json is written ("بجماعة اليوسفية", "اقليم آسفي"), compared without a
# glued و / ب and without the article
PLACE_LABELS = frozenset(ar_key(w) for w in """
ب جماعة مدينة اقليم إقليم أقليم دائرة عمالة قيادة باشوية مقاطعة دوار قرية المولود المولودة ولد
""".split())
# labels after which a person's name is written
NAME_LABELS = frozenset(ar_key(w) for w in """
الشخصي العائلي والده والدها والدته والدتها والد والدت بن بنت ابن ابنة السيد السيدة نحن
""".split())
# the first word of many place names: never corrected ("اولاد عزوز", "بني ملال", "سيدي بنور", "آيت ...")
PLACE_PREFIX_WORDS = frozenset(ar_key(w) for w in "اولاد أولاد بني سيدي عين ايت آيت مولاي دار أهل اهل".split())


def label_key(word):
    """The key of a label word, without a glued و / ب ("بجماعة" -> "جماعة")."""
    key = ar_key(word)
    if len(key) > 2 and key[0] in PREFIXES and (key[1:] in PLACE_LABELS or key[1:] in NAME_LABELS):
        return key[1:]
    return key


def slot_kind(units, k, protected):
    """"place" or "name": what the label before the value that contains unit k announces (None: unknown)."""
    j = k
    while j > 0 and protected.get(j - 1) == "value":
        j -= 1
    if j == 0:
        return None   # the label is on the line before: not guessed
    label = label_key(units[j - 1]["text"])
    if label in PLACE_LABELS:
        return "place"
    if label in NAME_LABELS:
        return "name"
    return None


@lru_cache(maxsize=None)
def get_places():
    """Arabic places of cities.json / places.json as written, by key (ar_key)."""
    from common import places_data
    return {ar_key(arabic): surface(arabic) for arabic in places_data if surface(arabic)}


@lru_cache(maxsize=None)
def get_regions():
    """Region of every place of cities.json, by key (ar_key)."""
    path = Path(BASE_DIR) / "cities.json"
    if not path.is_file():
        return {}
    return {ar_key(city["nameAr"]): (city.get("region") or {}).get("name", "")
            for city in json.loads(path.read_text(encoding="utf-8"))
            if isinstance(city, dict) and city.get("nameAr")}


def line_regions(units, places):
    """Regions of the places written on a line (as read, or one letter away)."""
    regions = get_regions()
    found = set()
    for unit in units:
        key = ar_key(unit["text"])
        for written in ([places[key]] if key in places else close_entries(unit["text"], places)):
            if regions.get(ar_key(written)):
                found.add(regions[ar_key(written)])
    return found


@lru_cache(maxsize=None)
def get_names():
    """First names of names.json as written, by key (ar_key)."""
    path = Path(BASE_DIR) / "names.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {ar_key(arabic): surface(arabic) for arabic in data if surface(arabic)}


def document_words(lines):
    """Every unit of the text: {unit_key: [(line, index)]} (to know whether a word is written elsewhere)."""
    found = {}
    for line_index, line in enumerate(lines):
        for k, unit in enumerate(line_units(line)):
            found.setdefault(unit_key(unit["text"]), []).append((line_index, k))
    return found


def close_entries(word, entries, limit=1):
    """The entries (key -> written form) whose key is at most `limit` letters from the word's key."""
    key = ar_key(word)
    return sorted({written for entry_key, written in entries.items()
                   if abs(len(entry_key) - len(key)) <= limit and edit_distance(key, entry_key) <= limit})


def value_word_suggestions(lines, protected_by_line, changes):
    """A misread word inside a name or a place gets the dictionary's word one letter away:

    - a word of an address ("المستتشفى" -> "المستشفى"): applied (these words are not names);
    - after a place marker, a place of cities.json ("اليوسفة" -> "اليوسفية", "أسي" -> "آسفي");
    - elsewhere, a first name of names.json ("محهد" -> "محمد").
    A place or a name is applied only when the corrected word is written elsewhere in the certificate (the
    same name or place is often written twice); otherwise it is a proposal the person accepts or not.
    Words that already are a known place or name are never touched."""
    places, names = get_places(), get_names()
    written = document_words(lines)
    out = []
    for line_index, protected in protected_by_line.items():
        units = line_units(lines[line_index])
        for k, unit in enumerate(units):
            word = surface(unit["text"])
            key = ar_key(word)
            bare = key[1:] if len(key) > 3 and key[0] in PREFIXES else key   # "بخريبڭة" is "ب" + a city
            if (protected.get(k) != "value" or len(word) < 3 or DIGIT_OR_LATIN.search(word)
                    or not ARABIC_WORD.search(word) or covered(changes, line_index, k)
                    or {key, bare} & (set(places) | set(names) | PLACE_PREFIX_WORDS)):
                continue
            kind = slot_kind(units, k, protected)
            if k and ar_key(units[k - 1]["text"]) in PLACE_PREFIX_WORDS:
                continue   # the second word of a compound place ("اولاد فارس" is not "فاس")
            new, source, accepted, options = None, None, False, []
            address = one_per_key(w for w in ADDRESS_WORDS if abs(len(w) - len(word)) <= 1
                                  and edit_distance(unit_key(word), unit_key(w)) <= (1 if len(w) < 7 else 2))
            if len(address) == 1 and len(word) >= 4:
                new, source, accepted = address[0], "address", True
            elif len(written.get(unit_key(word), [])) > 1:
                continue   # a name or place read the same way twice ("عتيقة ممير" ... "عتيقة ممير"): kept
            elif kind == "place" or (kind == "name" and len(word) >= 4):
                entries, source = (places, "cities.json") if kind == "place" else (names, "names.json")
                close = [w for w in close_entries(word, entries) if unit_key(w) != unit_key(word)]
                if source == "names.json":
                    # a misread letter, not another name: same length ("محهد" / "محمد"), never a letter more or
                    # less ("الإدريسي" is not "ادريس", "عزيزة" is not "عزيز")
                    close = [w for w in close if len(ar_key(w)) == len(ar_key(word))]
                # first the one written elsewhere in the certificate, then a place of the same region as the
                # other places of the line ("اليوسفة ... اقليم أسي": آسفي, Marrakech-Safi, before أسا)
                regions = line_regions(units[:k] + units[k + 1:], places) if source == "cities.json" else set()
                if regions:
                    # a place of another region than the rest of the address is not this place
                    close = [w for w in close if get_regions().get(ar_key(w)) in regions or written.get(unit_key(w))]
                close.sort(key=lambda w: (not written.get(unit_key(w)),
                                          get_regions().get(ar_key(w)) not in regions, w))
                if 1 <= len(close) <= 3:
                    new, options = close[0], close
            if not new or unit_key(new) == unit_key(word):
                continue
            elsewhere = [(l, i) for l, i in written.get(unit_key(new), []) if l != line_index or i != k]
            if source != "address":
                # applied only when the word is written so elsewhere, and no other word is (several options)
                accepted = bool(elsewhere) and not any(written.get(unit_key(w)) for w in options[1:])
            context = {"address": "a word of the address",
                       "cities.json": "a place of cities.json",
                       "names.json": "a first name of names.json"}[source]
            if elsewhere and source != "address":
                context += f", written so on line {elsewhere[0][0] + 1}"
            drop_refused_model_changes(changes, line_index, k)
            reason = "" if accepted else (
                "several are possible: " + " / ".join(options) if len(options) > 1 and not elsewhere
                else "not written so elsewhere in the certificate: check the scan")
            out.append({"line": line_index, "start": k, "end": k + 1, "old": [unit["text"]], "new": [new],
                        "accepted": accepted, "source": source, "context": context, "reason": reason,
                        **({"options": [[w] for w in options]} if len(options) > 1 else {})})
    return out


# ---------------------------------------------------------------------------------------------------------
# coherence of the whole certificate: dates against each other, the address written twice
# ---------------------------------------------------------------------------------------------------------
# ---- the declarant is a parent: the same person's name, written twice ----
PARENT_OF_DECLARANT = {ar_key(w): kind for w, kind in (("والده", "father"), ("والدها", "father"),
                                                        ("والدته", "mother"), ("والدتها", "mother"))}
NAME_STOP = {ar_key(w) for w in ("بن", "بنت", "ابن", "ابنة", "جنسيته", "جنسيتها", "المولود", "المولودة",
                                 "حرفته", "حرفتها", "تحت")}


def name_after(units, k):
    """Indexes of the name written after unit k, up to "بن" / "بنت" / a label / punctuation."""
    out = []
    for j in range(k + 1, len(units)):
        word = surface(units[j]["text"])
        if (ar_key(word) in NAME_STOP or label_kind(word) or DIGIT_OR_LATIN.search(word)
                or not ARABIC_WORD.search(word)):
            break
        out.append(j)
        if units[j]["trail"].strip() or len(out) == 3:
            break   # "رشيد تفاف (والده)": the name ends at the punctuation
    return out


def declarant_names(lines, changes):
    """"صرحت به السيدة نعيمة السكتاني (والدتها)": the declarant is the mother, whose name is also written in
    the header ("و والدتها بهمة السكتاني"). A word that differs between the two readings is the same word: the
    reading that is a first name of names.json is proposed for the other one (applied when two letters at most
    differ); when neither is known, both are flagged."""
    names = get_names()
    text_units = [(i, line_units(line)) for i, line in enumerate(lines)]
    declarant = header = None
    for line_index, units in text_units:
        for k, unit in enumerate(units):
            word = surface(unit["text"])
            if declarant is None and ar_key(word) in ("سيد", "سيده"):
                rest = [ar_key(u["text"]) for u in units[k + 1:]]
                kind = next((PARENT_OF_DECLARANT[r] for r in rest if r in PARENT_OF_DECLARANT), None)
                if kind:
                    declarant = (kind, line_index, name_after(units, k))
    if not declarant or not declarant[2]:
        return []
    kind, declaration_line, declarant_span = declarant
    for line_index, units in text_units:
        if line_index >= declaration_line:
            break
        for k, unit in enumerate(units):
            key = ar_key(unit["text"])
            if PARENT_OF_DECLARANT.get(key) == kind and key in (ar_key("والدته"), ar_key("والدتها"), ar_key("والده"),
                                                                ar_key("والدها")):
                # the header's label: "من والده" / "و والدتها" (not the declarant's own "(والدتها)")
                if header is None:
                    header = (line_index, name_after(units, k))
    if not header or len(header[1]) != len(declarant_span):
        return []
    family = child_names(lines).get("family name")
    latin_family = " ".join(span_words(lines, family["latin"])) if family else ""
    header_units = dict(text_units)[header[0]]
    declarant_units = dict(text_units)[declaration_line]
    out = []
    for hk, dk in zip(header[1], declarant_span):
        a, b = surface(header_units[hk]["text"]), surface(declarant_units[dk]["text"])
        if unit_key(a) == unit_key(b):
            continue
        known_a, known_b = ar_key(a) in names, ar_key(b) in names
        if latin_family and names_agree(a, latin_family) != names_agree(b, latin_family):
            # "لحنيش" / "لحنيسن": the child's NOM LAHNICHE agrees with one reading only
            known_a, known_b = names_agree(a, latin_family), names_agree(b, latin_family)
            why = f"it agrees with the child's {latin_family}"
        else:
            why = None
        if known_a == known_b:
            for (line, k, word, other) in ((header[0], hk, a, b), (declaration_line, dk, b, a)):
                if not covered(changes, line, k):
                    out.append({"line": line, "start": k, "end": k + 1, "old": [word], "new": [other],
                                "accepted": False, "check": True, "source": "declarant",
                                "context": f"the declarant is the {kind}: the same name",
                                "reason": f"the same person is written {other} on another line: click to use it"})
            continue
        line, k, old, new = (header[0], hk, a, b) if known_b else (declaration_line, dk, b, a)
        if covered(changes, line, k):
            continue
        drop_refused_model_changes(changes, line, k)
        distance = edit_distance(unit_key(old), unit_key(new))
        out.append({"line": line, "start": k, "end": k + 1, "old": [old], "new": [new],
                    "accepted": distance <= 2, "check": distance > 2, "source": "declarant",
                    "context": f"the declarant is the {kind}: the same name, and "
                               + (why or f"{new} is a first name of names.json"),
                    "reason": "" if distance <= 2 else f"{distance} letters differ: check the scan"})
    return out


def form_word_suggestions(lines, protected_by_line, changes):
    """A misread word of the printed form outside a value ("عمرهار" -> "عمرها"): the one form word one letter
    away, applied (form words are not names)."""
    out = []
    for line_index, protected in protected_by_line.items():
        for k, unit in enumerate(line_units(lines[line_index])):
            word = surface(unit["text"])
            if (k in protected or len(word) < 4 or DIGIT_OR_LATIN.search(word) or not ARABIC_WORD.search(word)
                    or is_known(word) or covered(changes, line_index, k) or label_kind(word)):
                continue
            # a four-letter word only towards a longer form word ("الرق" -> "الرقم")
            close = one_per_key(w for w in FORM_VOCABULARY if len(w) >= max(4, 9 - len(word))
                                and abs(len(w) - len(word)) <= 1 and edit_distance(unit_key(word), unit_key(w)) <= 1)
            if len(close) != 1:
                continue
            drop_refused_model_changes(changes, line_index, k)
            out.append({"line": line_index, "start": k, "end": k + 1, "old": [unit["text"]], "new": close,
                        "accepted": True, "source": "form", "context": "a word of the form", "reason": ""})
    return out


# ---- the hour and the minute: a lost "الدقيقة" ("الواحدة و العشرين وة: الصفر") ----
def time_words(lines):
    """[(line, unit, text)] after "الساعة", on its line and the next one, up to "ولد" / "ولدت"."""
    for line_index, line in enumerate(lines):
        units = line_units(line)
        start = next((k + 1 for k, u in enumerate(units) if unit_key(u["text"]) in ("الساعة", "ساعة")), None)
        if start is None:
            continue
        out = []
        for at in (line_index, line_index + 1):
            if at >= len(lines):
                break
            for k, unit in enumerate(line_units(lines[at])):
                if at == line_index and k < start:
                    continue
                if surface(unit["text"]) in ("ولد", "ولدت") or label_kind(surface(unit["text"])):
                    return out
                out.append((at, k, unit["text"]))
        return out
    return []


def minute_label(lines, changes):
    """The minute's label lost or misread: the hour phrase then swallows the minute ("الواحدة و العشرين وة
    الصفر" is no hour). The split where both the hour and the minute are phrases of the time dictionary gets
    "و الدقيقة" back, in place of a fragment of it ("وة") or before the minute. Proposed; applied by the record
    judge (propose) when the hour and the minute become readable."""
    from converter import time_value
    words = time_words(lines)
    texts = [surface(w[2]) for w in words]
    if not words or any(edit_distance(unit_key(w), "الدقيقة") <= 2 or "دقيق" in w for w in texts):
        return []

    def readable(phrase):
        return bool(phrase) and isinstance(time_value(" ".join(phrase)), int)

    found = []
    for j in range(1, len(words)):
        hour, rest = texts[:j], texts[j:]
        fragment = (not is_known(rest[0]) and len(rest[0]) <= 4 and rest[0][:1] in ("و", "د", "ق")
                    and set(unit_key(rest[0])) <= set("والدقيقة"))
        minute = rest[1:] if fragment else (rest[1:] if rest[0] == "و" else rest)
        if hour and hour[-1] == "و":
            hour = hour[:-1]
        if readable(hour) and readable(minute) and not readable(texts):
            found.append((j, fragment))
    if len(found) != 1:
        return []   # no split or several: the person reads the scan
    j, fragment = found[0]
    line, k, text = words[j]
    if covered(changes, line, k):
        return []
    drop_refused_model_changes(changes, line, k)
    if fragment:
        new = ["و", "الدقيقة"]
    elif text == "و":
        new = ["و", "الدقيقة"]
    else:
        new = (["الدقيقة"] if texts[j - 1] == "و" else ["و", "الدقيقة"]) + [text]
    return [{"line": line, "start": k, "end": k + 1, "old": [text], "new": new, "accepted": False,
             "source": "time", "context": "the minute's label, lost by the recognition",
             "reason": "not applied: the hour and the minute were not readable with it"}]


def judge_by_record(text, changes, sources):
    """Apply the refused changes of these sources that make the record better (an error less, nothing worse)."""
    accepted = {i for i, c in enumerate(changes) if c["accepted"]}
    errors, mismatches = record_problems(apply_changes(text, changes, accepted))
    for i, change in enumerate(changes):
        if change["accepted"] or change.get("source") not in sources:
            continue
        new_errors, new_mismatches = record_problems(apply_changes(text, changes, accepted | {i}))
        if new_errors < errors and new_mismatches <= mismatches:
            changes[i] = {**change, "accepted": True, "reason": "",
                          "context": change["context"] + ": the record reads one more field"}
            accepted.add(i)
            errors, mismatches = new_errors, new_mismatches


# ---- dates written in words, as on the certificates: the proposal of a date the Hijri one gives ----
UNITS = ("", "واحد", "اثنان", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة", "ثمانية", "تسعة")
TEENS = ("عشرة", "أحد عشر", "اثنا عشر", "ثلاثة عشر", "أربعة عشر", "خمسة عشر", "ستة عشر", "سبعة عشر",
         "ثمانية عشر", "تسعة عشر")
TENS = ("", "", "عشرين", "ثلاثين", "أربعين", "خمسين", "ستين", "سبعين", "ثمانين", "تسعين")
HUNDREDS = ("", "مائة", "مائتين", "ثلاثمائة", "أربعمائة", "خمسمائة", "ستمائة", "سبعمائة", "ثمانمائة", "تسعمائة")
ORDINALS = ("", "فاتح", "ثاني", "ثالث", "رابع", "خامس", "سادس", "سابع", "ثامن", "تاسع", "عاشر")


def year_words(year):
    """2003 -> "ألفين و ثلاثة", 1981 -> "ألف و تسعمائة و واحد و ثمانين"."""
    parts = ["ألف" if year // 1000 == 1 else "ألفين"]
    if (year % 1000) // 100:
        parts.append(HUNDREDS[(year % 1000) // 100])
    rest = year % 100
    if 0 < rest < 10:
        parts.append(UNITS[rest])
    elif 10 <= rest < 20:
        parts.append(TEENS[rest - 10])
    elif rest:
        parts.append(f"{UNITS[rest % 10]} و {TENS[rest // 10]}" if rest % 10 else TENS[rest // 10])
    return " و ".join(parts)


def day_words(day):
    """28 -> "ثامن و عشري", 13 -> "ثالث عشر", 1 -> "فاتح"."""
    if day <= 10:
        return ORDINALS[day]
    if day < 20:
        return ("حادي" if day == 11 else ORDINALS[day - 10]) + " عشر"
    if day % 10 == 0:
        return TENS[day // 10]
    unit = "حادي" if day % 10 == 1 else ORDINALS[day % 10]
    return f"{unit} و " + ("عشري" if day < 30 else "ثلاثين")


def gregorian_words(date):
    return f"{day_words(date.day)} {GREGORIAN_MONTHS[date.month - 1]} سنة {year_words(date.year)}"


def reads_as(field, phrase):
    """The value the converter reads from this phrase for a date field ("2003-11-28")."""
    import converter
    from common import TEMPLATE
    return str(converter.convert({**TEMPLATE, field: phrase}).get(field, ""))


def year_proposal(text, changes, record, lines, line_index, hijri, gregorian, h, g):
    """"ثاني مارس سنة ... تسعة و ستين" next to 10 Dhu al-Qa'da 1385 (2 March 1966): the day and the month are
    those of the Hijri date, only the year differs. The year the Hijri date gives is proposed in orange (never
    applied by the rules: the registry sometimes writes another year on purpose)."""
    import datetime
    try:
        expected = hijri_to_gregorian(*map(int, h.split("-")))
        read = datetime.date(*map(int, g.split("-")))
    except ValueError:
        return None
    if read.year == expected.year:
        return None
    try:
        same_day = abs((read.replace(year=expected.year) - expected).days) <= 2
    except ValueError:
        return None
    if not same_day:
        return None   # the day and month differ too: not a misread year
    units = line_units(lines[line_index])
    start = next((k + 1 for k, u in enumerate(units) if ar_key(u["text"]) in ("سنه", "لسنه")), None)
    end = next((k for k, u in enumerate(units) if start is not None and k >= start
                and surface(u["text"]).startswith("ميلادي")), len(units))
    if start is None or end <= start or any(c["line"] == line_index and c["start"] < end
                                            and max(c["end"], c["start"] + 1) > start for c in changes):
        return None
    new = year_words(expected.year).split()
    change = {"line": line_index, "start": start, "end": end, "old": [u["text"] for u in units[start:end]],
              "new": new, "accepted": True}
    base = {i for i, c in enumerate(changes) if c["accepted"]}
    trial = record_of(apply_changes(text, changes + [change], base | {len(changes)}))[0]
    if str(trial.get(gregorian)) != read.replace(year=expected.year).isoformat():
        return None
    return {**change, "accepted": False, "check": True, "source": "dates",
            "context": f"the year of the Hijri date {h}",
            "reason": f"the day and the month are those of the Hijri date ({expected.isoformat()}), only the year "
                      f"differs ({read.year}): click to use {expected.year}"}


GREGORIAN_MONTHS = ("يناير", "فبراير", "مارس", "أبريل", "ماي", "يونيو", "يوليوز", "غشت", "شتنبر", "أكتوبر",
                    "نونبر", "دجنبر")


def date_flags(text, changes):
    """A Gregorian date that cannot be read next to a Hijri date that can: flagged with the date the Hijri one
    gives ("≠ 28 نونبر 2003"), so the person knows what to look for on the scan. The Gregorian dates are
    found in the order of the form: the child, the father, the mother, the declaration ("موافق" lines)."""
    record, _ = record_of(apply_changes(text, changes))
    lines = text.split("\n")
    agree_lines = [i for i, line in enumerate(lines) if re.search(r"(?:^|\s)و?الموافق|(?:^|\s)موافق", line)]
    order = ("DATENAISSAR", "DATENAISSPEREAR", "DATENAISSMEREAR", "DATEDECLAAR")
    present = [h for h in order if str(record.get(h, "")).strip()]
    if len(agree_lines) != len(present):
        return []   # the form is not the usual one: no guess about which line is which date
    flags = []
    gregorian_of = dict(DATE_PAIRS)
    for hijri, line_index in zip(present, agree_lines):
        h, g = str(record.get(hijri, "")), str(record.get(gregorian_of[hijri], ""))
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", h) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", g):
            year_check = year_proposal(text, changes, record, lines, line_index, hijri, gregorian_of[hijri], h, g)
            if year_check:
                flags.append(year_check)
            continue
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", h):
            continue   # (a lone year next to a full Hijri date: the day and month were not read)
        try:
            expected = hijri_to_gregorian(*map(int, h.split("-")))
        except ValueError:
            continue
        units = line_units(lines[line_index])
        start = next((k + 1 for k, u in enumerate(units) if re.search("موافق", u["text"])), 0)
        end = next((k for k, u in enumerate(units) if k >= start and surface(u["text"]).startswith("ميلادي")),
                   len(units))
        last = max((k for k, u in enumerate(units) if surface(u["text"]).startswith("ميلادي")), default=len(units))
        if end <= start:
            continue
        for k in range(start, max(end, last)):   # the refused proposals under the date: the date says it all
            drop_refused_model_changes(changes, line_index, k)
            for change in [c for c in changes if c["line"] == line_index and c["start"] == k and not c["accepted"]
                           and c.get("source") == "date_time.json"]:
                changes.remove(change)
        if any(c["line"] == line_index and c["start"] < max(end, last) and max(c["end"], c["start"] + 1) > start
               for c in changes):
            continue   # another change on these words: shown alone (no overlapping marks)
        told = f"{expected.day} {GREGORIAN_MONTHS[expected.month - 1]} {expected.year}"
        # the words of that date (and the day before / after: the calendar of Morocco can differ by a day): the
        # misread words up to the last "ميلادية" of the line are replaced ("ثامن من دعر رسنة ميلادية الفينة"),
        # or up to the end of the line when the year is written on the next one ("سامع عشر دجرة" / "سنة ...")
        import datetime
        milad = [k for k, u in enumerate(units) if surface(u["text"]).startswith("ميلادي")]
        last = max(milad) if milad else len(units)
        # the whole date when the line ends with "ميلادية"; the day and month only when the year is on the next line
        with_year = bool(milad) or any(ar_key(u["text"]) in ("سنه", "لسنه") for u in units[start:last])
        old = [u["text"] for u in units[start:last]]
        options = []
        if last > start:
            base = {i for i, c in enumerate(changes) if c["accepted"]}
            for shift in (0, -1, 1):
                date = expected + datetime.timedelta(days=shift)
                words = (gregorian_words(date) if with_year else
                         f"{day_words(date.day)} {GREGORIAN_MONTHS[date.month - 1]}").split()
                trial = changes + [{"line": line_index, "start": start, "end": last, "old": old, "new": words,
                                    "accepted": True}]
                # kept only when the record read with it has this date (the converter reads the words back)
                if str(record_of(apply_changes(text, trial, base | {len(changes)}))[0].get(gregorian_of[hijri])) \
                        == date.isoformat():
                    options.append(words)
        if options:
            # the day closest to what was read first ("سامع عشر": the 17th, not the 18th)
            read = unit_key(" ".join(old))
            options.sort(key=lambda o: -SequenceMatcher(None, read, unit_key(" ".join(o))).ratio())
            likeness = SequenceMatcher(None, read, unit_key(" ".join(options[0]))).ratio()
            # applied when the reading and the Hijri date confirm each other (a few letters misread);
            # otherwise a check: shown in orange, applied by a click
            accepted = likeness >= 0.75
            flags.append({"line": line_index, "start": start, "end": last, "old": old, "new": options[0],
                          "options": options, "accepted": accepted, "check": not accepted, "source": "dates",
                          "context": f"the date of the Hijri date {h}" + (", close to what was read" if accepted else ""),
                          "reason": "" if accepted else
                                    f"the Gregorian date cannot be read; the Hijri date {h} falls on {told} "
                                    "(± 1 day, click again for the day before / after)"})
            continue
        flags.append({"line": line_index, "start": start, "end": end,
                      "old": [u["text"] for u in units[start:end]], "new": [u["text"] for u in units[start:end]],
                      "accepted": False, "flag": True, "source": "dates", "context": f"≠ {told}",
                      "reason": f"cannot be read as a date; the Hijri date {h} falls on {told} (± 2 days)"})
    return flags


ADDRESS_NUMBER_LABELS = frozenset(ar_key(w) for w in ("زنقة", "رقم", "عمارة", "الشطر", "شقة", "درب", "بلوك"))


def address_numbers(lines, start_words, stop_words):
    """{label: (number, line, unit)} of the first address after one of start_words, up to a stop word."""
    numbers, inside = {}, False
    for line_index, line in enumerate(lines):
        units = line_units(line)
        for k, unit in enumerate(units):
            key = ar_key(unit["text"])
            if not inside:
                inside = key in start_words
                continue
            if key in stop_words or (numbers and key in start_words):
                return numbers
            label = next((l for l in ADDRESS_NUMBER_LABELS if key == l or (len(key) >= 2 and edit_distance(key, l) <= 1
                                                                           and len(l) >= 3)), None)
            number = next((units[j] for j in range(k + 1, min(k + 3, len(units))) if units[j]["text"].isdigit()), None)
            if label and number is not None:
                numbers.setdefault(label, (number["text"], line_index, units.index(number)))
        if inside and numbers and any(ar_key(u["text"]) in ("بناء", "بناءا") for u in units):
            return numbers
    return numbers


def address_flags(lines, changes):
    """The parents' address and the declarant's address: the same street / house number twice. A number
    that differs ("الزنقة 11" / "الزنقة 12") is flagged on both (digits are never corrected: the scan tells
    which one is right)."""
    stop = {ar_key(w) for w in ("حرفتها", "حرفته", "بناءا", "بناء", "وحرر", "حرر", "تحت")}
    parents = address_numbers(lines, {ar_key("الساكنان")}, stop | {ar_key("الساكن")})
    declaration_lines = next((i for i, l in enumerate(lines) if re.search(r"بناء", l)), None)
    if not parents or declaration_lines is None:
        return []
    declarant = address_numbers(lines[declaration_lines:], {ar_key("الساكن"), ar_key("الساكنة")},
                                {ar_key("وحرر"), ar_key("حرر")})
    flags = []
    for label, (number, line_index, k) in parents.items():
        other = declarant.get(label)
        if not other or other[0] == number:
            continue
        other_line = other[1] + declaration_lines
        for (value, at_line, at_unit), against in (((number, line_index, k), other[0]),
                                                    ((other[0], other_line, other[2]), number)):
            if covered(changes, at_line, at_unit):
                continue
            flags.append({"line": at_line, "start": at_unit, "end": at_unit + 1, "old": [value], "new": [against],
                          "accepted": False, "check": True, "source": "address",
                          "context": "the number of the other address",
                          "reason": f"the same address gives {against} on another line: click to use it"})
    return flags


# labels that the HTR sometimes glues to the previous word ("العربجننسيته" = "العربي" + "جنسيته")
GLUED_LABELS = ("جنسيته", "جنسيتها", "حرفته", "حرفتها", "مهنته", "مهنتها", "المولود", "المولودة",
                "الساكن", "الساكنان")


def glued_labels(lines, changes):
    """A label glued to the word before it is split again: "العربجننسيته" -> "العرب جنسيته" (the label also
    corrected, one letter at most; the first word is kept as read). Without the space the parser reads the
    nationality as a part of the name. Applied by default; any model change on the word is replaced."""
    out = []
    for line_index, line in enumerate(lines):
        for k, unit in enumerate(line_units(line)):
            word = surface(unit["text"])
            if len(word) < 7 or not ARABIC_WORD.search(word):
                continue
            best = None
            for label in GLUED_LABELS:
                for size in (len(label) - 1, len(label), len(label) + 1):
                    head, tail = word[:-size], word[-size:]
                    if len(head) < 2 or size <= 0 or tail[0] != label[0]:
                        continue   # the label starts where its first letter is ("عادل" + "جننسيته")
                    distance = edit_distance(tail, label)
                    if distance <= 1 and (best is None or (distance, -len(label)) < (best[0], -len(best[2]))):
                        best = (distance, head, label)
            if best and not is_known(word):
                for change in [c for c in changes if c["line"] == line_index and c["start"] <= k < max(c["end"], c["start"] + 1)]:
                    changes.remove(change)
                out.append({"line": line_index, "start": k, "end": k + 1, "old": [unit["text"]],
                            "new": [best[1], best[2]], "accepted": True, "reason": "", "source": "split",
                            "context": "a label glued to the word before it"})
    return out


def hijri_to_gregorian(year, month, day):
    """Tabular (arithmetic) Hijri calendar -> Gregorian date: one or two days from the observed calendar of
    Morocco, which is close enough to tell "ثالث" (3rd) from "ثامن" (8th)."""
    import datetime
    import math
    jd = day + math.ceil(29.5 * (month - 1)) + (year - 1) * 354 + (3 + 11 * year) // 30 + 1948439.5 - 1
    return datetime.date.fromordinal(int(jd - 1721424.5))


DATE_PAIRS = (("DATENAISSAR", "DATENAISSFR"), ("DATEDECLAAR", "DATEDECLAFR"),
              ("DATENAISSPEREAR", "DATENAISSPEREFR"), ("DATENAISSMEREAR", "DATENAISSMEREFR"))


def record_of(text):
    """The record the rules read from this text (no NER, no translation) and its validation issues."""
    import converter
    import mapper
    import parser as record_parser
    import validator

    record = mapper.map_fields(converter.convert(record_parser.parse(text, None)))
    return record, validator.validate(record, check_translation=False)["issues"]


def record_problems(text):
    """(errors, Gregorian / Hijri mismatches) of the record the rules read from this text (no NER, no translation)."""
    _, issues = record_of(text)
    errors = sum(1 for i in issues if i["level"] == "error")
    mismatches = sum(1 for i in issues if "do not match" in i["message"])
    return errors, mismatches


def day_gaps(text):
    """{date pair: days between its full Hijri date and its Gregorian date} for the record read from the text."""
    import datetime
    record, _ = record_of(text)
    gaps = {}
    for hijri, gregorian in DATE_PAIRS:
        h, g = str(record.get(hijri, "")), str(record.get(gregorian, ""))
        if not (re.fullmatch(r"\d{4}-\d{2}-\d{2}", h) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", g)):
            continue
        try:
            converted = hijri_to_gregorian(*map(int, h.split("-")))
            gaps[hijri] = abs((datetime.date(*map(int, g.split("-"))) - converted).days)
        except ValueError:
            continue
    return gaps


def record_dates(text):
    """The dates and times the record reads from this text (to tell options that change nothing apart)."""
    record, _ = record_of(text)
    return tuple(str(record.get(field, "")) for pair in DATE_PAIRS for field in pair) + (
        str(record.get("HEURENAISS", "")), str(record.get("MINNAISS", "")))


def joint_date_fixes(text, changes):
    """Two misread words of the same date ("سامع عشر دجرة": day and month): neither makes the date valid alone,
    so they are judged together. The pair (and its options) that makes the record better and whose Hijri date
    falls on its Gregorian date (3 days at most) is applied when it is the only one."""
    words = date_words()
    candidates = [i for i, c in enumerate(changes)
                  if not c["accepted"] and c.get("source") == "date_time.json" and c["old"] and c["new"]
                  and all(all(surface(w) in words for w in o) for o in (c.get("options") or [c["new"]]))]
    if len(candidates) < 2:
        return
    accepted = {i for i, c in enumerate(changes) if c["accepted"]}
    errors, mismatches = record_problems(apply_changes(text, changes, accepted))
    from itertools import combinations, product
    for a, b in combinations(candidates, 2):
        if changes[a]["line"] != changes[b]["line"] or changes[a]["accepted"] or changes[b]["accepted"]:
            continue
        good = []
        for oa, ob in product(changes[a].get("options") or [changes[a]["new"]],
                              changes[b].get("options") or [changes[b]["new"]]):
            trial_changes = list(changes)
            trial_changes[a] = {**changes[a], "new": oa}
            trial_changes[b] = {**changes[b], "new": ob}
            trial = apply_changes(text, trial_changes, accepted | {a, b})
            new_errors, new_mismatches = record_problems(trial)
            gaps = day_gaps(trial)
            if (new_errors < errors and new_mismatches <= mismatches and gaps
                    and max(gaps.values()) <= 3):
                good.append((oa, ob, new_errors, new_mismatches))
        if len({(tuple(oa), tuple(ob)) for oa, ob, _, _ in good}) == 1:
            oa, ob, errors, mismatches = good[0]
            for i, option in ((a, oa), (b, ob)):
                options = changes[i].get("options")
                changes[i] = {**changes[i], "new": option, "accepted": True, "reason": "",
                              "context": "with the other misread word of this date, it agrees with the Hijri date",
                              **({"options": [option] + [o for o in options if o != option]} if options else {})}
                accepted.add(i)


def date_fixes(text, changes):
    """Accept the number / month corrections that make a date possible.

    "سنة ألفين و ثمانين" (2080) next to the Hijri year 1428 is a misreading of "ثمانية" (2008), although
    "ثمانين" is a correct word. A refused change towards number or month words (1 or 2 letters) is accepted when
    the record read with it has fewer errors (an invalid date, a year in the future, a declaration before the
    birth ...) and no new Gregorian / Hijri mismatch: the validator of the whole record is the judge. When
    several words are possible ("ثالن": "ثالث" or "ثامن"), the one whose Hijri date falls on its Gregorian date
    (3 days at most, and 3 days better than the others) is chosen.
    """
    words = date_words()
    candidates = [i for i, c in enumerate(changes)
                  if not c["accepted"] and c["old"] and c["new"] and c.get("source") not in ("jobs.json", "names")
                  and not c.get("flag")
                  and all(all(surface(w) in words for w in option) for option in (c.get("options") or [c["new"]]))
                  and edit_distance("".join(map(unit_key, c["old"])), "".join(map(unit_key, c["new"]))) <= 2]
    if not candidates:
        return
    accepted = {i for i, c in enumerate(changes) if c["accepted"]}
    errors, mismatches = record_problems(apply_changes(text, changes, accepted))
    for i in candidates:
        options = changes[i].get("options") or [changes[i]["new"]]
        good = []
        for option in options:
            trial_changes = list(changes)
            trial_changes[i] = {**changes[i], "new": option}
            trial = apply_changes(text, trial_changes, accepted | {i})
            new_errors, new_mismatches = record_problems(trial)
            # better: an error less, or a Gregorian / Hijri mismatch less, and nothing worse
            if (new_errors <= errors and new_mismatches <= mismatches
                    and (new_errors, new_mismatches) != (errors, mismatches)):
                good.append((option, trial, new_errors, new_mismatches))
        context = "the date becomes possible and agrees with the Hijri date"
        if len(good) > 1:
            # options that give the same dates ("ثلاثة" / "ثلاث" / "ثلاثه": 2003) are one choice: the closest
            # spelling to what was read is kept
            by_dates = {}
            for entry in good:
                by_dates.setdefault(record_dates(entry[1]), []).append(entry)
            if len(by_dates) == 1:
                old = "".join(map(unit_key, changes[i]["old"]))
                # a misread letter before a lost one ("ثلاثن": "ثلاثة", not "ثلاث")
                good = [min(good, key=lambda e: (edit_distance(old, "".join(map(unit_key, e[0]))),
                                                 abs(len(old) - len("".join(map(unit_key, e[0])))), e[0]))]
        if len(good) > 1:
            # only the date pairs the options change are compared (another pair may be off for its own reasons)
            per_option = [day_gaps(trial) for _, trial, _, _ in good]
            pairs = {p for gaps in per_option for p in gaps
                     if len({gaps_.get(p) for gaps_ in per_option}) > 1 and all(p in g for g in per_option)}
            if not pairs:
                continue   # the dates cannot tell: the person chooses among the options
            totals = sorted((sum(gaps[p] for p in pairs), n) for n, gaps in enumerate(per_option))
            best_gap, best = totals[0]
            if best_gap > 3 * len(pairs) or any(total < best_gap + 3 for total, _ in totals[1:]):
                continue
            good = [good[best]]
            context = f"the Hijri date falls on the Gregorian date ({best_gap} day(s) apart)"
        if len(good) == 1:
            option, _, errors, mismatches = good[0]
            rest = [o for o in options if o != option]
            changes[i] = {**changes[i], "new": option, "accepted": True, "reason": "", "context": context,
                          **({"options": [option] + rest} if changes[i].get("options") else {})}
            accepted.add(i)


def correct_text(text, model=None):
    """Text -> (text with the corrections accepted by the rules, notes). The line structure never changes."""
    changes, notes = propose(text, model)
    notes = notes + [describe(change) for change in changes]
    return apply_changes(text, changes), notes or ["spelling check: no correction"]


def main():
    parser = argparse.ArgumentParser(description="Correct the HTR spelling mistakes with a local model (Ollama).")
    parser.add_argument("input_file", type=Path, help="text written by recognition.py")
    parser.add_argument("-o", "--output", type=Path, help="write the corrected text here instead of stdout")
    parser.add_argument("--model", default=MODEL, help=f"Ollama model (default {MODEL})")
    args = parser.parse_args()
    if not args.input_file.is_file():
        parser.error(f"Input file not found: {args.input_file}")
    text, notes = correct_text(read_text_file(args.input_file), args.model)
    for note in notes:
        log(f"[spelling] {note}")
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()