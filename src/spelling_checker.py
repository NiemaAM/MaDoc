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
def surface(token):
    """A word as written, without diacritics / tatweel / invisible characters.

    Unlike ar_key the letters are NOT unified: "جنسيتة" and "جنسيته" are different words here, because the
    parser needs the right letter ("ة" / "ه" / "ى" / hamza are exactly what the HTR confuses).
    """
    return ARABIC_DIACRITICS.sub("", INVISIBLE_CHARS.sub("", token)).strip()


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


def is_value_label(word):
    """A label after which a value (name, place, address, job) is written."""
    return label_kind(word) is not None


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
                continue   # a word added by the model: it has no place in the original, never proposed
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
    source = " (jobs.json" + (", " + change["context"] if change.get("context") else "") + ")" \
        if change.get("source") == "jobs.json" else ""
    if change["accepted"]:
        return f"line {change['line'] + 1}: {before!r} -> {after!r}{source}"
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


def merge_line(original, proposed, line_number, notes, vocabulary=None, protected=None):
    """The original line with only the changes accepted by the rules applied (notes: every change)."""
    changes = line_changes(original, proposed, 0, vocabulary, protected)
    for change in changes:
        notes.append(describe({**change, "line": line_number - 1}))
    return apply_changes(original, changes) if changes else original


# ---------------------------------------------------------------------------------------------------------
# local model (Ollama)
# ---------------------------------------------------------------------------------------------------------
PROMPT = """You correct the spelling of Moroccan Arabic birth certificates read by a handwriting recognition model.
The user sends a JSON object {"lines": [...]}, one string per line of the document.

Correct ONLY spelling mistakes made by the reading model:
- a missing, extra, doubled or wrong letter in a word ("البيتت" -> "البيت", "موفق" -> "موافق"),
- a word cut in two ("ميلاي ية" -> "ميلادية"),
- a misread profession after "حرفته" / "حرفتها" / "مهنته" / "مهنتها": write the usual Arabic name of the
  profession it most probably is ("صباخ" -> "صباغ").
NEVER change: personal names, family names, place names you are not sure of, numbers, digits, Latin text
(PRENOM, NOM and the French names), punctuation, the order of the words. Never add or remove a word, never
rephrase, never translate, never complete missing information. When unsure, keep the word as it is.

Return ONLY a JSON object {"lines": [...]} with exactly as many strings as the input, in the same order."""


def ollama_url(path):
    """Address of the Ollama API; refuses anything that is not this computer unless explicitly allowed."""
    host = urlparse(OLLAMA_URL).hostname or ""
    if host not in LOCAL_HOSTS and os.environ.get("MADOC_SPELL_ALLOW_REMOTE") != "1":
        raise PermissionError(
            f"MADOC_OLLAMA_URL={OLLAMA_URL!r} is not this computer: the documents must not leave it "
            "(set MADOC_SPELL_ALLOW_REMOTE=1 only for a server of your own network)"
        )
    return OLLAMA_URL.rstrip("/") + path


def ask_model(lines, model=None):
    """Lines -> the model's corrected lines (list of str), or raise."""
    body = {
        "model": model or MODEL,
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": json.dumps({"lines": lines}, ensure_ascii=False)},
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
    proposed = json.loads(content).get("lines")
    if not isinstance(proposed, list) or not all(isinstance(line, str) for line in proposed):
        raise ValueError("the model did not return a list of lines")
    return proposed


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
        if len(proposed) != len(positions):
            notes.append(f"spelling model skipped: it returned {len(proposed)} lines for {len(positions)}")
            proposed = None
    except PermissionError as e:
        notes.append(f"spelling model skipped: {e}")
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        notes.append(f"spelling model skipped: Ollama not reachable at {OLLAMA_URL} ({e})")
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        notes.append(f"spelling model skipped: unusable answer from the model ({e})")
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
    return sorted(changes, key=lambda c: (c["line"], c["start"])), notes


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