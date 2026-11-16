"""Page 1 of an act: the act number, the child's names (Arabic and French) and the marginal notes.

The page ("البيانات الهامشية") has a short printed header:

    الصفحة رقم : ...
    البيانات الهامشية
    رسم ولادة رقم : 159
    الاسم الشخصي : هشام          PRENOM HICHAM
    الاسم العائلي : الإدريسي       NOM IDRISSI

and, under it, ruled lines where the marginal notes are written by hand (marriage, death, rectification...; the
French names of the parents are sometimes written there). parse() reads the header fields and keeps every line
under the header as marginal notes: the Arabic lines in TEXTEMENTIONS, the lines in Latin script in
TEXTEMENTIONSFR. merge() puts them in the record read from page 2 and compares what both pages say.
"""

import re

from common import DIGIT_MAP, ar_key, normalize_text
from parser import latin_field

HEADER_LINES = 12     # the header fields are looked for in the first lines only: a note may quote another act
PAGE_NUMBER = re.compile(r"الصفحة\s+رقم")
TITLE = re.compile(r"البيانات\s+الهامشية")
ACT = re.compile(r"رسم\s+(?:ال)?ولادة\s+رقم\s*[:：]?\s*([0-9٠-٩۰-۹]+)?")
FIRST_NAME = re.compile(r"ال[إأا]سم\s+الشخصي\s*[:：]?\s*(.*)")
FAMILY_NAME = re.compile(r"ال[إأا]سم\s+العائلي\s*[:：]?\s*(.*)")
LATIN_LABEL = re.compile(r"(?<![A-Za-z])(?:PRENOM|NOM)(?![A-Za-z]).*")
LATIN = re.compile(r"[A-Za-zÀ-ÿ]")
ARABIC = re.compile(r"[؀-ۿ]")
# the French names of the parents, when a note gives them
PARENTS = re.compile(r"(?i)\b(?:fils|fille)\s+de\s+(.+?)\s+et\s+de\s+(.+)")
FATHER = re.compile(r"(?i)(?<![a-zà-ÿ])p[eèé]re\s*[:：]?\s*(.+)")
MOTHER = re.compile(r"(?i)(?<![a-zà-ÿ])m[eèé]re\s*[:：]?\s*(.+)")
COMPARED = ("NUMACTE", "PRENOMAR", "PRENOM", "NOMAR", "NOM")
FIELDS = (*COMPARED, "TEXTEMENTIONS", "TEXTEMENTIONSFR")


def is_latin(text):
    return len(LATIN.findall(text)) > len(ARABIC.findall(text))


def clean_name(value):
    return normalize_text(value).strip(" .:،-_")


def arabic_value(match):
    """The Arabic value after a label: without the Latin label and name the same line may hold."""
    return clean_name(LATIN_LABEL.sub("", match.group(1)))


def parse(texts):
    """The text lines of page 1, in reading order -> {"fields": {...}, "sources": {field: [line indexes]},
    "parents": {"father": "", "mother": ""}}."""
    fields = {field: "" for field in FIELDS}
    sources = {}
    header_end = -1

    def found(field, value, index):
        nonlocal header_end
        if value and not fields[field]:
            fields[field] = value
            sources[field] = [index]
        header_end = max(header_end, index)

    for index, raw in enumerate(texts[:HEADER_LINES]):
        line = normalize_text(raw)
        if not line:
            continue
        if PAGE_NUMBER.search(line) or TITLE.search(line):
            header_end = max(header_end, index)
        match = ACT.search(line)
        if match:
            found("NUMACTE", (match.group(1) or "").translate(DIGIT_MAP), index)
        match = FIRST_NAME.search(line)
        if match:
            found("PRENOMAR", arabic_value(match), index)
        match = FAMILY_NAME.search(line)
        if match:
            found("NOMAR", arabic_value(match), index)
        for label, field in (("PRENOM", "PRENOM"), ("NOM", "NOM")):
            value = latin_field(label, line + " ")
            if value:
                found(field, value.upper(), index)
        if fields["NOM"] and fields["NOMAR"]:
            break     # the last line of the header
    arabic, french, arabic_lines, french_lines = [], [], [], []
    for index, raw in enumerate(texts):
        line = normalize_text(raw)
        if index <= header_end or not line:
            continue
        if is_latin(line):
            french.append(line)
            french_lines.append(index)
        else:
            arabic.append(line)
            arabic_lines.append(index)
    fields["TEXTEMENTIONS"], fields["TEXTEMENTIONSFR"] = "\n".join(arabic), "\n".join(french)
    if arabic_lines:
        sources["TEXTEMENTIONS"] = arabic_lines
    if french_lines:
        sources["TEXTEMENTIONSFR"] = french_lines
    parents = {"father": "", "mother": ""}
    for line in french:
        match = PARENTS.search(line)
        if match:
            parents["father"] = parents["father"] or clean_name(match.group(1)).upper()
            parents["mother"] = parents["mother"] or clean_name(match.group(2)).upper()
            continue
        for key, pattern in (("father", FATHER), ("mother", MOTHER)):
            match = pattern.search(line)
            if match and not parents[key]:
                parents[key] = clean_name(match.group(1)).upper()
    return {"fields": fields, "sources": sources, "parents": parents}


def same(field, a, b):
    """Do the two pages give the same value (spelling variants of the Arabic letters and case apart)?"""
    if field == "NUMACTE":
        return str(a).translate(DIGIT_MAP).strip() == str(b).translate(DIGIT_MAP).strip()
    if field in ("PRENOM", "NOM"):
        return normalize_text(a).casefold() == normalize_text(b).casefold()
    return ar_key(a) == ar_key(b)


def merge(record, page_one):
    """The record read from page 2 with what page 1 gives -> (record, comparisons, notes).

    Page 2 is the reference: a field it gives is kept, and a different value on page 1 is reported; a field it
    does not give is taken from page 1. The marginal notes come from page 1 only. A French name of a parent
    written on page 1 replaces the translated one (the clerk wrote it)."""
    record = dict(record)
    comparisons, notes = [], []
    fields = page_one["fields"]
    for field in COMPARED:
        first, second = fields[field], record.get(field, "")
        second = "" if second is None else str(second)
        if not first:
            continue
        if not second:
            record[field] = first
            notes.append(f"[page 1] {field} not read on page 2: taken from page 1")
            comparisons.append({"field": field, "page1": first, "page2": "", "same": None})
        else:
            comparisons.append({"field": field, "page1": first, "page2": second, "same": same(field, first, second)})
    record["TEXTEMENTIONS"], record["TEXTEMENTIONSFR"] = fields["TEXTEMENTIONS"], fields["TEXTEMENTIONSFR"]
    for key, field in (("father", "PRENOMPEREFR"), ("mother", "PRENOMMEREFR")):
        if page_one["parents"][key]:
            record[field] = page_one["parents"][key]
            notes.append(f"[page 1] {field}: the French name written on page 1 is used")
    return record, comparisons, notes


def issues(comparisons):
    """The validation warnings of the values that differ between the two pages."""
    return [{"level": "warning", "field": c["field"],
             "message": f"page 1 and page 2 differ: {c['page1']!r} (page 1), {c['page2']!r} (page 2)"}
            for c in comparisons if c["same"] is False]
