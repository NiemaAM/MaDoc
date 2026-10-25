"""Stage 3 - converter: JSON object -> JSON object with dates, time and sex as numbers.

    python src/converter.py parsed.json [-o converted.json]

Dates become YYYY-MM-DD (or YYYY when the document only gives a year), hours / minutes become integers,
SEXE becomes 1 (male) or 0 (female). Anything that cannot be converted is left as it was.
"""

import argparse
import copy
import re
from functools import lru_cache
from pathlib import Path

from common import ar_key, dates_data, normalize_text, read_json_file, write_json


# alias builder for converter
def build_alias_map(data):
    """Build a mapping from aliases to canonical values."""
    aliases = {}
    def walk(obj):
        if not isinstance(obj, dict):
            return
        for key, value in obj.items():
            if isinstance(value, list):
                for alias in value:
                    if isinstance(alias, str):
                        aliases[normalize_text(alias)] = str(key)
            elif isinstance(value, dict):
                walk(value)
    walk(data)
    return aliases


def build_key_map(data):
    """Same as build_alias_map but the aliases are normalised with ar_key."""
    return {ar_key(alias): value for alias, value in build_alias_map(data).items() if ar_key(alias)}


@lru_cache(maxsize=None)
def get_number_keys():
    """Number aliases (dates.json 'numbers') -> int."""
    keys = {}
    for alias, value in build_key_map({"numbers": dates_data.get("numbers", {})}).items():
        if str(value).isdigit():
            keys[alias] = int(value)
    # ordinals that may be missing from the dictionary (the dictionary always wins)
    for alias, value in {"عاشر": 10, "عاشره": 10}.items():
        keys.setdefault(ar_key(alias), value)
    return keys


_shapes = {}


def key_shape(keys):
    """(first words, longest key in words) of a number / month table, computed once per table.

    The tables are built once (lru_cache) and never modified, so the result is kept by table identity.
    """
    cached = _shapes.get(id(keys))
    if cached is None or cached[0] is not keys:
        shape = (frozenset(key.split()[0] for key in keys), max((len(key.split()) for key in keys), default=0))
        cached = _shapes[id(keys)] = (keys, shape)
    return cached[1]


def spaceless(phrase):
    """Key used to match a time phrase with the dictionary: normalised, without spaces ("واحد و عشرين" = "واحد وعشرين")."""
    return ar_key(phrase).replace(" ", "")


@lru_cache(maxsize=None)
def get_time_keys():
    """dates.json 'time' section: whole phrase (spaces ignored) -> int. No addition, the phrase must be listed.

    When two values list the same phrase the lowest value keeps it (the dictionary has a few such typos).
    """
    keys = {}
    time_section = dates_data.get("time", {})
    for value in sorted(time_section, key=lambda v: int(v) if str(v).isdigit() else 10 ** 9):
        if not str(value).isdigit():
            continue
        for alias in time_section[value]:
            if isinstance(alias, str) and spaceless(alias):
                keys.setdefault(spaceless(alias), int(value))
    return keys


@lru_cache(maxsize=None)
def get_month_keys():
    """Hijri + miladi month aliases -> month number."""
    keys = {}
    for section in ("hijri_months", "miladi_months"):
        for alias, value in build_key_map({section: dates_data.get(section, {})}).items():
            if str(value).isdigit():
                keys[alias] = int(value)
    return keys


def phrase_to_number(phrase, number_keys):
    """Convert a written number phrase to an int (unknown words ignored, None if nothing known)."""
    tokens = ar_key(phrase).split()
    if not tokens or not number_keys:
        return None
    first_tokens, max_len = key_shape(number_keys)
    # split the glued conjunction: "وأربعمائة" -> "و" + "اربعمائة"
    expanded = []
    for token in tokens:
        if token in first_tokens or token.isdigit():
            expanded.append(token)
        elif len(token) > 1 and token.startswith("و") and (
            token[1:] in first_tokens or token[1:].isdigit()
        ):
            expanded.extend(["و", token[1:]])
        else:
            expanded.append(token)
    total = 0
    found = False
    lost = False   # an unknown word right after "و": a misread number ("ألفين و ثلاثن" is not 2000)
    i = 0
    while i < len(expanded):
        token = expanded[i]
        if token.isdigit():
            total += int(token)
            found = True
            i += 1
            continue
        # prefer the longest alias first ("ثالث عشر" before "ثالث")
        for n in range(min(max_len, len(expanded) - i), 0, -1):
            candidate = " ".join(expanded[i:i + n])
            if candidate in number_keys:
                total += number_keys[candidate]
                found = True
                i += n
                break
        else:
            if found and i > 0 and expanded[i - 1] == "و" and not token.isdigit():
                lost = True
            i += 1
    return None if lost else (total if found else None)


# dates builder for converter
def replace_aliases(value):
    """Convert a written date into numeric day, month, and year."""
    if not value:
        return value
    result = normalize_text(value)
    number_keys = get_number_keys()
    month_keys = get_month_keys()
    if not month_keys:
        return result
    tokens = ar_key(result).split()
    max_len = key_shape(month_keys)[1]
    # find every month alias in the phrase
    matches = []
    for i in range(len(tokens)):
        for n in range(min(max_len, len(tokens) - i), 0, -1):
            if " ".join(tokens[i:i + n]) in month_keys:
                matches.append((n, i))
    # Prefer longer month aliases ("ربيع الثاني" before "ربيع"), then the first one.
    for n, i in sorted(matches, key=lambda m: (-m[0], m[1])):
        # 1. Get month.
        month = month_keys[" ".join(tokens[i:i + n])]
        # 2. Get day from text before month.
        day = phrase_to_number(" ".join(tokens[:i]), number_keys)
        # 3. Get year from text after month.
        year = phrase_to_number(" ".join(tokens[i + n:]), number_keys)
        if day and year and 1 <= day <= 31 and 1 <= month <= 12:
            return f"{year:04d}-{month:02d}-{day:02d}"
    # Year only dates ("عام ألف و ثلاثمائة ...", no day or month): keep the year
    if not matches:
        year = phrase_to_number(result, number_keys)
        if year and year >= 1000:
            return str(year)
    # Preserve the original if day or year could not be parsed.
    return result


# time builder for converter
def time_value(value, time_keys=None):
    """Hour / minute: the whole phrase must be in dates.json 'time', digits are kept. Otherwise unchanged."""
    phrase = normalize_text(value)
    key = spaceless(phrase)
    if not key:
        return phrase
    if key.isdigit():
        return int(key)
    number = (time_keys or get_time_keys()).get(key)
    return phrase if number is None else number


# 2. convert datesn time and seve into numerical values
def convert(data):
    converted = copy.deepcopy(data)
    # 1. Convert dates using dates.json.
    date_fields = [
        "DATENAISSFR",
        "DATENAISSAR",
        "DATENAISSPEREFR",
        "DATENAISSPEREAR",
        "DATENAISSMEREFR",
        "DATENAISSMEREAR",
        "DATEDECLAFR",
        "DATEDECLAAR",
    ]
    for field in date_fields:
        if converted.get(field):
            converted[field] = replace_aliases(converted[field])
    # the declaration year comes from the converted declaration date (or from its own text)
    declaration_date = str(converted.get("DATEDECLAFR", ""))
    if re.match(r"\d{4}-\d{2}-\d{2}$", declaration_date):
        converted["ANNEEDECLA"] = declaration_date[:4]
    elif converted.get("ANNEEDECLA"):
        year = phrase_to_number(converted["ANNEEDECLA"], get_number_keys())
        if year:
            converted["ANNEEDECLA"] = str(year)
    # 2. Hours and minutes: matched as whole phrases with the 'time' section of dates.json.
    time_keys = get_time_keys()
    for field in ("HEURENAISS", "MINNAISS"):
        value = normalize_text(converted.get(field, ""))
        if not value:
            continue
        converted[field] = time_value(value, time_keys)
    # 3. Convert sex to numbers: 1 = male, 0 = female.
    sex = ar_key(converted.get("SEXE", ""))
    if sex == "ذكر":
        converted["SEXE"] = 1
    elif sex == "انثي":
        converted["SEXE"] = 0
    return converted


def main():
    parser = argparse.ArgumentParser(description="Convert the dates, time and sex of a parsed record.")
    parser.add_argument("input_file", type=Path, help="JSON written by parser.py")
    parser.add_argument("-o", "--output", type=Path, help="write the JSON here instead of stdout")
    args = parser.parse_args()
    if not args.input_file.is_file():
        parser.error(f"Input file not found: {args.input_file}")
    write_json(convert(read_json_file(args.input_file)), args.output)


if __name__ == "__main__":
    main()