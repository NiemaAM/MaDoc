"""Stage 6 - validator: JSON object -> validation report (structure, types, dates, language, parsing leaks).

    python src/validator.py translated.json [-o report.json] [--no-translation]

Report:
    {"valid": true/false, "errors": 2, "warnings": 1,
     "issues": [{"level": "error" | "warning", "field": "...", "message": "..."}]}

"error"   = the record is wrong, "warning" = suspicious, to be reviewed.
--no-translation (check_translation=False) skips the checks that need the translation stage.
Exit code of the command line: 1 when there is at least one error.
"""

import argparse
import re
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from common import EMPLOYEE_TITLE, TEMPLATE, read_json_file, write_json

ARABIC = re.compile(r"[؀-ۿ]")
LATIN = re.compile(r"[A-Za-z]")
FULL_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
YEAR_ONLY = re.compile(r"^\d{4}$")

# fields that must always be filled
REQUIRED = [
    "NUMACTE", "DATENAISSFR", "DATENAISSAR", "PRENOMAR", "PRENOM", "SEXE",
    "NOMAR", "NOM", "LIEUNAISS", "PRENOMPEREAR", "PRENOMMEREAR",
    "DATEDECLAFR", "DATEDECLAAR", "EMPLOYECREANT",
]

# (Gregorian field, Hijri field, label, full date required)
DATE_PAIRS = [
    ("DATENAISSFR", "DATENAISSAR", "child", True),
    ("DATENAISSPEREFR", "DATENAISSPEREAR", "father", False),
    ("DATENAISSMEREFR", "DATENAISSMEREAR", "mother", False),
    ("DATEDECLAFR", "DATEDECLAAR", "declaration", True),
]

# Arabic fields (must contain Arabic letters and no Latin letters)
ARABIC_FIELDS = [
    "PRENOMAR", "NOMAR", "LIEUNAISS", "NATIONALITE", "PRENOMPEREAR", "NOMPEREAR",
    "NATIONALITEPERE", "ADRESSENAISSPERE", "PROFESSIONPERE", "PRENOMMEREAR",
    "NOMMEREAR", "NATIONALITEMERE", "ADRESSENAISSMERE", "PROFESSIONMERE",
    "ADRESSEPARENT", "TEXTEDECLA", "EMPLOYECREANT", "BECNAME",
]
# an address can contain Latin letters (roman numbers "II"): only a warning
LATIN_ALLOWED = {"ADRESSEPARENT", "TEXTEDECLA"}
# French fields (must not contain Arabic letters)
FRENCH_FIELDS = [
    "PRENOM", "NOM", "LIEUNAISSFR", "NATIONALITEFR", "PRENOMPEREFR", "NOMPEREFR",
    "NATIONALITEPEREFR", "PROFESSIONPEREFR", "PRENOMMEREFR", "NOMMEREFR",
    "NATIONALITEMEREFR", "PROFESSIONMEREFR", "EMPLOYECREANTFR", "BECNAMEFR",
]
# (Arabic field, French field, needs the translation stage)
PAIRS = [
    ("NATIONALITE", "NATIONALITEFR", False),
    ("NATIONALITEPERE", "NATIONALITEPEREFR", False),
    ("NATIONALITEMERE", "NATIONALITEMEREFR", False),
    ("PROFESSIONPERE", "PROFESSIONPEREFR", False),
    ("PROFESSIONMERE", "PROFESSIONMEREFR", False),
    ("PRENOMPEREAR", "PRENOMPEREFR", True),
    ("NOMPEREAR", "NOMPEREFR", True),
    ("PRENOMMEREAR", "PRENOMMEREFR", True),
    ("NOMMEREAR", "NOMMEREFR", True),
    ("LIEUNAISS", "LIEUNAISSFR", True),
    ("EMPLOYECREANT", "EMPLOYECREANTFR", True),
]
# label words that must never appear inside a name / place / job / nationality:
# they reveal a regex that captured too much
LEAK_WORDS = [
    "جنسيت", "المولود", "حرفت", "مهنت", "موافق", "موفق", "هجري", "ميلادي",
    "الإسم", "الاسم", "الساكن", "توقيع", "بناء على", "PRENOM", "NOM ",
]
LEAK_FIELDS = [f for f in ARABIC_FIELDS if f not in ("TEXTEDECLA", "BECNAME")]


def parse_date(value, hijri=False):
    """Return ('full', date) / ('year', int) / (None, None).
    A Hijri date is only range checked (month 1-12, day 1-30): it is not a valid Gregorian date."""
    value = str(value)
    match = FULL_DATE.match(value)
    if match:
        year, month, day = int(match[1]), int(match[2]), int(match[3])
        if hijri:
            if 1 <= month <= 12 and 1 <= day <= 30:
                return "full", SimpleNamespace(year=year, month=month, day=day)
            return None, None
        try:
            return "full", date(year, month, day)
        except ValueError:
            return None, None
    if YEAR_ONLY.match(value):
        return "year", int(value)
    return None, None


def hijri_to_gregorian_year(hijri_year):
    """Approximate Gregorian year of a Hijri year."""
    return hijri_year * 0.970224 + 621.5744


def validate(result, check_translation=True):
    """JSON object -> validation report (see the module docstring)."""
    issues = []

    def error(field, message):
        issues.append({"level": "error", "field": field, "message": message})

    def warning(field, message):
        issues.append({"level": "warning", "field": field, "message": message})

    # 1. structure
    for field in sorted(set(TEMPLATE) - set(result)):
        error(field, "missing field")
    for field in sorted(set(result) - set(TEMPLATE)):
        error(field, "unexpected field")

    # 2. required fields
    for field in REQUIRED:
        if field in result and result[field] in ("", None):
            error(field, "required field is empty")

    # 3. types
    for field, value in result.items():
        if field == "SEXE":
            if isinstance(value, bool) or value not in (0, 1):
                error(field, f"must be 0 or 1, got {value!r}")
        elif field in ("HEURENAISS", "MINNAISS"):
            limit = 23 if field == "HEURENAISS" else 59
            if value == "":
                # "و الدقيقة" with nothing after it: the scan has no value (often an exact hour), or it was not read
                warning(field, "not written on the certificate (or not read): check the scan")
            elif not isinstance(value, int) or isinstance(value, bool):
                error(field, f"must be an integer, got {value!r} (conversion failed?)")
            elif not 0 <= value <= limit:
                error(field, f"out of range 0-{limit}: {value}")
        elif not isinstance(value, str):
            error(field, f"must be a string, got {type(value).__name__}")
    if isinstance(result.get("NUMACTE"), str) and result["NUMACTE"] and not re.fullmatch(r"[0-9]+", result["NUMACTE"]):
        error("NUMACTE", f"must be digits, got {result['NUMACTE']!r}")
    if result.get("ANNEEDECLA") and not YEAR_ONLY.match(str(result["ANNEEDECLA"])):
        error("ANNEEDECLA", f"must be a 4 digit year, got {result['ANNEEDECLA']!r}")

    # 4. dates
    parsed = {}
    this_year = date.today().year
    for greg_field, hijri_field, label, full_required in DATE_PAIRS:
        for field, low, high in ((greg_field, 1800, this_year), (hijri_field, 1200, 1500)):
            value = result.get(field, "")
            if value == "":
                continue
            kind, parsed_value = parse_date(value, hijri=field == hijri_field)
            if kind is None or (full_required and kind != "full"):
                error(field, f"{label}: invalid date {value!r}")
                continue
            parsed[field] = (kind, parsed_value)
            year = parsed_value.year if kind == "full" else parsed_value
            if not low <= year <= high:
                error(field, f"{label}: year {year} outside {low}-{high}")
        # Gregorian and Hijri dates should describe the same year (catches mis-parsed numbers). Only a warning:
        # the municipality staff sometimes enter a mismatching pair on purpose.
        if greg_field in parsed and hijri_field in parsed:
            years = []
            for field in (greg_field, hijri_field):
                kind, parsed_value = parsed[field]
                years.append(parsed_value.year if kind == "full" else parsed_value)
            if abs(years[0] - hijri_to_gregorian_year(years[1])) > 1.2:
                warning(greg_field, f"{label}: Gregorian {years[0]} and Hijri {years[1]} do not match")

    def year_of(field):
        if field not in parsed:
            return None
        kind, value = parsed[field]
        return value.year if kind == "full" else value

    birth = parsed.get("DATENAISSFR")
    declaration = parsed.get("DATEDECLAFR")
    if birth and declaration and birth[0] == declaration[0] == "full":
        gap = (declaration[1] - birth[1]).days
        if gap < 0:
            error("DATEDECLAFR", "declaration is before the birth")
        elif gap > 365:
            warning("DATEDECLAFR", f"declared {gap} days after the birth")
    if declaration and result.get("ANNEEDECLA") and str(declaration[1].year) != str(result["ANNEEDECLA"]):
        error("ANNEEDECLA", "does not match DATEDECLAFR")
    child_year = year_of("DATENAISSFR")
    for field, label in (("DATENAISSPEREFR", "father"), ("DATENAISSMEREFR", "mother")):
        parent_year = year_of(field)
        if child_year and parent_year:
            age = child_year - parent_year
            if age < 0:
                error(field, f"{label} born after the child")
            elif age < 14 or age > 80:
                warning(field, f"{label} would be {age} at the birth")

    # 5. language
    for field in ARABIC_FIELDS:
        value = result.get(field)
        if isinstance(value, str) and value:
            if not ARABIC.search(value):
                error(field, f"expected Arabic text, got {value!r}")
            elif LATIN.search(value):
                (warning if field in LATIN_ALLOWED else error)(
                    field, f"Latin letters inside an Arabic field: {value!r}"
                )
    # French fields filled by the translation stage are only checked when that stage ran
    translated_fields = {french for _, french, needs in PAIRS if needs}
    for field in FRENCH_FIELDS:
        if not check_translation and field in translated_fields:
            continue
        value = result.get(field)
        if isinstance(value, str) and value and ARABIC.search(value):
            error(field, f"Arabic script in a French field: {value!r}")
    for field in ("PRENOM", "NOM"):
        value = result.get(field)
        if isinstance(value, str) and value and not LATIN.search(value):
            error(field, f"expected Latin letters, got {value!r}")
    for field in ("NOM", "NATIONALITEFR", "NATIONALITEPEREFR", "NATIONALITEMEREFR"):
        value = result.get(field)
        if isinstance(value, str) and value and value != value.upper():
            warning(field, f"expected upper case, got {value!r}")

    # 6. every Arabic value has its French counterpart
    for arabic_field, french_field, needs_translation in PAIRS:
        if needs_translation and not check_translation:
            continue
        if result.get(arabic_field) and not result.get(french_field):
            if arabic_field.startswith(("NATIONALITE", "PROFESSION")):
                warning(french_field, f"{result[arabic_field]!r} not found in the dictionary (and not translated)")
            else:
                error(french_field, "missing French value")

    # 7. conventions
    if check_translation:
        for arabic_field, french_field in (("NOMPEREAR", "NOMPEREFR"), ("NOMMEREAR", "NOMMEREFR")):
            arabic, french = result.get(arabic_field, ""), result.get(french_field, "")
            if arabic and french:
                first = arabic.split()[0]
                expected = "Fille de" if first in ("بنت", "ابنة") else "Fils de" if first in ("بن", "ابن") else ""
                if expected and not french.startswith(expected):
                    error(french_field, f"must start with {expected!r}, got {french!r}")
        if result.get("EMPLOYECREANTFR") and EMPLOYEE_TITLE not in result["EMPLOYECREANTFR"]:
            error("EMPLOYECREANTFR", f"must contain {EMPLOYEE_TITLE!r}")
    if result.get("BECNAME") != "بالتفويض" or result.get("BECNAMEFR") != "PAR DELEGATION":
        error("BECNAME", "fixed delegation values were modified")

    # 8. parsing leaks: label words or very long values inside short fields
    for field in LEAK_FIELDS:
        value = result.get(field)
        if not isinstance(value, str) or not value:
            continue
        for word in LEAK_WORDS:
            if word in value:
                error(field, f"contains the label word {word!r}: {value!r}")
                break
        if len(value) > 120:
            warning(field, f"{len(value)} characters, the regex may have captured too much")

    errors = sum(1 for i in issues if i["level"] == "error")
    return {
        "valid": errors == 0,
        "errors": errors,
        "warnings": len(issues) - errors,
        "issues": issues,
    }


def format_report(report):
    """Report -> readable text (one line per issue)."""
    lines = [f"[{i['level'].upper()}] {i['field']}: {i['message']}" for i in report["issues"]]
    lines.append(
        f"{'VALID' if report['valid'] else 'INVALID'}: {report['errors']} error(s), {report['warnings']} warning(s)"
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Validate a generated birth record JSON.")
    parser.add_argument("input_file", type=Path, help="JSON written by translator.py")
    parser.add_argument("-o", "--output", type=Path, help="write the report here instead of stdout")
    parser.add_argument("--no-translation", action="store_true", help="skip the checks that need the translation stage")
    args = parser.parse_args()
    if not args.input_file.is_file():
        parser.error(f"Input file not found: {args.input_file}")
    report = validate(read_json_file(args.input_file), check_translation=not args.no_translation)
    write_json(report, args.output)
    print(format_report(report), file=sys.stderr)
    sys.exit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()