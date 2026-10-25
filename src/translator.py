"""Stage 5 - translator: JSON object -> JSON object with the French names, birth places, employee and missing jobs.

    python src/translator.py mapped.json [-o translated.json]

One request to the local model (Ollama, qwen2.5:7b by default, the same server as the spelling checker)
translates every name / address / employee of the record, and the professions that the dictionary (mapper.py)
did not find. Nothing is sent online: the Ollama address must be this computer, as for the spelling checker
(MADOC_OLLAMA_URL, MADOC_SPELL_ALLOW_REMOTE). Official place names come from cities.json / places.json and
the child's French family name is sent with the request, so the model cannot choose another spelling.
When Ollama is not running or the model is not pulled, the original Arabic values are kept, the validator then
reports the French fields as untranslated.
MADOC_TRANSLATE_MODEL (environment or .env) chooses another Ollama model (default: MADOC_SPELL_MODEL).
Messages go to stderr, never to stdout.
"""

import argparse
import copy
import json
import os
import re
import sys
import time
import urllib.request
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from common import (
    EMPLOYEE_TITLE,
    ar_key,
    get_places_lookup,
    read_json_file,
    write_json,
)

load_dotenv()

import spelling_checker  # noqa: E402  (after load_dotenv: it reads MADOC_OLLAMA_URL / MADOC_SPELL_MODEL)

# Ollama model of the translation (MADOC_TRANSLATE_MODEL in the environment / .env replaces it)
MODEL = os.environ.get("MADOC_TRANSLATE_MODEL") or spelling_checker.MODEL
# seconds to wait for "is Ollama running": the answer is immediate when it is
CHECK_TIMEOUT = 3
# Arabic script block: a French value must not contain any of it
ARABIC_SCRIPT = re.compile("[\u0600-\u06ff]")


def log(*args):
    """Messages go to stderr: stdout carries the JSON record (python src/pipeline.py x.txt > record.json)."""
    print(*args, file=sys.stderr)


def unavailable_reason():
    """None when the local model can be used, otherwise why not (Ollama not running, model not pulled ...).

    Asked again at every record: Ollama may be started while the web interface is running."""
    try:
        url = spelling_checker.ollama_url("/api/tags")
        with urllib.request.urlopen(url, timeout=CHECK_TIMEOUT) as response:
            models = json.loads(response.read().decode("utf-8")).get("models") or []
    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"Ollama not reachable at {spelling_checker.OLLAMA_URL} ({e})"
    names = {m.get("name") for m in models} | {m.get("model") for m in models}
    if MODEL not in names and f"{MODEL}:latest" not in names:
        return f"the model {MODEL} is not pulled (ollama pull {MODEL})"
    return None


def is_available():
    """True when translate() can really call the local model."""
    return unavailable_reason() is None


@lru_cache(maxsize=None)
def places_max_words():
    """Number of words of the longest known place name (the table has thousands of entries)."""
    return max((len(key.split()) for key in get_places_lookup()), default=0)


def find_known_places(text):
    """Return {arabic name: official French name} for every place of cities.json / places.json found in text."""
    lookup = get_places_lookup()
    if not lookup:
        return {}
    tokens = text.split()
    keys = [ar_key(token) for token in tokens]
    max_len = places_max_words()
    found = {}
    i = 0
    while i < len(tokens):
        for n in range(min(max_len, len(tokens) - i), 0, -1):
            candidate = " ".join(keys[i:i + n])
            if candidate not in lookup and candidate.startswith("ب"):
                # attached preposition: "بالمحمدية" -> "المحمدية"
                candidate = ar_key(candidate[1:])
            if candidate in lookup:
                found[" ".join(tokens[i:i + n])] = lookup[candidate]
                i += n
                break
        else:
            i += 1
    return found


def strip_employee_job(text):
    """Remove the Arabic job words after the employee name (the French title is fixed)."""
    stripped = re.sub(r"\s+(?:ال)?موظف[ةه]?\b.*$", "", text).strip()
    return stripped or text


def postprocess_output(item, output):
    """Enforce the mandatory conventions on the LLM output."""
    text = output.strip()
    source = ar_key(item["text"])
    if item["type"] == "name":
        family = item.get("family_name") or {}
        if family and family["fr"].lower() not in text.lower():
            log(f"Warning: family name '{family['fr']}' missing from the translation of '{item['text']}'.")
        # lineage words are mandatory: بن = Fils de, بنت = Fille de
        if re.match(r"(?:ابن|بن)\s", source + " ") and not re.match(r"fils de\b", text, re.I):
            text = "Fils de " + re.sub(r"^(?:ben|ibn|bin)\s+", "", text, flags=re.I)
        elif re.match(r"(?:ابنه|بنت)\s", source + " ") and not re.match(r"fille de\b", text, re.I):
            text = "Fille de " + re.sub(r"^(?:bent|bint|benti)\s+", "", text, flags=re.I)
    elif item["type"] == "address":
        for arabic, french in item.get("glossary", {}).items():
            if french.lower() not in text.lower():
                log(f"Warning: official name '{french}' ({arabic}) missing from the translation.")
    elif item["type"] == "job":
        # same style as jobs.json: upper case, no punctuation around
        text = re.sub(r"\s+", " ", text.strip(" .,;:-\"'")).upper()
    elif item["type"] == "employee":
        # the job title is always the same fixed sentence
        if EMPLOYEE_TITLE not in text.upper():
            text = f"{text} {EMPLOYEE_TITLE}"
    return text


def answer_results(answer, count):
    """The model's {"results": [...]} -> exactly count strings (None for a missing one), or None.

    The items are numbered: an answer of {"id", "text"} objects is matched by id, a plain list of strings by
    position when the count is right."""
    results = answer.get("results") if isinstance(answer, dict) else None
    if not isinstance(results, list):
        return None
    if results and all(isinstance(r, dict) for r in results):
        by_id = {}
        for r in results:
            try:
                by_id.setdefault(int(r.get("id")), r.get("text"))
            except (TypeError, ValueError):
                continue
        found = [by_id.get(n) for n in range(1, count + 1)]
        return found if any(isinstance(f, str) for f in found) else None
    return results if len(results) == count else None


# local model call (1 request)
def translate_batch(items):
    """Translate/transliterate Moroccan civil registry names, addresses,
    and employee information in one request to the local model."""

    if not items:
        return []

    originals = [item["text"] for item in items]

    reason = unavailable_reason()
    if reason:
        log(f"Translation model unavailable: {reason}; keeping original values.")
        return originals

    prompt = """
    You process Moroccan civil registry data. Convert each input item into
    conventional French as appropriate for its specified type.

    GENERAL RULES:
    - Use Latin characters only. Do not return Arabic script.
    - Preserve all meaningful information, numbers, and name components.
    - Do not invent information or omit content.
    - Follow the input order exactly.
    - Return one result per input item.

    TYPE: name
    - Transliterate Moroccan Arabic personal names into conventional French spellings used in Moroccan civil records.
    - Transliterate personal names; never translate their meanings.
    - Preserve the order of name components.
    - The Arabic lineage term "بن" must be rendered as "Fils de".
    - The Arabic lineage term "بنت" must be rendered as "Fille de".
    - These lineage translations are mandatory.
    - Do not transliterate "بن" as "Ben" or "بنت" as "Bent".
    - Transliterate the remaining personal-name components.
    - Some name items have a "family_name" object {"ar": ..., "fr": ...}. If the item text
      contains that Arabic family name, you MUST write it exactly as the given "fr" value
      (same spelling, capitalised like the other words) and transliterate only the other words.

    TYPE: address
    - Translate Arabic addresses into offical Moroccan town name in French.
    - Moroccan city, town, village, and province names must use their established French/Latin spellings.
    - Never guess or improvise the spelling of a Moroccan place name.
    - Preserve street numbers, neighborhood names, place names, and all administrative divisions.
    - Translate common geographic and administrative terms.
    - Transliterate proper place names; do not translate their meanings.
    - Use established French spellings for Moroccan places when known. Otherwise use a consistent Latin transliteration.
    - Preserve all numbers and address details.
    - Do not invent missing details.
    - Some address items have a "glossary" object (Arabic name -> official French name).
      You MUST use exactly those French spellings for those names, never another spelling.

    TYPE: job
    - The text is a profession written on a Moroccan birth certificate (Arabic).
    - Return the usual French name of the profession in UPPER CASE, short (one to four words), no article.
    - Use the feminine form when the Arabic is feminine (ending in ة), the masculine form otherwise.
    - "ربة البيت" is "FEMME AU FOYER"; no profession ("لا شيء", "بدون مهنة") is "SANS PROFESSION".
    - A job at a known employer keeps the usual abbreviation: O C P (phosphates office), O N E (electricity),
      O N C F (railways). Do not write long explanations or the city name.
    - Do not invent a different job: translate what is written.

    TYPE: employee
    - Convert employee information into French.
    - Transliterate the employee's personal name using conventional Moroccan French spelling.
    - Translate job titles and occupational descriptions into French.
    - Preserve the employee's name and role if both are present.
    - Do not invent a job title or other information.
    - The employee's job position must always be rendered exactly as: "FONCTIONAIRE ET OFFECIER DETAT CIVIL".

    OUTPUT:
    - The input is {"items": [{"id": 1, "type": ..., "text": ...}, ...]}.
    - Return ONLY a valid JSON object in this exact format:
    {"results": [{"id": 1, "text": "result 1"}, {"id": 2, "text": "result 2"}]}
    - One result per input item, with the same "id".
    - Do not include explanations or Markdown.
    """
    payload = {"items": [dict(item, id=n) for n, item in enumerate(items, 1)]}

    outputs = None
    # one retry: a small model sometimes answers a malformed JSON
    for attempt in range(2):
        if attempt:
            time.sleep(1)
        try:
            answer = spelling_checker.ollama_json(prompt, payload, MODEL)
            candidate = answer_results(answer, len(items))
            if candidate is None:
                log("Invalid model response: incorrect results array.")
                log("Raw response:", json.dumps(answer, ensure_ascii=False)[:500])
                continue
            outputs = candidate
            break
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError, ValueError) as e:
            log(f"Could not parse the model response: {e}")
        except Exception as e:
            log(f"Translation model error: {e}")

    if outputs is None:
        return originals

    # validate item by item: one bad answer no longer discards the good ones
    final = []
    for item, original, output in zip(items, originals, outputs):
        if not isinstance(output, str) or not output.strip():
            log("Invalid output; preserving original value.")
            final.append(original)
        elif ARABIC_SCRIPT.search(output):
            log("Arabic script found in output; preserving original value.")
            final.append(original)
        else:
            final.append(postprocess_output(item, output))
    return final


# 4. translate names and addresses to French with the local model
def translate(data):
    """Populate French fields for names, addresses, and employee information."""
    translated = copy.deepcopy(data)
    field_mapping = {
        # Personal names
        "PRENOMPEREAR": ("PRENOMPEREFR", "name"),
        "NOMPEREAR": ("NOMPEREFR", "name"),
        "PRENOMMEREAR": ("PRENOMMEREFR", "name"),
        "NOMMEREAR": ("NOMMEREFR", "name"),
        # Addresses
        "LIEUNAISS": ("LIEUNAISSFR", "address"),
        # Employee name and job title
        "EMPLOYECREANT": ("EMPLOYECREANTFR", "employee"),
        # Professions: only those the dictionary did not find (mapper.py left the French empty)
        "PROFESSIONPERE": ("PROFESSIONPEREFR", "job"),
        "PROFESSIONMERE": ("PROFESSIONMEREFR", "job"),
    }
    active_fields = []
    for source_field, (target_field, item_type) in field_mapping.items():
        value = translated.get(source_field)
        if item_type == "job" and str(translated.get(target_field) or "").strip():
            continue  # found in the dictionary
        if isinstance(value, str) and value.strip():
            text = value.strip()
            # the French job title is fixed, only the employee name is sent
            if item_type == "employee":
                text = strip_employee_job(text)
            # the father usually has the child's family name: it is sent with the request
            family_name = {}
            if source_field == "PRENOMPEREAR" and translated.get("NOMAR") and translated.get("NOM"):
                family_name = {"ar": translated["NOMAR"], "fr": translated["NOM"].title()}
            active_fields.append(
                {
                    "source": source_field,
                    "target": target_field,
                    "type": item_type,
                    "text": text,
                    "family_name": family_name,
                }
            )
    if not active_fields:
        return translated
    # All names, addresses, and employee information are sent in one request.
    items = []
    for field in active_fields:
        item = {"type": field["type"], "text": field["text"]}
        if field["family_name"]:
            item["family_name"] = field["family_name"]
        if field["type"] == "address":
            # official names found in cities.json / places.json are imposed to the model
            glossary = find_known_places(field["text"])
            if glossary:
                item["glossary"] = glossary
        items.append(item)
    outputs = translate_batch(items)
    for field, output in zip(active_fields, outputs):
        # no model / refused answer: the Arabic comes back unchanged, a job must then stay empty (not Arabic)
        if field["type"] == "job" and (output == field["text"] or ARABIC_SCRIPT.search(output)):
            continue
        translated[field["target"]] = output
    return translated


def main():
    parser = argparse.ArgumentParser(description="Translate the names, birth place and employee to French (local Ollama model).")
    parser.add_argument("input_file", type=Path, help="JSON written by mapper.py")
    parser.add_argument("-o", "--output", type=Path, help="write the JSON here instead of stdout")
    args = parser.parse_args()
    if not args.input_file.is_file():
        parser.error(f"Input file not found: {args.input_file}")
    write_json(translate(read_json_file(args.input_file)), args.output)


if __name__ == "__main__":
    main()