"""Stage 4 - mapper: JSON object -> JSON object with the French jobs and nationalities (dictionary lookups).

    python src/mapper.py converted.json [-o mapped.json]

A value is looked up exactly (after normalisation), then with typo tolerance (see common.fuzzy_lookup).
A value that is not in the dictionaries is left empty (the validator reports it as a warning).
A profession that is not found stays empty here: the translator stage asks the local model for it.
"""

import argparse
import copy
from pathlib import Path

from common import (
    ar_key,
    fuzzy_lookup,
    get_jobs_lookup,
    get_nationality_lookup,
    jobs_data,
    normalize_text,
    read_json_file,
    write_json,
)


# 3. map translation for jobs and nationalities
def map_fields(data):
    mapped = copy.deepcopy(data)
    # 1. Map Arabic professions to French using jobs.json
    jobs_lookup = get_jobs_lookup()
    for arabic_field, french_field in [
        ("PROFESSIONPERE", "PROFESSIONPEREFR"),
        ("PROFESSIONMERE", "PROFESSIONMEREFR"),
    ]:
        arabic_job = normalize_text(mapped.get(arabic_field, ""))
        if not arabic_job:
            continue
        if arabic_job in jobs_data and isinstance(jobs_data[arabic_job], str):
            mapped[french_field] = jobs_data[arabic_job]
        else:
            # exact after normalisation, or the closest job when the scan has a typo ("ربة البيتت")
            french_job = fuzzy_lookup(ar_key(arabic_job), jobs_lookup)
            if french_job:
                mapped[french_field] = french_job
            # not found: stays empty, translator.py asks the local model for it
    # 2. Map Arabic nationalities to French using nationalities.json
    nationality_fields = [
        ("NATIONALITE", "NATIONALITEFR"),
        ("NATIONALITEPERE", "NATIONALITEPEREFR"),
        ("NATIONALITEMERE", "NATIONALITEMEREFR"),
    ]
    # Arabic-to-French lookup table (the first French variant is the canonical value).
    nationality_map = get_nationality_lookup()
    for arabic_field, french_field in nationality_fields:
        french_nationality = fuzzy_lookup(ar_key(mapped.get(arabic_field, "")), nationality_map)
        if french_nationality:
            mapped[french_field] = french_nationality
    return mapped


def main():
    parser = argparse.ArgumentParser(description="Map the Arabic jobs and nationalities to French.")
    parser.add_argument("input_file", type=Path, help="JSON written by converter.py")
    parser.add_argument("-o", "--output", type=Path, help="write the JSON here instead of stdout")
    args = parser.parse_args()
    if not args.input_file.is_file():
        parser.error(f"Input file not found: {args.input_file}")
    write_json(map_fields(read_json_file(args.input_file)), args.output)


if __name__ == "__main__":
    main()