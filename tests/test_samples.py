"""Known-good samples must go through the whole offline chain without validation errors.

Same idea as the old test (parse > convert > map > translate > validate), for the split pipeline:

    parser > converter > mapper > translator (Groq, skipped offline) > validator

No model and no network is needed: the NER models are not loaded (rules only) and the Groq translation is
replaced by "keep the Arabic", so the test is deterministic and fast, which is what a CI job needs.
Every sample in samples/*.txt is checked, drop a new .txt there to add a case.

    python -m pytest tests/test_samples.py
"""

import json
from pathlib import Path

import pytest

import converter
import mapper
import parser as record_parser
import validator
from common import TEMPLATE, read_text_file

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = sorted((ROOT / "samples").glob("*.txt"))


def run_chain(text):
    """raw text -> record, offline (no NER, no Groq)."""
    return mapper.map_fields(converter.convert(record_parser.parse(text, None)))


def issues_of(report, level):
    return [issue for issue in report["issues"] if issue["level"] == level]


def format_issues(issues):
    return "\n".join(f"  [{issue['level'].upper()}] {issue['field']}: {issue['message']}" for issue in issues)


def test_samples_exist():
    """An empty samples folder would make every test below pass without checking anything."""
    assert SAMPLES, f"no sample found in {ROOT / 'samples'}"


@pytest.mark.parametrize("sample_file", SAMPLES, ids=lambda path: path.name)
def test_parser_validates_samples(sample_file):
    """Known-good samples should parse without validation errors."""
    result = run_chain(read_text_file(sample_file))
    report = validator.validate(result, check_translation=False)

    errors, warnings = issues_of(report, "error"), issues_of(report, "warning")
    assert report["valid"] and not errors, f"{sample_file.name} failed validation:\n{format_issues(errors)}"
    if warnings:
        print(f"\nWarnings for {sample_file.name}:\n{format_issues(warnings)}")


@pytest.mark.parametrize("sample_file", SAMPLES, ids=lambda path: path.name)
def test_samples_have_the_whole_template(sample_file):
    """Every field of the output template is present, with the right type (str or int for time / sex)."""
    result = run_chain(read_text_file(sample_file))
    assert set(result) == set(TEMPLATE), sorted(set(result) ^ set(TEMPLATE))
    for field, value in result.items():
        assert isinstance(value, (str, int)), f"{field}: {type(value).__name__}"


@pytest.mark.parametrize("sample_file", SAMPLES, ids=lambda path: path.name)
def test_samples_are_json_serialisable(sample_file):
    result = run_chain(read_text_file(sample_file))
    assert json.loads(json.dumps(result, ensure_ascii=False)) == result