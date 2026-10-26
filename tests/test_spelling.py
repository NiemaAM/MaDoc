"""Spelling checker: only spelling fixes towards known words are kept, names / numbers are never touched.

The model is replaced by a fake Ollama server on this computer (same HTTP API), so no model is needed.

    python -m pytest tests/test_spelling.py
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

import spelling_checker as sc
from common import read_text_file
from documents import (COHERENCE_DOCUMENT, DATE_DOCUMENT, DECLARANT_DOCUMENT, DECLARANT_TWO_LINES,
                       EMPTY_MINUTE_DOCUMENT, FATHER_DECLARANT_DOCUMENT, NAME_DOCUMENT, sample)

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------------------------------------
# guard
# ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("old,new", [
    ("ميلاي ية", "ميلادية"),        # a word cut in two
    ("البيتت", "البيت"),            # a doubled letter
    ("موفق", "موافق"),              # a missing letter
    ("جنسيتة", "جنسيته"),           # wrong final letter
    ("بجماعه", "بجماعة"),           # glued preposition
])
def test_spelling_fixes_are_accepted(old, new):
    accepted, reason = sc.accept_change(old.split(), new.split())
    assert accepted, reason


@pytest.mark.parametrize("old,new,reason", [
    ("بهلولي", "بهلول", "not a known word"),                 # a name is never changed
    ("مدنب", "مدينة", ""),                                    # a name is not turned into a form word
    ("هشام", "هاشم", "already correct" if sc.is_known("هشام") else "not a known word"),
    ("الحضرية", "القروية", "already correct"),                # a correct word is never replaced
    ("159", "195", "digits"),
    ("SARA", "SARAH", "Latin"),
    ("ثالث", "ثالث و", "already correct"),
    ("موفق", "", "added or removed"),
])
def test_other_changes_are_refused(old, new, reason):
    accepted, why = sc.accept_change(old.split(), new.split())
    assert not accepted
    assert reason in why


def test_merge_line_keeps_only_the_accepted_changes():
    notes = []
    line = sc.merge_line(
        "موفق ثامن عشر دجنبر سنة الفين و ثلاثة ميلاي ية",
        "موافق ثامن عشر دجنبر سنة الفين و أربعة ميلادية",     # the model also changed the year: refused
        4, notes,
    )
    assert line == "موافق ثامن عشر دجنبر سنة الفين و ثلاثة ميلادية"
    assert any("'موفق' -> 'موافق'" in n for n in notes)
    assert any("refused 'ثلاثة' -> 'أربعة'" in n for n in notes)


# ---------------------------------------------------------------------------------------------------------
# with a fake Ollama server
# ---------------------------------------------------------------------------------------------------------
class FakeOllama(BaseHTTPRequestHandler):
    answer = None          # function: list of lines -> list of lines (or a raw string)
    names_answer = None    # function: list of names -> list of {"arabic", "latin"} (default: unchanged)
    with_ids = True        # answer with numbered lines (False: plain strings)
    requests = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
        FakeOllama.requests.append(body)
        payload = json.loads(body["messages"][1]["content"])
        if "names" in payload:   # the names check (child's Arabic / Latin names)
            answer = FakeOllama.names_answer or (lambda names: [{"arabic_of_latin": n["arabic"],
                                                                "latin_of_arabic": n["latin"]} for n in names])
            content = json.dumps({"names": answer(payload["names"])}, ensure_ascii=False)
        else:
            # the lines are numbered: {"id", "text"}; the answer keeps the ids when it has one line per input
            entries = payload["lines"]
            texts = [e["text"] if isinstance(e, dict) else e for e in entries]
            content = FakeOllama.answer(texts)
            if not isinstance(content, str):
                if FakeOllama.with_ids and len(content) == len(entries):
                    content = [{"id": e["id"], "text": t} for e, t in zip(entries, content)]
                content = json.dumps({"lines": content}, ensure_ascii=False)
        data = json.dumps({"message": {"role": "assistant", "content": content}}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def ollama(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), FakeOllama)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    monkeypatch.setattr(sc, "OLLAMA_URL", f"http://127.0.0.1:{server.server_port}")
    FakeOllama.requests = []
    FakeOllama.names_answer = None
    FakeOllama.with_ids = True
    yield FakeOllama
    server.shutdown()


def damaged_sample():
    """An anonymized sample with typical HTR mistakes, and what an over-eager model would 'fix'."""
    text = read_text_file(ROOT / "samples" / "1.txt")
    damaged = (text.replace("ميلادية", "ميلاي ية", 1).replace("موافق", "موفق", 1)
               .replace("جنسيته", "جنسيتة", 1).replace("ربة بيت", "ربة البيتت", 1))
    return text, damaged


def test_correct_text_with_a_local_model(ollama):
    original, damaged = damaged_sample()

    def model(lines):
        fixed = []
        for line in lines:
            line = (line.replace("ميلاي ية", "ميلادية").replace("موفق", "موافق").replace("جنسيتة", "جنسيته")
                    .replace("البيتت", "البيت"))
            # an over-eager model also "corrects" a name and a number: both must be refused
            line = line.replace("الإدريسي", "الادريس").replace("159", "195")
            fixed.append(line)
        return fixed

    ollama.answer = model
    text, notes = sc.correct_text(damaged)
    assert text.count("\n") == damaged.count("\n")                  # same lines
    assert "ميلاي ية" not in text and "موفق" not in text and "جنسيتة" not in text and "البيتت" not in text
    assert "الإدريسي" in text and "159" in text                      # name and number untouched
    assert any("refused" in n and "الإدريسي" in n for n in notes)
    assert any("refused" in n and "159" in n for n in notes)
    request = ollama.requests[0]
    assert request["stream"] is False and request["format"] == "json" and request["options"]["temperature"] == 0


def test_one_bad_line_does_not_cancel_the_others(ollama):
    """The model forgets the last line: the corrections of the other lines are still used."""
    _, damaged = damaged_sample()
    ollama.answer = lambda lines: [l.replace("موفق", "موافق") for l in lines[:-1]]
    text, notes = sc.correct_text(damaged)
    assert "موفق" not in text and text.count("\n") == damaged.count("\n")
    ollama.answer = lambda lines: "not json"
    sc._line_answers.clear()
    text, notes = sc.correct_text(damaged)
    # the model's answer is unusable, the rules alone still fix the words of the form
    assert "unusable answer" in notes[0] and "موفق" not in text
    assert any("a word of the form" in n for n in notes)


def test_a_line_split_by_the_model_is_joined_again(ollama):
    """Qwen returned 33 lines for 32 (a name glued to 'جنسيته' put on its own line): matched by id or by
    similarity, the split line is joined again and every other correction is kept."""
    lines = ["رسم ولادة رقم 63", "من والدها محمد بن العربجننسيته مغربية.", "موفق ثالث يناير"]
    by_id = [{"id": 1, "text": lines[0]}, {"id": 2, "text": "من والدها محمد بن العربي"},
             {"id": 2, "text": "جنسيته مغربية."}, {"id": 3, "text": "موافق ثالث يناير"}]
    assert sc.align_answer(lines, by_id) == [lines[0], "من والدها محمد بن العربي جنسيته مغربية.", "موافق ثالث يناير"]
    plain = [lines[0], "من والدها محمد بن العربي", "جنسيته مغربية.", "موافق ثالث يناير"]
    assert sc.align_answer(lines, plain) == sc.align_answer(lines, by_id)
    assert sc.align_answer(lines, [{"id": 3, "text": "x"}])[:2] == lines[:2]        # forgotten lines: as read
    ollama.with_ids = False
    ollama.answer = lambda texts: [t.replace("موفق", "موافق") for t in texts[:1]] + \
        ["من والدها محمد بن العربي", "جنسيته مغربية."] + [texts[2].replace("موفق", "موافق")]
    changes, notes = sc.propose("\n".join(lines) + "\n")
    assert not notes and any(c["old"] == ["موفق"] and c["accepted"] for c in changes)


def test_a_glued_label_is_split_again():
    lines = ["من والده نبيل الإدريسي بن عادلجننسيته مغربية."]
    [change] = sc.glued_labels(lines, [])
    assert (change["old"], change["new"], change["accepted"]) == (["عادلجننسيته"], ["عادل", "جنسيته"], True)
    assert sc.glued_labels(["جنسيتها مغربية", "المولودة بخريبڭة"], []) == []   # labels alone: nothing


def test_a_misread_date_word_is_chosen_by_the_calendar(monkeypatch):
    """'ثالن ربيع الأول 1438' is 'ثالث' (3rd) or 'ثامن' (8th): the Gregorian date of the declaration decides
    (3 Rabi' al-Awwal 1438 = 2 / 3 December 2016); a date in between decides nothing."""
    monkeypatch.setattr(sc, "OLLAMA_URL", "http://127.0.0.1:9")            # no model: the dictionary alone
    base = (read_text_file(ROOT / "samples" / "1.txt")
            .replace("وحرر في فاتح ربيع الأول", "وحرر في ثالن ربيع الأول"))

    def chosen(gregorian_day):
        text = base.replace("موافق ثالث يناير سنة ألفين و سبعة عشرة ميلادية.",
                            f"موافق {gregorian_day} دجنبر سنة ألفين و ستة عشرة ميلادية.")
        [change] = [c for c in sc.propose(text)[0] if c["old"] == ["ثالن"]]
        return change["new"][0] if change["accepted"] else None

    assert chosen("ثاني") == "ثالث"
    assert chosen("سابع") == "ثامن"
    assert chosen("خامس") is None


def test_hijri_to_gregorian():
    import datetime
    assert sc.hijri_to_gregorian(1428, 12, 23) == datetime.date(2008, 1, 2)
    assert abs((sc.hijri_to_gregorian(1438, 1, 1) - datetime.date(2016, 10, 2)).days) <= 1


def test_correct_text_without_ollama(monkeypatch):
    monkeypatch.setattr(sc, "OLLAMA_URL", "http://127.0.0.1:9")       # nothing listens there
    text, notes = sc.correct_text("رسم ولادة رقم 1\n")
    assert text == "رسم ولادة رقم 1\n" and "not reachable" in notes[0]


def test_remote_address_is_refused(monkeypatch):
    """The documents must not leave the computer: a non-local Ollama address is refused."""
    monkeypatch.setattr(sc, "OLLAMA_URL", "https://api.example.com")
    monkeypatch.delenv("MADOC_SPELL_ALLOW_REMOTE", raising=False)
    text, notes = sc.correct_text("رسم ولادة رقم 1\n")
    assert text == "رسم ولادة رقم 1\n" and "not this computer" in notes[0]


def test_spellcheck_does_not_change_a_clean_record(ollama):
    """On a correctly read text, a model that changes nothing leaves the parsed record identical."""
    import converter
    import mapper
    import parser as record_parser
    original, _ = damaged_sample()
    ollama.answer = lambda lines: lines
    text, notes = sc.correct_text(original)
    assert text == original and notes == ["spelling check: no correction"]
    record = mapper.map_fields(converter.convert(record_parser.parse(text, None)))
    assert record == mapper.map_fields(converter.convert(record_parser.parse(original, None)))


def test_spellcheck_repairs_the_record(ollama):
    """The damaged sample parses like the clean one once the spelling is corrected."""
    import converter
    import mapper
    import parser as record_parser
    original, damaged = damaged_sample()
    ollama.answer = lambda lines: [l.replace("ميلاي ية", "ميلادية").replace("موفق", "موافق")
                                   .replace("جنسيتة", "جنسيته").replace("البيتت", "البيت") for l in lines]
    text, _ = sc.correct_text(damaged)

    def chain(t):
        return mapper.map_fields(converter.convert(record_parser.parse(t, None)))

    clean = chain(original)
    fixed = chain(text)
    differing = {k for k in clean if clean[k] != fixed[k]}
    # "ربة البيتت" is corrected to "ربة البيت" (the dictionary form), the original had "ربة بيت"
    assert differing <= {"PROFESSIONMERE", "TEXTEDECLA"}, {k: (clean[k], fixed[k]) for k in differing}


# ---------------------------------------------------------------------------------------------------------
# values of the form (names, places, addresses, jobs) are never changed
# ---------------------------------------------------------------------------------------------------------
def test_values_are_protected():
    tokens = "الإسم الشخصي : وليد".split()
    assert sc.protected_positions(tokens)[0] == {3: "value"}
    tokens = "الساكنان بحي الأمل زنقة رام الله الرقم 171".split()
    assert set(sc.protected_positions(tokens)[0]) == {2, 4, 5}    # الأمل, رام الله
    tokens = "حرفته : حداد".split()
    assert sc.protected_positions(tokens)[0] == {2: "job"}        # a job: checked against jobs.json


def test_protection_crosses_lines_and_misspelt_labels():
    """'من والد' + 'ها ياسين السباعي' on two lines, a misspelt label: the name is still protected."""
    protected, inside = sc.protected_positions("من والد".split())
    assert inside
    protected, _ = sc.protected_positions("ها ياسين السباعي بن رضا".split(), inside=inside)
    assert {0, 1, 2, 4} <= set(protected)
    assert set(sc.protected_positions("الاسم الشحصي وليد".split())[0]) == {2}


def test_short_fragments_are_not_corrected():
    accepted, reason = sc.accept_change(["ها"], ["لها"])
    assert not accepted and "too short" in reason


def nearest_word_model(lines):
    """An over-eager model: every unknown word becomes the closest known word (what we must never trust)."""
    import difflib
    vocabulary = sorted(sc.get_vocabulary())

    def nearest(token):
        if sc.is_known(token) or not sc.ARABIC_WORD.search(token):
            return token
        match = difflib.get_close_matches(sc.surface(token), vocabulary, n=1, cutoff=0.6)
        return match[0] if match else token

    return [" ".join(nearest(t) for t in line.split()) for line in lines]


@pytest.mark.parametrize("sample", sorted((ROOT / "samples").glob("*.txt")), ids=lambda p: p.name)
def test_an_over_eager_model_never_changes_names(ollama, sample):
    import converter
    import mapper
    import parser as record_parser
    import validator
    ollama.answer = nearest_word_model
    text = read_text_file(sample)
    fixed, notes = sc.correct_text(text)

    def chain(t):
        return mapper.map_fields(converter.convert(record_parser.parse(t, None)))

    before, after = chain(text), chain(fixed)
    for field in ("PRENOMAR", "NOMAR", "PRENOMPEREAR", "NOMPEREAR", "PRENOMMEREAR", "NOMMEREAR",
                  "EMPLOYECREANT", "NUMACTE", "LIEUNAISS", "ADRESSEPARENT", "PROFESSIONPERE"):
        assert before[field] == after[field], (field, notes)
    errors = lambda r: validator.validate(r, check_translation=False)["errors"]
    assert errors(after) <= errors(before)


# ---------------------------------------------------------------------------------------------------------
# proposals as data: what the Spelling tab of the app shows and applies
# ---------------------------------------------------------------------------------------------------------
def test_propose_and_apply_a_choice(ollama):
    _, damaged = damaged_sample()
    ollama.answer = lambda lines: [l.replace("موفق", "موافق").replace("الإدريسي", "الادريس") for l in lines]
    changes, skipped = sc.propose(damaged)
    assert not skipped
    names = [i for i, c in enumerate(changes) if c["old"] == ["الإدريسي"]]
    fixes = [i for i, c in enumerate(changes) if c["old"] == ["موفق"]]
    assert names and fixes and not any(changes[i]["accepted"] for i in names)
    assert all(changes[i]["accepted"] for i in fixes)
    # the decisions of the rules
    assert "موفق" not in sc.apply_changes(damaged, changes) and "الإدريسي" in sc.apply_changes(damaged, changes)
    # a person accepts one name change and refuses every spelling fix
    chosen = sc.apply_changes(damaged, changes, {names[0]})
    assert chosen.count("الادريس") == 1 and "موفق" in chosen
    assert chosen.count("\n") == damaged.count("\n")
    # nothing chosen: the text comes back exactly
    assert sc.apply_changes(damaged, changes, set()) == damaged


def test_app_spelling_tab(ollama):
    pytest.importorskip("gradio")
    import app
    _, damaged = damaged_sample()
    ollama.answer = lambda lines: [l.replace("موفق", "موافق").replace("الإدريسي", "الادريس") for l in lines]
    changes, _ = sc.propose(damaged)
    state = {"text": damaged, "changes": changes, "accepted": [i for i, c in enumerate(changes) if c["accepted"]],
             "segments": []}
    view, details, state = app.spelling_review(state)
    # gradio keeps the pieces as {"token", "class_or_confidence"}
    pieces = [(p["token"], p["class_or_confidence"]) if isinstance(p, dict) else tuple(p) for p in view.value]
    labels = [label for _, label in pieces]
    assert len(state["segments"]) == len(pieces)
    # every highlighted piece points at its change, the text pieces at nothing
    for (text, label), index in zip(pieces, state["segments"]):
        assert (label is None) == (index is None)
        if index is not None:
            assert sc.surface(text).strip(sc.PUNCTUATION) == " ".join(changes[index]["old"])
    assert "changes applied" in details
    # click on a refused name change: it becomes accepted
    name_piece = next(k for k, i in enumerate(state["segments"]) if i is not None and changes[i]["old"] == ["الإدريسي"])

    class Click:
        index = name_piece
    _, _, state = app.toggle_change(state, Click())
    assert state["segments"][name_piece] in state["accepted"]
    notes = app.spelling_notes(state)
    assert "accepted in the spelling review" in notes
    _, _, state = app.set_all(state, False)
    assert state["accepted"] == []
    _, _, state = app.set_all(state, True)        # back to the rules: the refused name change is not accepted
    assert state["accepted"] == [i for i, c in enumerate(changes) if c["accepted"]]
    assert labels  # something was highlighted


# ---------------------------------------------------------------------------------------------------------
# what is not a correction, and changes judged word by word
# ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("original,proposed", [
    ("في يوم : رابع ذو القعدة", "في يوم: رابع ذو القعدة"),          # punctuation / spaces
    ("جنسيته:مغربية.", "جنسيته: مغربية."),
    ("الرقم : 564 خريبڭة.", "الرقم: 564 خريبكة."),                    # ڭ = ك for the parser
    ("دائرة قليحة اقليم خريبڭة.", "دائرة قليحة إقليم خريبكة."),       # hamza on alif
    ("المولود ب جماعة بن شكر", "المولود بجماعة بن شكر"),              # spaces after a prefix
])
def test_changes_the_parser_cannot_see_are_ignored(original, proposed):
    assert sc.line_changes(original, proposed, 0) == []


def test_only_the_wrong_word_of_a_time_is_changed():
    """'التانس عشرة' (19): the model fixes 'التانس' but drops 'عشرة' -> only 'التانس' is changed."""
    original = "على الساعة: التانس عشرة"
    changes = sc.line_changes(original, "على الساعة: التاسعة", 0)
    fix = next(c for c in changes if c["new"])
    removed = next(c for c in changes if not c["new"])
    assert (fix["old"], fix["new"], fix["accepted"]) == (["التانس"], ["التاسعة"], True)
    assert (removed["old"], removed["accepted"]) == (["عشرة"], False)
    corrected = sc.apply_changes(original, changes)
    assert corrected == "على الساعة: التاسعة عشرة"
    assert sc.time_values(corrected) == [19]


def test_a_time_fix_needs_a_known_time_phrase():
    """3 letters may differ in an hour only when the phrase becomes one the dictionary knows."""
    changes = sc.line_changes("على الساعة: التانس عشرة", "على الساعة: التاسعة عشرة", 0)
    assert [(c["old"], c["accepted"]) for c in changes] == [(["التانس"], True)]
    changes = sc.line_changes("في التانس عشرة", "في التاسعة عشرة", 0)       # not a time: refused
    assert [(c["old"], c["accepted"]) for c in changes] == [(["التانس"], False)]


@pytest.mark.parametrize("original,proposed", [
    ("موافق ثامن و عشري دجنبر", "موافق ثامن و عشرين دجنبر"),       # "عشري" is a correct form
    ("في ثاني و عشري جمادى", "في ثاني و عشر جمادى"),
    ("حرفته صباخ", "حرفتها صباخ"),                                    # a correct word, another gender
])
def test_correct_words_are_never_replaced(original, proposed):
    changes = sc.line_changes(original, proposed, 0)
    assert changes and not any(c["accepted"] for c in changes)
    assert sc.apply_changes(original, changes) == original


def test_punctuation_is_kept_around_a_fix():
    assert sc.apply_changes("حرفتها: ربة البيتت.", sc.line_changes("حرفتها: ربة البيتت.", "حرفتها: ربة البيت.", 0)) \
        == "حرفتها: ربة البيت."


def test_spelling_tab_shows_the_line_exactly(ollama):
    """Untouched text keeps its punctuation in the Spelling tab ("في يوم : رابع")."""
    pytest.importorskip("gradio")
    import app
    text = "في يوم : رابع ذو القعدة\nعلى الساعة: التانس عشرة\n"
    ollama.answer = lambda lines: [l.replace("التانس", "التاسعة") for l in lines]
    changes, _ = sc.propose(text)
    state = {"text": text, "changes": changes, "accepted": [i for i, c in enumerate(changes) if c["accepted"]],
             "segments": []}
    view, _, _ = app.spelling_review(state)
    shown = "".join(p["token"] if isinstance(p, dict) else p[0] for p in view.value)
    assert shown.rstrip("\n") == text.rstrip("\n")


# ---------------------------------------------------------------------------------------------------------
# jobs: corrected towards jobs.json, every possible job proposed
# ---------------------------------------------------------------------------------------------------------
def test_a_misread_job_gets_the_jobs_of_the_dictionary():
    """'صباخ' is one letter from 'صباغ' (painter) and 'طباخ' (cook): both proposed, nothing applied."""
    count, jobs = sc.job_candidates(["صباخ"])
    assert count == 1 and jobs == ["صباغ", "طباخ"]
    assert sc.job_candidates(["أجير"]) == (0, [])          # written right (hamza does not count)
    assert sc.job_candidates(["ربة", "البيتت"])[1] == ["ربة البيت"]


def test_job_suggestions_without_and_with_the_model(ollama):
    text = "و والدتها نادية العمراني بنت رضا.\nحرفته صباخ\nجنسيتها : مغربية\n"
    ollama.answer = lambda lines: lines                      # the model proposes nothing
    changes, _ = sc.propose(text)
    job = next(c for c in changes if c.get("source") == "jobs.json")
    assert job["old"] == ["صباخ"] and job["options"] == [["صباغ"], ["طباخ"]] and not job["accepted"]
    ollama.answer = lambda lines: [l.replace("صباخ", "طباخ") for l in lines]   # the model chooses: first
    sc._line_answers.clear()                                 # another model: its answers are not the first one's
    changes, _ = sc.propose(text)
    job = next(c for c in changes if c.get("source") == "jobs.json")
    assert job["options"][0] == ["طباخ"] and len([c for c in changes if c["old"] == ["صباخ"]]) == 1
    assert sc.apply_changes(text, changes) == text           # several jobs: nothing applied by itself


def test_a_single_possible_job_is_applied(ollama):
    text = "و والدتها نادية العمراني بنت رضا.\nحرفته : فلاحح\n"
    ollama.answer = lambda lines: lines
    changes, _ = sc.propose(text)
    assert [(c["old"], c["new"], c["accepted"]) for c in changes] == [(["فلاحح"], ["فلاح"], True)]
    assert sc.apply_changes(text, changes) == "و والدتها نادية العمراني بنت رضا.\nحرفته : فلاح\n"


def test_jobs_without_ollama(monkeypatch):
    """Ollama not running: the jobs are still checked against jobs.json."""
    monkeypatch.setattr(sc, "OLLAMA_URL", "http://127.0.0.1:9")
    changes, notes = sc.propose("حرفته صباخ\n")
    assert "not reachable" in notes[0] and changes[0]["options"] == [["صباغ"], ["طباخ"]]


def test_app_cycles_through_the_jobs(ollama):
    pytest.importorskip("gradio")
    import app
    text = "حرفته صباخ\n"
    ollama.answer = lambda lines: lines
    changes, _ = sc.propose(text)
    state = {**app.NO_SPELLING, "text": text, "changes": changes, "accepted": []}
    view, _, state = app.spelling_review(state)
    piece = next(k for k, i in enumerate(state["segments"]) if i is not None)

    class Click:
        index = piece
    seen = []
    for _ in range(3):
        view, _, state = app.toggle_change(state, Click())
        accepted = bool(state["accepted"])
        seen.append(" ".join(app.resolved(state)[0]["new"]) if accepted else None)
    assert seen == ["صباغ", "طباخ", None]                    # first job, second job, refused
    view, _, state = app.toggle_change(state, Click())
    view, _, state = app.toggle_change(state, Click())       # choose the cook
    assert app.apply_changes(text, app.resolved(state), set(state["accepted"])) == "حرفته طباخ\n"


# ---------------------------------------------------------------------------------------------------------
# context: the declarant's job is the parent's job, written a second time
# ---------------------------------------------------------------------------------------------------------
def job_document(father_job, declarant_job, relation="والدها"):
    """The parents' lines of samples/2.txt, with the father's job written twice (header, declaration)."""
    return (
        "من والدها: ياسين السباعي بن رضا\n"
        "و والدتها زينب الحجوجي بنت نبيل.\n"
        f"حرفته {father_job}\n"
        "جنسيتها : مغربية\n"
        "حرفتها : ربة البيت\n"
        f"بناءا على ما صرح به السيد ياسين السباعي ({relation}) تحت عدد : 24 عمره تسعة و أربعون سنة\n"
        f"حرفته {declarant_job} جنسيته مغربية الساكن بحي الأمل\n"
    )


def jobs_after(text, model=lambda lines: lines, ollama=None):
    ollama.answer = model
    changes, _ = sc.propose(text)
    fixed = sc.apply_changes(text, changes)
    return [line.split()[1] for line in fixed.splitlines() if line.split()[0] == "حرفته"], changes


def test_two_misread_mentions_decide_together(ollama):
    """'صباخ' (painter or cook) + 'جاغ': only 'صباغ' fits both -> applied to both."""
    jobs, _ = jobs_after(job_document("صباخ", "جاغ"), ollama=ollama)
    assert jobs == ["صباغ", "صباغ"]


def test_the_context_overrules_a_model_guess(ollama):
    """The model guesses 'جزار' for 'جاغ' and 'طباخ' for 'صباخ': both refused, the context gives 'صباغ'."""
    model = lambda lines: [l.replace("جاغ", "جزار").replace("صباخ", "طباخ") for l in lines]
    jobs, _ = jobs_after(job_document("صباخ", "جاغ"), model, ollama)
    assert jobs == ["صباغ", "صباغ"]


def test_one_mention_right_fixes_the_other(ollama):
    jobs, changes = jobs_after(job_document("صباغ", "جاغ"), ollama=ollama)
    assert jobs == ["صباغ", "صباغ"]
    assert "same job as line 3" in sc.describe(next(c for c in changes if c["old"] == ["جاغ"]))


def test_mentions_that_agree_are_left_alone(ollama):
    """Both read 'حداد' (not in jobs.json): the reading is right, nothing is proposed."""
    _, changes = jobs_after(job_document("حداد", "حداد"), ollama=ollama)
    assert not [c for c in changes if c.get("source") == "jobs.json"]


def test_no_context_when_the_declarant_is_not_the_parent(ollama):
    """'(جده)': the declarant's job is someone else's, each mention is judged alone."""
    jobs, changes = jobs_after(job_document("صباخ", "جاغ", relation="جده"), ollama=ollama)
    assert jobs == ["صباخ", "جاغ"]                                   # nothing applied
    job = next(c for c in changes if c["old"] == ["صباخ"])
    assert job["options"] == [["صباغ"], ["طباخ"]]


def test_a_far_model_guess_is_not_a_job_proposal(ollama):
    """Alone, 'جاغ' -> 'جزار' (2 letters on 3) is a guess: not proposed."""
    text = "حرفته جاغ جنسيته مغربية\n"
    ollama.answer = lambda lines: [l.replace("جاغ", "جزار") for l in lines]
    changes, _ = sc.propose(text)
    assert not any(c["accepted"] for c in changes)
    assert sc.apply_changes(text, changes) == text

# ---------------------------------------------------------------------------------------------------------
# numbers judged by the whole record; no bias towards an example of the instructions
# ---------------------------------------------------------------------------------------------------------


def test_a_misread_year_is_corrected_by_the_context(ollama):
    """'ألفين و ثمانين' (2080, impossible, Hijri 1428 / 1429): 'ثمانية' makes both dates right."""
    ollama.answer = lambda lines: [l.replace("ألفين و ثمانين", "ألفين و ثمانية") for l in lines]
    changes, _ = sc.propose(DATE_DOCUMENT)
    fixes = [c for c in changes if c["old"] == ["ثمانين"]]
    assert len(fixes) == 2 and all(c["accepted"] for c in fixes)
    assert "Hijri" in sc.describe(fixes[0])
    assert sc.record_problems(sc.apply_changes(DATE_DOCUMENT, changes))[0] < sc.record_problems(DATE_DOCUMENT)[0]


def test_a_number_change_that_fixes_nothing_stays_refused(ollama):
    """'عشري' -> 'عشرين' changes no error: a correct number word is kept."""
    ollama.answer = lambda lines: [l.replace("سادس و عشري", "سادس و عشرين") for l in lines]
    changes, notes = sc.propose(DATE_DOCUMENT)
    # refused, and not even shown: a correct word changed for nothing is noise, it goes to the notes
    assert any("ignored" in n and "عشري" in n for n in notes)
    # only the year the Hijri dates give (2080 -> 2008, same day and month), in orange: never applied
    assert changes and all(c["check"] and not c["accepted"] and c["new"] == ["ألفين", "و", "ثمانية"]
                           for c in changes)


def test_a_number_change_that_breaks_a_date_is_refused(ollama):
    ollama.answer = lambda lines: [l.replace("ثمانية و عشرين هجرية", "ثمانية و ثلاثين هجرية") for l in lines]
    changes, _ = sc.propose(DATE_DOCUMENT)
    assert not any(c["accepted"] for c in changes)


def test_a_right_job_gets_no_proposal(ollama):
    """'تقني' is a job of jobs.json: the model's 'صباغ' is not even shown."""
    ollama.answer = lambda lines: [l.replace("تقني", "صباغ") for l in lines]
    changes, _ = sc.propose(DATE_DOCUMENT)
    assert not [c for c in changes if c["old"] == ["تقني"]]


def test_the_instructions_have_no_job_example():
    """An example in the instructions is copied by a small model ('صباغ' proposed for any job)."""
    assert "صباخ" not in sc.PROMPT and "صباغ" not in sc.PROMPT



# ---------------------------------------------------------------------------------------------------------
# the child's names: the Arabic and the Latin reading are compared
# ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("arabic,latin", [
    ("بلال", "BILAL"), ("قاسم", "KACIM"), ("الإدريسي", "IDRISSI"), ("السباعي", "SBAI"), ("الياس", "ILYASSE"),
    ("عبد الرحيم", "ABDERRAHIM"), ("عبد الكبير", "ABDELKEBIR"), ("يوسف", "YOUSSEF"), ("خديجة", "KHADIJA"),
    ("غزلان", "GHIZLANE"), ("ڭمير", "GAMIR"), ("زكرياء", "ZAKARIYA"), ("سعاد", "SOUAD"), ("الودغيري", "OUDGHIRI"),
])
def test_names_that_agree(arabic, latin):
    assert sc.names_agree(arabic, latin)


@pytest.mark.parametrize("arabic,latin", [("ملال", "BILAL"), ("سارة", "HIND"), ("قبلي", "KACIM"),
                                          ("بلال", "BIRAL"), ("بلال", "BITAL")])
def test_names_that_disagree(arabic, latin):
    assert not sc.names_agree(arabic, latin)


def test_every_sample_name_agrees():
    """No false alarm on the samples: every child name pair of samples/ agrees."""
    for sample in sorted((ROOT / "samples").glob("*.txt")):
        lines = read_text_file(sample).split("\n")
        names = sc.child_names(lines)
        assert set(names) == {"first name", "family name"}, sample.name
        for sides in names.values():
            arabic = " ".join(sc.span_words(lines, sides["arabic"]))
            latin = " ".join(sc.span_words(lines, sides["latin"]))
            assert sc.names_agree(arabic, latin), (sample.name, arabic, latin)


@pytest.fixture
def no_names_json(monkeypatch):
    """The name tests below check the model's path: names.json (which knows أيوب) is left out."""
    monkeypatch.setattr(sc, "get_names", lambda: {})




def test_a_misread_arabic_name_gets_a_proposal(ollama, no_names_json):
    """'أيرب' / 'AYOUB': the Arabic of AYOUB is 'أيوب' (1 letter), the Latin of أيرب 'AYRAB' (2 letters):
    the Arabic gets the proposal, not applied (names are decided by a person)."""
    ollama.answer = lambda lines: lines
    ollama.names_answer = lambda names: [{"arabic_of_latin": "أيوب", "latin_of_arabic": "AYRAB"}]
    changes, _ = sc.propose(NAME_DOCUMENT)
    [change] = [c for c in changes if c.get("source") == "names"]
    assert (change["old"], change["new"], change["accepted"]) == (["أيرب"], ["أيوب"], False)
    assert "AYOUB" in sc.describe(change)
    assert sc.apply_changes(NAME_DOCUMENT, changes, {changes.index(change)}).count("الإسم الشخصي : أيوب") == 1
    assert sc.apply_changes(NAME_DOCUMENT, changes) == NAME_DOCUMENT      # nothing applied by default


def test_a_misread_latin_name_gets_a_proposal(ollama):
    text = NAME_DOCUMENT.replace("أيرب", "أيوب").replace("AYOUB", "AYOUR")
    ollama.answer = lambda lines: lines
    ollama.names_answer = lambda names: [{"arabic_of_latin": "أيور", "latin_of_arabic": "AYOUB"}]
    changes, _ = sc.propose(text)
    [change] = [c for c in changes if c.get("source") == "names"]
    assert (change["old"], change["new"]) == (["AYOUR"], ["AYOUB"])


def test_both_sides_are_proposed_on_a_tie(ollama, no_names_json):
    ollama.answer = lambda lines: lines
    ollama.names_answer = lambda names: [{"arabic_of_latin": "أيوب", "latin_of_arabic": "AYRUB"}]
    changes, _ = sc.propose(NAME_DOCUMENT)
    assert {tuple(c["new"]) for c in changes if c.get("source") == "names"} == {("أيوب",), ("AYRUB",)}


@pytest.mark.parametrize("answer", [
    {"arabic_of_latin": "هشام", "latin_of_arabic": "HICHAM"},     # another name altogether
    {"arabic_of_latin": "أيرب", "latin_of_arabic": "AYOUB"},      # the readings sent back: no correction
    {},                                                          # nothing usable
])
def test_without_a_usable_proposal_the_name_is_flagged(ollama, answer, no_names_json):
    ollama.answer = lambda lines: lines
    ollama.names_answer = lambda names: [answer]
    changes, notes = sc.propose(NAME_DOCUMENT)
    [flag] = [c for c in changes if c.get("source") == "names"]
    assert flag["flag"] and flag["old"] == flag["new"] == ["أيرب"] and not flag["accepted"]
    assert "AYOUB" in flag["context"] and any("do not match" in n for n in notes)
    assert sc.apply_changes(NAME_DOCUMENT, changes, {changes.index(flag)}) == NAME_DOCUMENT


@pytest.mark.parametrize("shape", [
    lambda entry: {"names": [entry]}, lambda entry: [entry], lambda entry: entry,
])
def test_name_answers_in_other_shapes(shape):
    entry = {"arabic_of_latin": "بلال", "latin_of_arabic": "MALAL"}
    assert sc.name_answers(shape(entry), 1) == [entry]


def test_names_without_the_model(monkeypatch, no_names_json):
    monkeypatch.setattr(sc, "OLLAMA_URL", "http://127.0.0.1:9")
    changes, notes = sc.propose(NAME_DOCUMENT)
    [flag] = [c for c in changes if c.get("source") == "names"]
    assert flag["flag"]
    assert any("'أيرب'" in n and "'AYOUB'" in n for n in notes)


def test_agreeing_names_cost_no_request(ollama):
    ollama.answer = lambda lines: lines
    sc.propose(NAME_DOCUMENT.replace("أيرب", "أيوب"))
    assert not any("names" in json.loads(r["messages"][1]["content"]) for r in ollama.requests)


def test_app_shows_a_flag_that_cannot_be_accepted(ollama, no_names_json):
    pytest.importorskip("gradio")
    import app
    ollama.answer = lambda lines: lines
    ollama.names_answer = lambda names: [{}]
    changes, _ = sc.propose(NAME_DOCUMENT)
    state = {**app.NO_SPELLING, "text": NAME_DOCUMENT, "changes": changes, "accepted": []}
    view, details, state = app.spelling_review(state)
    pieces = [(p["token"], p["class_or_confidence"]) if isinstance(p, dict) else tuple(p) for p in view.value]
    assert ("أيرب", "≠ AYOUB") in pieces and "no correction found" in details
    piece = pieces.index(("أيرب", "≠ AYOUB"))

    class Click:
        index = piece
    _, _, state = app.toggle_change(state, Click())
    assert state["accepted"] == []

# ---------------------------------------------------------------------------------------------------------
# the certificate read as a whole (samples with reading mistakes, see documents.py)
# ---------------------------------------------------------------------------------------------------------
def test_names_json_fixes_the_child_name(monkeypatch):
    monkeypatch.setattr(sc, "OLLAMA_URL", "http://127.0.0.1:9")
    changes, _ = sc.propose(NAME_DOCUMENT)
    [fix] = [c for c in changes if c.get("source") == "names"]
    assert fix["accepted"] and fix["new"] == ["أيوب"] and "names.json" in fix["context"]


def test_the_certificate_is_read_as_a_whole(monkeypatch):
    monkeypatch.setattr(sc, "OLLAMA_URL", "http://127.0.0.1:9")
    changes, _ = sc.propose(COHERENCE_DOCUMENT)
    by_old = {" ".join(c["old"]): c for c in changes}
    assert by_old["خمسن"]["new"] == ["خمسة"] and by_old["خمسن"]["accepted"]         # 2015, as the Hijri date
    assert by_old["وة"]["new"] == ["و", "الدقيقة"] and by_old["وة"]["accepted"]      # the minute's label
    assert by_old["المستتشفى"]["new"] == ["المستشفى"] and by_old["المستتشفى"]["accepted"]
    assert by_old["إدريش"]["new"] in (["إدريس"], ["ادريس"]) and by_old["إدريش"]["accepted"]   # names.json + IDRISS
    assert by_old["الودعيري"]["new"] == ["الودغيري"] and by_old["الودعيري"]["accepted"]  # the father's name
    assert by_old["برشبد"]["new"] == ["برشيد"] and by_old["برشبد"]["accepted"]       # written on the same line
    assert by_old["حالد"]["new"] == ["خالد"] and by_old["حالد"]["accepted"]          # the father is خالد
    assert "رحال" not in by_old                                       # in "سيدي رحال": a place, not a name
    assert not by_old["اليوسفة"]["accepted"]                                          # proposed only
    assert by_old["أسي"]["new"] == ["آسفي"] and "options" not in by_old["أسي"]       # أسا: another region
    assert by_old["591"]["check"] and by_old["591"]["new"] == ["519"] and not by_old["591"]["accepted"]
    assert by_old["519"]["check"] and by_old["519"]["new"] == ["591"]                 # the address twice
    record, issues = sc.record_of(sc.apply_changes(COHERENCE_DOCUMENT, changes))
    assert (record["HEURENAISS"], record["MINNAISS"], record["DATENAISSFR"]) == (4, 30, "2015-01-03")
    assert record["NOMAR"] == "الودغيري" and record["NOMMEREAR"] == "بنت خالد"
    for c in changes:
        assert c["old"] != ["ثالث"] and c["old"] != ["الرابعة"]                       # correct words left alone


def test_an_article_is_kept(monkeypatch):
    """"بنت عبد العزير": the first name of names.json is written with the article read ("العزيز")."""
    monkeypatch.setattr(sc, "OLLAMA_URL", "http://127.0.0.1:9")
    text = sample(3).replace("بنت عبد العزيز", "بنت عبد العزير")
    changes, _ = sc.propose(text)
    [fix] = [c for c in changes if c["old"] == ["العزير"]]
    assert fix["new"] == ["العزيز"] and fix["accepted"]                # written so for the father's father


def test_an_unreadable_gregorian_date_gets_the_hijri_one(monkeypatch):
    monkeypatch.setattr(sc, "OLLAMA_URL", "http://127.0.0.1:9")
    text = COHERENCE_DOCUMENT.replace("ثالث يناير سنة ألفين و خمسن عشرة", "ثالث من دعر رسنة")
    changes, _ = sc.propose(text)
    [date] = [c for c in changes if c.get("source") == "dates" and c["line"] == 3]   # (line 24: the mother's year)
    assert not date["accepted"] and date["check"] and len(date["options"]) == 3
    import datetime
    days = [sc.reads_as("DATENAISSFR", " ".join(o)) for o in date["options"]]
    expected = sc.hijri_to_gregorian(1436, 3, 12)
    assert days[0] == expected.isoformat()
    assert sorted(days) == [(expected + datetime.timedelta(days=d)).isoformat() for d in (-1, 0, 1)]
    record, _ = sc.record_of(sc.apply_changes(text, changes, {changes.index(date)}))
    assert record["DATENAISSFR"] == expected.isoformat()


@pytest.mark.parametrize("year", [1972, 1977, 1980, 1981, 1999, 2000, 2003, 2004, 2008, 2011, 2019, 2024])
def test_year_words_read_back(year):
    import datetime
    for day in (1, 9, 10, 11, 13, 20, 21, 28, 30, 31):
        try:
            date = datetime.date(year, 1 if day <= 30 else 3, day)
        except ValueError:
            continue
        assert sc.reads_as("DATENAISSFR", sc.gregorian_words(date)) == date.isoformat()


def test_a_form_word_added_by_the_model(ollama, monkeypatch):
    """The model restores "الدقيقة": kept when the record needs it (the rule is switched off to see the model)."""
    monkeypatch.setattr(sc, "minute_label", lambda lines, changes: [])
    text = COHERENCE_DOCUMENT.replace("وة : الثلاثين.", "و الثلاثين.")
    ollama.answer = lambda lines: [l.replace("و الثلاثين.", "و الدقيقة الثلاثين.") for l in lines]
    changes, _ = sc.propose(text)
    [added] = [c for c in changes if c.get("source") == "added"]
    assert added["accepted"] and added["new"] == ["الدقيقة", "الثلاثين"]
    record, _ = sc.record_of(sc.apply_changes(text, changes))
    assert (record["HEURENAISS"], record["MINNAISS"]) == (4, 30)


def test_a_name_added_by_the_model_is_never_proposed(ollama):
    ollama.answer = lambda lines: [l.replace("خالد الودغيري", "خالد محمد الودغيري") for l in lines]
    changes, _ = sc.propose(COHERENCE_DOCUMENT)
    assert not [c for c in changes if c.get("source") == "added"]


def test_the_declarant_and_the_dates(monkeypatch):
    monkeypatch.setattr(sc, "OLLAMA_URL", "http://127.0.0.1:9")
    changes, _ = sc.propose(DECLARANT_DOCUMENT)
    by_old = {" ".join(c["old"]): c for c in changes}
    # day and month misread: the Hijri date gives the date, the closest day to what was read (4, not 5)
    date = next(c for c in changes if c.get("source") == "dates")
    assert date["accepted"] and date["new"][:2] == ["رابع", "يناير"]
    assert "فارس" not in by_old                                    # "اولاد فارس" is a commune (not Fès)
    mother = by_old["عرلاب"]                                        # the declarant is the mother
    assert mother["new"] == ["غزلان"] and not mother["accepted"]    # 3 letters: proposed only
    assert mother["check"]                                          # orange in the review, applied by a click
    assert by_old["عمرهار"]["new"] == ["عمرها"] and by_old["الرق"]["new"] == ["الرقم"]
    assert by_old["57"]["check"] and by_old["57"]["new"] == ["75"] and not by_old["57"]["accepted"]
    assert by_old["75"]["new"] == ["57"]
    record, issues = sc.record_of(sc.apply_changes(DECLARANT_DOCUMENT, changes))
    assert record["DATENAISSFR"] == "2008-01-04"
    assert not [i for i in issues if i["level"] == "error"]


def test_the_year_on_the_next_line(monkeypatch):
    """The scan cuts the date after the month: the day and the month are proposed, the year line is kept. Too
    many letters differ to apply it by itself: in orange, a click applies it."""
    monkeypatch.setattr(sc, "OLLAMA_URL", "http://127.0.0.1:9")
    changes, _ = sc.propose(DECLARANT_TWO_LINES)
    date = next(c for c in changes if c.get("source") == "dates")
    assert date["check"] and date["new"] == ["رابع", "يناير"]
    record, _ = sc.record_of(sc.apply_changes(DECLARANT_TWO_LINES, changes,
                                              {i for i, c in enumerate(changes) if c["accepted"] or c is date}))
    assert record["DATENAISSFR"] == "2008-01-04"


def test_app_a_check_is_orange_and_clickable(monkeypatch):
    app = pytest.importorskip("app")
    monkeypatch.setattr(sc, "OLLAMA_URL", "http://127.0.0.1:9")
    changes, _ = sc.propose(DECLARANT_DOCUMENT)
    accepted = [i for i, c in enumerate(changes) if c["accepted"]]
    state = {**app.NO_SPELLING, "text": DECLARANT_DOCUMENT, "changes": changes, "accepted": accepted}
    view, details, state = app.spelling_review(state)
    pieces = [(p["token"], p["class_or_confidence"]) if isinstance(p, dict) else tuple(p) for p in view.value]
    assert ("57", "≠ 75") in pieces and view.color_map["≠ 75"] == app.FLAG_COLOR
    index = next(i for i, c in enumerate(changes) if c["old"] == ["57"])
    segment = state["segments"].index(index)
    view, details, state = app.toggle_change(state, type("Evt", (), {"index": segment})())
    pieces = [(p["token"], p["class_or_confidence"]) if isinstance(p, dict) else tuple(p) for p in view.value]
    assert ("57", "✓ 75") in pieces and index in state["accepted"]


def test_the_child_family_name_and_the_declarant(monkeypatch):
    """الإيريشي / IDRISSI: the father is الإدريسي (two letters); the declarant's الإدريسن is the misreading, not
    the father's line (the child's Latin name decides)."""
    monkeypatch.setattr(sc, "OLLAMA_URL", "http://127.0.0.1:9")
    changes, _ = sc.propose(FATHER_DECLARANT_DOCUMENT)
    by_old = {" ".join(c["old"]): c for c in changes}
    assert by_old["الإيريشي"]["new"] == ["الإدريسي"] and by_old["الإيريشي"]["accepted"]
    assert by_old["الإدريسن"]["new"] == ["الإدريسي"] and by_old["الإدريسن"]["accepted"]
    assert "الإدريسي" not in by_old                                     # the father's reading is kept
    assert by_old["ثمانة"]["new"] == ["ثمانية"]                         # the last letter kept
    year = by_old["ألف و تسعمائة و تسعة و ثمانين"]                       # 3 Rabi I 1407 = 6 November 1986
    assert year["check"] and not year["accepted"] and year["new"] == "ألف و تسعمائة و ستة و ثمانين".split()
    date = by_old["ثاني عشر تير"]                                       # the year on the next line
    assert date["accepted"] and date["new"] == ["ثاني", "عشر", "نونبر"]
    record, _ = sc.record_of(sc.apply_changes(FATHER_DECLARANT_DOCUMENT, changes))
    assert record["NOMAR"] == "الإدريسي" and record["DATENAISSFR"] == "2016-11-12"
    record, _ = sc.record_of(sc.apply_changes(FATHER_DECLARANT_DOCUMENT, changes,
                                              {i for i, c in enumerate(changes) if c["accepted"] or c is year}))
    assert record["DATENAISSPEREFR"] == "1986-11-06"


def test_an_empty_minute_and_the_address(monkeypatch):
    monkeypatch.setattr(sc, "OLLAMA_URL", "http://127.0.0.1:9")
    record, issues = sc.record_of(EMPTY_MINUTE_DOCUMENT)
    assert (record["HEURENAISS"], record["MINNAISS"]) == (16, "")       # "و الدقيقة" alone: empty, not the hour
    assert [i["level"] for i in issues if i["field"] == "MINNAISS"] == ["warning"]
    changes, _ = sc.propose(EMPTY_MINUTE_DOCUMENT)
    by_old = {" ".join(c["old"]): c for c in changes}
    # a neighbourhood, not "القروية": only the reading of the other address is offered, in orange
    assert by_old["القاومة"]["check"] and by_old["القاومة"]["new"] == ["المعاومة"]
    assert by_old["العمرانيبست"]["new"] == ["العمراني", "بنت"] and by_old["العمرانيبست"]["accepted"]
    assert by_old["19"]["check"] and by_old["19"]["new"] == ["1"]         # the address after the mother's job
    record, _ = sc.record_of(sc.apply_changes(EMPTY_MINUTE_DOCUMENT, changes))
    assert record["NOMMEREAR"] == "بنت رضا" and record["NOMAR"] == "الساحلي"


def test_the_neighbourhoods_of_spelling_words(monkeypatch):
    monkeypatch.setattr(sc, "OLLAMA_URL", "http://127.0.0.1:9")
    monkeypatch.setattr(sc, "user_words", lambda: frozenset({"المقاومة"}))
    changes, _ = sc.propose(EMPTY_MINUTE_DOCUMENT)
    fixed = sorted(" ".join(c["old"]) for c in changes if c["new"] == ["المقاومة"] and c["accepted"])
    assert fixed == ["القاومة", "المعاومة", "المناومة"]


def test_only_new_lines_go_to_the_model(ollama):
    """After a hand edit, the lines the model already answered are not sent again."""
    _, damaged = damaged_sample()
    ollama.answer = lambda lines: [l.replace("موفق", "موافق") for l in lines]
    sc.propose(damaged)
    first = len(json.loads(ollama.requests[-1]["messages"][1]["content"])["lines"])
    edited = damaged.replace("الساكنان", "الساكنان المعدلان", 1)
    changes, _ = sc.propose(edited)
    sent = json.loads(ollama.requests[-1]["messages"][1]["content"])["lines"]
    assert first > 5 and len(sent) == 1 and "المعدلان" in sent[0]["text"]
    assert any(c["old"] == ["موفق"] and c["accepted"] for c in changes)   # the cached answers still count

def test_the_street_name_of_both_addresses(monkeypatch):
    """samples/6.txt: "الساكنان بتجزئة الفتح" / "الساكن بتجزئة الفيق" (the same place read twice): each reading
    gets the other one in orange; the reading listed in spelling_words.txt is applied to the other one."""
    monkeypatch.setattr(sc, "OLLAMA_URL", "http://127.0.0.1:9")
    text = sample(6).replace("الساكن بتجزئة الفيق", "الساكن ب تجزئة الفيقة")   # the label cut from "ب"
    changes, _ = sc.propose(text)
    by_old = {" ".join(c["old"]): c for c in changes if c.get("source") == "address"}
    assert by_old["الفتح"]["check"] and by_old["الفتح"]["new"] == ["الفيقة"] and not by_old["الفتح"]["accepted"]
    assert by_old["الفيقة"]["check"] and by_old["الفيقة"]["new"] == ["الفتح"]
    monkeypatch.setattr(sc, "user_words", lambda: frozenset({"الفتح"}))
    changes, _ = sc.propose(text)
    [fix] = [c for c in changes if c.get("source") == "address"]
    assert fix["old"] == ["الفيقة"] and fix["new"] == ["الفتح"] and fix["accepted"]


def test_the_same_street_name_is_not_flagged(monkeypatch):
    """"بحي الامل" / "بحي الأمل" (samples/2.txt): the same name for the parser, nothing to check."""
    monkeypatch.setattr(sc, "OLLAMA_URL", "http://127.0.0.1:9")
    changes, _ = sc.propose(sample(2))
    assert not [c for c in changes if c.get("source") == "address"]