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
    thread = threading.Thread(target=server.serve_forever, daemon=True)
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
    text, notes = sc.correct_text(damaged)
    assert "موفق" in text and "unusable answer" in notes[0]


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
    """'من والد' + 'ها حسن هنور' on two lines, a misspelt label: the name is still protected."""
    protected, inside = sc.protected_positions("من والد".split())
    assert inside
    protected, _ = sc.protected_positions("ها حسن هنور بن العربي".split(), inside=inside)
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
    text = "و والدتها نادية بنت ناصر.\nحرفته صباخ\nجنسيتها : مغربية\n"
    ollama.answer = lambda lines: lines                      # the model proposes nothing
    changes, _ = sc.propose(text)
    job = next(c for c in changes if c.get("source") == "jobs.json")
    assert job["old"] == ["صباخ"] and job["options"] == [["صباغ"], ["طباخ"]] and not job["accepted"]
    ollama.answer = lambda lines: [l.replace("صباخ", "طباخ") for l in lines]   # the model chooses: first
    changes, _ = sc.propose(text)
    job = next(c for c in changes if c.get("source") == "jobs.json")
    assert job["options"][0] == ["طباخ"] and len([c for c in changes if c["old"] == ["صباخ"]]) == 1
    assert sc.apply_changes(text, changes) == text           # several jobs: nothing applied by itself


def test_a_single_possible_job_is_applied(ollama):
    text = "و والدتها نادية بنت ناصر.\nحرفته : فلاحح\n"
    ollama.answer = lambda lines: lines
    changes, _ = sc.propose(text)
    assert [(c["old"], c["new"], c["accepted"]) for c in changes] == [(["فلاحح"], ["فلاح"], True)]
    assert sc.apply_changes(text, changes) == "و والدتها نادية بنت ناصر.\nحرفته : فلاح\n"


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
    return (
        "من والدها: عمر قبلي بن صالح\n"
        "و والدتها نادية ايت احريس بنت ناصر.\n"
        f"حرفته {father_job}\n"
        "جنسيتها : مغربية\n"
        "حرفتها : ربة البيت\n"
        f"بناءا على ما صرح به السيد عمر قبلي ({relation}) تحت عدد : 18 عمره ثمانية و ستون سنة\n"
        f"حرفته {declarant_job} جنسيته مغربية الساكن بحي القدس الأول\n"
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
DATE_DOCUMENT = (
    "رسم ولادة رقم: 42\n"
    "في يوم ثالث و عشري ذو الحجة\n"
    "عام : ألف و أربعمائة و ثمانية و عشرين هجرية\n"
    "موافق ثالث يناير سنة ألفين و ثمانين ميلادية.\n"
    "على الساعة: الخامسة عشرة\n"
    "الإسم الشخصي : بلال\n"
    "ذكر\n"
    "من والده عبد الجبار قاسم بن محمد\n"
    "حرفته : مستخدم\n"
    "و والدته عتيقة ممير بنت عبد العزيز.\n"
    "الساكنان بحي الأمل الرقم: 494 خريبڭة.\n"
    "بناءا على ما صرحت به السيدة عتيقة ممير (والدته) تحت عدد : 38\n"
    "وحرر في تاسع محرم عام ألف و أربعمائة و تسعة و عشرين هجرية\n"
    "موافق ثامن عشر يناير سنة ألفين و ثمانين ميلادية.\n"
    "من طرفنا نحن عبد الرحيم ڭلفاع موظف جماعي\n"
)


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
    ollama.answer = lambda lines: [l.replace("ثالث و عشري", "ثالث و عشرين") for l in lines]
    changes, _ = sc.propose(DATE_DOCUMENT)
    assert [(c["old"], c["accepted"]) for c in changes] == [(["عشري"], False)]


def test_a_number_change_that_breaks_a_date_is_refused(ollama):
    ollama.answer = lambda lines: [l.replace("ثمانية و عشرين هجرية", "ثمانية و ثلاثين هجرية") for l in lines]
    changes, _ = sc.propose(DATE_DOCUMENT)
    assert not any(c["accepted"] for c in changes)


def test_a_right_job_gets_no_proposal(ollama):
    """'مستخدم' is a job of jobs.json: the model's 'صباغ' is not even shown."""
    ollama.answer = lambda lines: [l.replace("مستخدم", "صباغ") for l in lines]
    changes, _ = sc.propose(DATE_DOCUMENT)
    assert not [c for c in changes if c["old"] == ["مستخدم"]]


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


NAME_DOCUMENT = (
    "في يوم ثالث و عشري ذو الحجة\n"
    "PRENOM BILAL\n"
    "NOM KACIM\n"
    "الإسم الشخصي : ملال\n"
    "ذكر\n"
    "الإسم العائلي قاسم\n"
    "من والده عبد الجبار قاسم بن محمد\n"
)


def test_a_misread_arabic_name_gets_a_proposal(ollama):
    """'ملال' / 'BILAL': the Arabic of BILAL is 'بلال' (1 letter), the Latin of ملال 'MALAL' (2 letters):
    the Arabic gets the proposal, not applied (names are decided by a person)."""
    ollama.answer = lambda lines: lines
    ollama.names_answer = lambda names: [{"arabic_of_latin": "بلال", "latin_of_arabic": "MALAL"}]
    changes, _ = sc.propose(NAME_DOCUMENT)
    [change] = [c for c in changes if c.get("source") == "names"]
    assert (change["old"], change["new"], change["accepted"]) == (["ملال"], ["بلال"], False)
    assert "BILAL" in sc.describe(change)
    assert sc.apply_changes(NAME_DOCUMENT, changes, {changes.index(change)}).count("الإسم الشخصي : بلال") == 1
    assert sc.apply_changes(NAME_DOCUMENT, changes) == NAME_DOCUMENT      # nothing applied by default


def test_a_misread_latin_name_gets_a_proposal(ollama):
    text = NAME_DOCUMENT.replace("ملال", "بلال").replace("BILAL", "BIRAL")
    ollama.answer = lambda lines: lines
    ollama.names_answer = lambda names: [{"arabic_of_latin": "بيرال", "latin_of_arabic": "BILAL"}]
    changes, _ = sc.propose(text)
    [change] = [c for c in changes if c.get("source") == "names"]
    assert (change["old"], change["new"]) == (["BIRAL"], ["BILAL"])


def test_both_sides_are_proposed_on_a_tie(ollama):
    ollama.answer = lambda lines: lines
    ollama.names_answer = lambda names: [{"arabic_of_latin": "بلال", "latin_of_arabic": "MILAL"}]
    changes, _ = sc.propose(NAME_DOCUMENT)
    assert {tuple(c["new"]) for c in changes if c.get("source") == "names"} == {("بلال",), ("MILAL",)}


@pytest.mark.parametrize("answer", [
    {"arabic_of_latin": "هشام", "latin_of_arabic": "HICHAM"},     # another name altogether
    {"arabic_of_latin": "ملال", "latin_of_arabic": "BILAL"},      # the readings sent back: no correction
    {},                                                          # nothing usable
])
def test_without_a_usable_proposal_the_name_is_flagged(ollama, answer):
    ollama.answer = lambda lines: lines
    ollama.names_answer = lambda names: [answer]
    changes, notes = sc.propose(NAME_DOCUMENT)
    [flag] = [c for c in changes if c.get("source") == "names"]
    assert flag["flag"] and flag["old"] == flag["new"] == ["ملال"] and not flag["accepted"]
    assert "BILAL" in flag["context"] and any("do not match" in n for n in notes)
    assert sc.apply_changes(NAME_DOCUMENT, changes, {changes.index(flag)}) == NAME_DOCUMENT


@pytest.mark.parametrize("shape", [
    lambda entry: {"names": [entry]}, lambda entry: [entry], lambda entry: entry,
])
def test_name_answers_in_other_shapes(shape):
    entry = {"arabic_of_latin": "بلال", "latin_of_arabic": "MALAL"}
    assert sc.name_answers(shape(entry), 1) == [entry]


def test_names_without_the_model(monkeypatch):
    monkeypatch.setattr(sc, "OLLAMA_URL", "http://127.0.0.1:9")
    changes, notes = sc.propose(NAME_DOCUMENT)
    [flag] = [c for c in changes if c.get("source") == "names"]
    assert flag["flag"]
    assert any("'ملال'" in n and "'BILAL'" in n for n in notes)


def test_agreeing_names_cost_no_request(ollama):
    ollama.answer = lambda lines: lines
    sc.propose(NAME_DOCUMENT.replace("ملال", "بلال"))
    assert not any("names" in json.loads(r["messages"][1]["content"]) for r in ollama.requests)


def test_app_shows_a_flag_that_cannot_be_accepted(ollama):
    pytest.importorskip("gradio")
    import app
    ollama.answer = lambda lines: lines
    ollama.names_answer = lambda names: [{}]
    changes, _ = sc.propose(NAME_DOCUMENT)
    state = {**app.NO_SPELLING, "text": NAME_DOCUMENT, "changes": changes, "accepted": []}
    view, details, state = app.spelling_review(state)
    pieces = [(p["token"], p["class_or_confidence"]) if isinstance(p, dict) else tuple(p) for p in view.value]
    assert ("ملال", "≠ BILAL") in pieces and "no correction found" in details
    piece = pieces.index(("ملال", "≠ BILAL"))

    class Click:
        index = piece
    _, _, state = app.toggle_change(state, Click())
    assert state["accepted"] == []