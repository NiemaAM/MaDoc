"""The NER as the second reading of the parser: how its entities are placed in the text and reconciled with the
rules (parser.py: build_entities, ner_candidates, reconcile, fill_french_names).

The entities are written by hand on samples/1.txt, the way MagBERT gives them (a label, the exact text and its
offsets in the document), so no model is loaded."""
import parser as record_parser

from documents import moved, sample

# (text, label, which occurrence in the document)
SAMPLE_1_ENTITIES = [
    ("حادي عشر صفر", "DATE", 1), ("ألف و أربعمائة و ثمانية و ثلاثين", "DATE", 1),
    ("خامس عشر دجنبر سنة ألفين و ستة عشرة", "DATE", 1), ("خريبڭة", "GPE", 1), ("هشام", "PERSON", 1),
    ("الإدريسي", "PERSON", 1), ("نبيل الإدريسي بن عادل", "PERSON", 1), ("خريبڭة", "GPE", 3),
    ("ثالث ربيع الأول عام ألف و أربعمائة و سبعة", "DATE", 1), ("تاسع دجنبر سنة ألف و تسعمائة و ستة و ثمانين", "DATE", 1),
    ("ريم الشاوي بنت أنس", "PERSON", 1), ("امرتريك", "GPE", 1), ("سطات", "GPE", 1),
    ("ثامن عشر رجب عام ألف و أربعمائة و ثمانية عشرة", "DATE", 1),
    ("حادي و عشري دجنبر سنة ألف و تسعمائة و سبعة و تسعين", "DATE", 1), ("نبيل الإدريسي", "PERSON", 2),
    ("فاتح ربيع الأول عام ألف و أربعمائة و ثمانية و ثلاثين", "DATE", 1),
    ("ثالث يناير سنة ألفين و سبعة عشرة", "DATE", 1), ("فؤاد السعدي", "PERSON", 1),
]


def entity(text, word, occurrence=1, label="PERSON"):
    position = -1
    for _ in range(occurrence):
        position = text.find(word, position + 1)
    assert position >= 0, word
    return {"entity": word, "label": label, "start": position, "end": position + len(word), "score": 0.99}


def ner_of(text, french=("HICHAM", "IDRISSI")):
    """The NER json of `text`: the entities of SAMPLE_1_ENTITIES still found in it, offsets in this text."""
    return {"arabic_ner": [entity(text, w, n, label) for w, label, n in SAMPLE_1_ENTITIES if text.count(w) >= n],
            "french_ner": [entity(text, w) for w in french if w in text]}


def differences(record, expected):
    return {field: record[field] for field in expected if record[field] != expected[field]}


def test_ner_that_agrees_changes_nothing():
    text = sample(1)
    record, notes = record_parser.parse_with_notes(text, ner_of(text))
    assert record == record_parser.parse(text)
    assert not [n for n in notes if "filled from NER" in n or "replaced by NER" in n]


def test_ner_fills_a_value_the_rules_missed():
    """The label of the first name misread: the rules have no first name, the NER gives it."""
    expected = record_parser.parse(sample(1))
    text = sample(1).replace("الإسم الشخصي : هشام", "هشام")
    assert record_parser.parse(text)["PRENOMAR"] == ""
    record, notes = record_parser.parse_with_notes(text, ner_of(text))
    assert record["PRENOMAR"] == expected["PRENOMAR"]
    assert "PRENOMAR: empty after the rules, filled from NER: 'هشام'" in notes


def test_ner_offsets_follow_the_lines_put_back_in_order():
    expected = record_parser.parse(sample(1))
    text = moved(sample(1), 20, 23)               # the mother's line read after her dates
    record, _ = record_parser.parse_with_notes(text, ner_of(text))
    assert not differences(record, expected)


def test_ner_without_offsets_or_with_unknown_entities():
    text = sample(1)
    expected = record_parser.parse(text)
    no_offsets = {"arabic_ner": [{k: v for k, v in e.items() if k not in ("start", "end")}
                                 for e in ner_of(text)["arabic_ner"]]}
    record, notes = record_parser.parse_with_notes(text, no_offsets)
    assert record == expected and any("no usable offsets" in n for n in notes)
    wrong = {"arabic_ner": [*ner_of(text)["arabic_ner"], {"entity": "كلمة غير موجودة", "label": "PER"}]}
    record, notes = record_parser.parse_with_notes(text, wrong)
    assert record == expected and any("'كلمة غير موجودة' (PERSON) not found" in n for n in notes)


def test_latin_names_without_their_labels():
    """"PRENOM" and "NOM" not read: the Latin names went on the Arabic ones; they are split off again, with or
    without the NER."""
    expected = record_parser.parse(sample(1))
    text = sample(1).replace("PRENOM HICHAM", "HICHAM").replace("NOM IDRISSI", "IDRISSI")
    for ner in (None, ner_of(text)):
        record, notes = record_parser.parse_with_notes(text, ner)
        assert not differences(record, expected), ner is not None
        assert any("PRENOM: its label was not read" in n for n in notes)


def test_french_names_from_the_french_ner():
    parsed = {"PRENOM": "", "NOM": "", "PRENOMAR": "هشام"}
    notes = []
    record_parser.fill_french_names(parsed, ["HICHAM", "IDRISSI"], notes)
    assert parsed["PRENOM"] == "HICHAM" and parsed["NOM"] == "IDRISSI" and len(notes) == 2
    parsed = {"PRENOM": "HICHAM", "NOM": "", "PRENOMAR": "هشام"}
    record_parser.fill_french_names(parsed, ["HICHAM", "IDRISSI"], notes)
    assert parsed["NOM"] == "IDRISSI"


def test_reconcile_decisions():
    """What the parser does with the rules' value and the NER's value of one field."""
    def decide(field, rules, candidate):
        parsed, notes = {field: rules}, []
        record_parser.reconcile(parsed, {field: candidate}, notes)
        return parsed[field], notes

    # the rules read nothing: the NER value
    assert decide("PRENOMAR", "", "هشام") == ("هشام", ["PRENOMAR: empty after the rules, filled from NER: 'هشام'"])
    # the same, or nothing from the NER: the rules, no note
    assert decide("PRENOMAR", "هشام", "هشام") == ("هشام", [])
    assert decide("PRENOMAR", "هشام", "") == ("هشام", [])
    # a label swallowed by the rules: the NER value
    value, notes = decide("PRENOMPEREAR", "نبيل جنسيته", "نبيل")
    assert value == "نبيل" and "a label was swallowed" in notes[0]
    # another difference: the rules are kept, the difference noted
    value, notes = decide("PRENOMAR", "هشام", "سعيد")
    assert value == "هشام" and "NER says 'سعيد'" in notes[0]
    # the employee is compared without his job
    assert decide("EMPLOYECREANT", "فؤاد السعدي موظف جماعي", "فؤاد السعدي")[1] == []
    # dates: the valid one wins; two valid and different dates keep the rules, with a note
    value, notes = decide("DATENAISSFR", "خامس عشر دجنبر سنة", "خامس عشر دجنبر سنة ألفين و ستة عشرة")
    assert value == "خامس عشر دجنبر سنة ألفين و ستة عشرة" and "not a valid date" in notes[0]
    value, notes = decide("DATENAISSFR", "خامس عشر دجنبر سنة ألفين و ستة عشرة", "سادس عشر دجنبر سنة ألفين و ستة عشرة")
    assert value.startswith("خامس") and "both valid, rules kept" in notes[0]
    value, notes = decide("DATENAISSFR", "دجنبر", "سنة")
    assert value == "دجنبر" and "neither the rules" in notes[0]