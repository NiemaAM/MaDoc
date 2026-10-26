"""Unit tests of the helpers around the chain: file reading, kraken calls, translator output, app files.

No model, no kraken, no network: kraken is replaced by a fake command and Ollama by a
small local HTTP server.

    python -m pytest tests/test_units.py
"""

import json
import os
import sys
import time
from pathlib import Path

import pytest

import common
import converter
import recognition
import segmentation
import translator
from common import read_text_file
from documents import APP_TEXT, NATIONALITY_DOCUMENT

JOB = "موظف بالصناعة التقليدية"   # the father's job in APP_TEXT (samples/2.txt)

ROOT = Path(__file__).resolve().parents[1]
TEXT = "رسم ولادة رقم 01\nفي يوم ثالث و عشري شوال\n"


# ---------------------------------------------------------------------------------------------------------
# common
# ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16", "cp1256"])
def test_read_text_file_encodings(tmp_path, encoding):
    """Every usual encoding of the scans' text files gives the same text (UTF-16 used to come back garbled)."""
    path = tmp_path / "doc.txt"
    path.write_bytes(TEXT.encode(encoding))
    assert read_text_file(path) == TEXT


def test_read_text_file_line_ends(tmp_path):
    path = tmp_path / "doc.txt"
    path.write_bytes(TEXT.replace("\n", "\r\n").encode("utf-8"))
    assert read_text_file(path) == TEXT


def test_read_text_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_text_file(tmp_path / "nothing.txt")


def test_ar_key():
    assert common.ar_key("رَسْمُ وِلادة") == "رسم ولاده"
    assert common.ar_key("الخمسين") == common.ar_key("خمسين")
    assert common.ar_key(5) == "5"


def test_kraken_command():
    """kraken is started through the running python when it is installed (a moved .venv breaks kraken.exe)."""
    import importlib.util
    try:
        installed = importlib.util.find_spec("kraken.kraken") is not None
    except ImportError:
        installed = False
    cmd = common.kraken_command()
    assert cmd == ["kraken"] if not installed else cmd[0] == sys.executable


# ---------------------------------------------------------------------------------------------------------
# converter
# ---------------------------------------------------------------------------------------------------------
def test_converter_dates_and_time():
    assert converter.replace_aliases("ثالث و عشري شوال عام ألف و أربعمائة و أربعة و عشرين") == "1424-10-23"
    assert converter.replace_aliases("ثامن عشر دجنبر سنة الفين و ثلاثة") == "2003-12-18"
    assert converter.replace_aliases("عام ألف و أربعمائة و واحد") == "1401"
    assert converter.time_value("السادسة عشرة") == 16
    assert converter.time_value("الصفر") == 0


def test_key_shape_is_computed_once():
    keys = converter.get_number_keys()
    assert converter.key_shape(keys) is converter.key_shape(keys)
    other = {"واحد اثنان": 1}
    assert converter.key_shape(other) == (frozenset({"واحد"}), 2)


# ---------------------------------------------------------------------------------------------------------
# segmentation / recognition (kraken replaced by a fake command)
# ---------------------------------------------------------------------------------------------------------
def fake_kraken(monkeypatch, module, text="سطر اول\n\nسطر ثاني\n"):
    """Replace subprocess.run of a module: records the command, writes the output file kraken would write."""
    calls = []

    def run(cmd, check):
        calls.append(cmd)
        output = Path(cmd[cmd.index("-i") + 2])
        if "segment" in cmd:
            output.write_text('<PcGts><Page imageFilename="%s"/></PcGts>' % cmd[cmd.index("-i") + 1], encoding="utf-8")
        elif "-x" in cmd:
            output.write_text(
                '<PcGts><Page><TextLine id="l1"><Coords points="0,0 10,0 10,5 0,5"/>'
                '<TextEquiv conf="0.9"><Unicode>لوا رطس</Unicode></TextEquiv></TextLine></Page></PcGts>',
                encoding="utf-8",
            )
        else:
            output.write_text(text, encoding="utf-8")

    monkeypatch.setattr("subprocess.run", run)
    return calls


def test_kraken_errors_are_raised(monkeypatch, tmp_path):
    """kraken is run with -r (its real error stops it) and an output it did not write is a clear error."""
    calls = []
    monkeypatch.setattr("subprocess.run", lambda cmd, check: calls.append(cmd))   # writes nothing
    image = tmp_path / "scan.jpg"
    image.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="segmentation failed"):
        segmentation.segment_image(image, "models/blla.mlmodel", "cpu")
    assert "-r" in calls[0] and calls[0].index("-r") < calls[0].index("segment")


class FakeTorch:
    """torch stand-in: a CPU-only build, or a build with one GPU."""
    __version__ = "2.5.0+cpu"

    def __init__(self, gpus):
        self.cuda = type("cuda", (), {"is_available": staticmethod(lambda: gpus > 0),
                                      "device_count": staticmethod(lambda: gpus)})
        self.backends = type("backends", (), {"cuda": type("c", (), {"is_built": staticmethod(lambda: gpus > 0)})})


@pytest.mark.parametrize("gpus,device,expected", [
    (0, "cuda:0", "cpu"),          # CPU-only torch (the default pip install on Windows)
    (1, "cuda:0", "cuda:0"),
    (1, "cuda:1", "cpu"),          # there is no second GPU
    (0, "cpu", "cpu"),
])
def test_resolve_device(monkeypatch, capsys, gpus, device, expected):
    monkeypatch.setitem(sys.modules, "torch", FakeTorch(gpus))
    common.resolve_device.cache_clear()
    assert common.resolve_device(device) == expected
    assert ("cannot be used" in capsys.readouterr().err) == (expected != device)
    common.resolve_device.cache_clear()


def test_segmentation_returns_the_xml(monkeypatch, tmp_path):
    image = tmp_path / "scan.jpg"
    image.write_bytes(b"x")
    calls = fake_kraken(monkeypatch, segmentation)
    xml = segmentation.segment_image(image, "models/blla.mlmodel", "cpu")
    assert calls[0][-4:] == ["segment", "-bl", "-i", "models/blla.mlmodel"]
    assert str(image.resolve()) in xml


def test_recognition_text(monkeypatch, tmp_path):
    image = tmp_path / "scan.jpg"
    image.write_bytes(b"x")
    fake_kraken(monkeypatch, recognition)
    xml = '<PcGts><Page imageFilename="%s"/></PcGts>' % image
    assert recognition.recognize_xml(xml, None, "m", "cpu", fix_order=False) == "سطر اول\nسطر ثاني\n"


def test_recognition_escapes_the_image_path(monkeypatch, tmp_path):
    """A folder named "a & 'b'" must not break the xml given to kraken ('"' is not allowed on Windows)."""
    folder = tmp_path / "a & 'b'"
    folder.mkdir()
    image = folder / "scan.jpg"
    image.write_bytes(b"x")
    xml = recognition.prepare_xml('<PcGts><Page imageFilename="gone.jpg"/></PcGts>', image)
    from xml.etree import ElementTree
    page = ElementTree.fromstring(xml).find("Page")
    assert page.get("imageFilename") == str(image.resolve())


def test_recognition_lines_in_logical_order(monkeypatch, tmp_path):
    pytest.importorskip("bidi")
    image = tmp_path / "scan.jpg"
    image.write_bytes(b"x")
    fake_kraken(monkeypatch, recognition)
    lines = recognition.recognize_lines('<PcGts><Page imageFilename="%s"/></PcGts>' % image, None, "m", "cpu")
    assert lines[0]["text"] == "سطر اول" and lines[0]["confidence"] == 0.9 and lines[0]["polygon"]


# ---------------------------------------------------------------------------------------------------------
# translator: offline behaviour
# ---------------------------------------------------------------------------------------------------------
def test_translator_messages_never_go_to_stdout(monkeypatch, capsys):
    """stdout carries the JSON record: python src/pipeline.py x.txt > record.json must stay valid JSON."""
    import spelling_checker
    monkeypatch.setattr(spelling_checker, "OLLAMA_URL", "http://127.0.0.1:9")    # nothing listens there
    record = dict(common.TEMPLATE, PRENOMPEREAR="خالد الغزواني", PROFESSIONPERE="حداد التقليدي")
    out = translator.translate(record)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Translation model unavailable" in captured.err
    assert out["PROFESSIONPEREFR"] == ""          # an untranslated job stays empty, never Arabic
    assert out["PRENOMPEREFR"] == "خالد الغزواني"  # kept whole (never half Latin), reported by the validator


def test_translator_refuses_a_remote_ollama(monkeypatch):
    import spelling_checker
    monkeypatch.delenv("MADOC_SPELL_ALLOW_REMOTE", raising=False)
    monkeypatch.setattr(spelling_checker, "OLLAMA_URL", "https://api.example.com")
    assert "not this computer" in translator.unavailable_reason()
    assert not translator.is_available()


@pytest.fixture
def translation_ollama(monkeypatch):
    """A local stand-in of Ollama: lists the model, answers every item with ANSWER[type] (ids kept)."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    import spelling_checker

    class Fake(BaseHTTPRequestHandler):
        answer = {"name_part": "Ghazouani", "address": "Khouribga", "job": "forgeron"}
        models = [{"name": "qwen2.5:7b", "model": "qwen2.5:7b"}]
        requests = []

        def reply(self, obj):
            data = json.dumps(obj).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply({"models": Fake.models})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
            Fake.requests.append(body)
            items = json.loads(body["messages"][1]["content"])["items"]
            results = [{"id": i["id"], "text": Fake.answer[i["type"]]} for i in items]
            self.reply({"message": {"content": json.dumps({"results": results}, ensure_ascii=False)}})

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    monkeypatch.setattr(spelling_checker, "OLLAMA_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setattr(translator, "MODEL", "qwen2.5:7b")
    yield Fake
    server.shutdown()


def test_translator_local_model(translation_ollama):
    record = dict(common.TEMPLATE, PRENOMPEREAR="خالد الغزواني", PROFESSIONPERE="حداد التقليدي", LIEUNAISS="خريبكة")
    out = translator.translate(record)
    assert out["PRENOMPEREFR"] == "Khalid Ghazouani"       # Khalid from names.json, Ghazouani from the model
    assert out["PROFESSIONPEREFR"] == "FORGERON"           # post-processed like jobs.json
    assert out["LIEUNAISSFR"] == "Khouribga"
    body = translation_ollama.requests[0]
    assert body["model"] == "qwen2.5:7b" and body["format"] == "json"
    assert translator.is_available()


def test_translator_model_not_pulled(translation_ollama, capsys):
    translation_ollama.models = [{"name": "llama3:8b", "model": "llama3:8b"}]
    out = translator.translate(dict(common.TEMPLATE, PRENOMPEREAR="الغزواني"))
    assert out["PRENOMPEREFR"] == "الغزواني"
    assert "ollama pull qwen2.5:7b" in capsys.readouterr().err
    assert translation_ollama.requests == []


def test_translator_rejects_arabic_answers(translation_ollama):
    translation_ollama.answer = dict(translation_ollama.answer, name_part="الغزواني")
    out = translator.translate(dict(common.TEMPLATE, PRENOMPEREAR="خالد الغزواني"))
    assert out["PRENOMPEREFR"] == "خالد الغزواني"            # the original, never the model's Arabic


def test_translator_a_sample_record(translation_ollama):
    """The record of samples/5.txt: what a 7B model got wrong ("Fille de Fille d'...", another first name, an
    English word, the officer's title twice) is written by the code."""
    from documents import sample
    import spelling_checker
    translation_ollama.answer = {"name_part": "Fille d'Saadi", "job": "x",
                                 "address": "Boulevard de l'Hôpital, Khouribga"}
    record, _ = spelling_checker.record_of(sample(5).replace("ولد ب جماعة خريبڭة الحضرية اقليم خريبڭة.",
                                                             "ولد ب محج المستشفى مدينة خريبڭة."))
    out = translator.translate(record)
    assert out["PRENOMPEREFR"] == "Khalid Oudghiri"        # names.json + the child's family name
    assert out["NOMPEREFR"] == "Fils de Rida"
    assert out["PRENOMMEREFR"] == "Loubna Saadi"
    assert out["NOMMEREFR"] == "Fille de Khalid"           # no model at all: lineage + names.json
    assert out["EMPLOYECREANTFR"] == "Fouad Saadi " + common.EMPLOYEE_TITLE
    asked = json.loads(translation_ollama.requests[0]["messages"][1]["content"])["items"]
    assert [i["text"] for i in asked if i["type"] == "name_part"] == ["الغزواني", "السعدي"]
    address = next(i for i in asked if i["type"] == "address")
    assert address["glossary"]["المستشفى"] == "Hôpital" and address["glossary"]["محج"] == "Boulevard"


def test_translator_cleans_the_model_extras():
    part = {"type": "name_part", "text": "ادريس"}
    assert translator.postprocess_output(part, "Fille d'Idriss") == "Idriss"
    assert translator.postprocess_output(part, "fils de ben idriss.") == "Idriss"
    employee = {"type": "employee", "text": "فؤاد السعدي"}
    assert translator.postprocess_output(employee, "Fouad Saadi FONCTIONNAIRE ET OFFICIER DU STATUT CIVIL") \
        == "Fouad Saadi " + common.EMPLOYEE_TITLE
    assert translator.split_lineage("بنت خالد") == ("Fille de", "خالد")
    assert translator.name_segments("عبد الرحيم الغزواني", translator.get_names_lookup()) == \
        [("عبد الرحيم", "Abderrahim"), ("الغزواني", None)]


def test_answer_results():
    assert translator.answer_results({"results": [{"id": 2, "text": "b"}, {"id": 1, "text": "a"}]}, 2) == ["a", "b"]
    assert translator.answer_results({"results": [{"id": 1, "text": "a"}]}, 2) == ["a", None]
    assert translator.answer_results({"results": ["a", "b"]}, 2) == ["a", "b"]
    assert translator.answer_results({"results": ["a"]}, 2) is None
    assert translator.answer_results({"lines": []}, 1) is None


def test_translator_postprocess():
    item = {"type": "name", "text": "بن صالح"}
    assert translator.postprocess_output(item, "Ben Salah") == "Fils de Salah"
    item = {"type": "job", "text": "حداد"}
    assert translator.postprocess_output(item, " forgeron. ") == "FORGERON"
    item = {"type": "employee", "text": "فؤاد السعدي"}
    assert translator.postprocess_output(item, "Fouad Saadi").endswith(common.EMPLOYEE_TITLE)


# ---------------------------------------------------------------------------------------------------------
# app (only when gradio is installed)
# ---------------------------------------------------------------------------------------------------------
def test_app_run_folders_are_cleaned(monkeypatch, tmp_path):
    pytest.importorskip("gradio")
    import app
    monkeypatch.setattr(app, "RUNS_DIR", tmp_path / "runs")
    old = app.new_run_folder()
    past = time.time() - app.KEEP_SECONDS - 10
    os.utime(old, (past, past))
    new = app.new_run_folder()
    assert new.is_dir() and not old.exists()


def test_app_prepare_image(tmp_path):
    pytest.importorskip("gradio")
    import app
    from PIL import Image
    plain = tmp_path / "plain.jpg"
    Image.new("RGB", (40, 20)).save(plain)
    run = tmp_path / "run"
    run.mkdir()
    copied = app.prepare_image(plain, run)                             # nothing to fix: copied as it is
    assert Path(copied).parent == run and Path(copied).read_bytes() == plain.read_bytes()
    rotated = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6                                                   # phone photo turned 90 degrees
    Image.new("RGB", (40, 20)).save(rotated, exif=exif)
    prepared = app.prepare_image(rotated, tmp_path)
    with Image.open(prepared) as image:
        assert image.size == (20, 40)
    rgba = tmp_path / "alpha.png"
    Image.new("RGBA", (5, 5)).save(rgba)
    with Image.open(app.prepare_image(rgba, tmp_path)) as image:
        assert image.mode == "RGB"




def app_document(tmp_path, text=APP_TEXT):
    """A document of the interface without kraken: one polygon per line, parsed with the rules only."""
    import app
    from PIL import Image
    scan = tmp_path / "scan.png"
    Image.new("RGB", (400, 30 * text.count("\n") + 30), "white").save(scan)
    lines = [{"text": t, "confidence": 0.5 if i == 4 else 0.95, "polygon": [[0, 30 * i], [400, 30 * i], [400, 30 * i + 25]],
              "baseline": []} for i, t in enumerate(text.rstrip("\n").split("\n"))]
    doc = {"folder": str(tmp_path), "image": str(scan), "name": "scan.png", "xml": "<PcGts/>", "lines": lines,
           "htr": text, "threshold": 0.8, "log": [], "history": [text], "position": 0}
    options = app.options_of(ner=False, french_ner=False, translate=False, spellcheck=False)
    return app.parse_text(doc, text, options), options


def test_app_panels_link_every_line(tmp_path):
    """Polygon, text line and record row of the same line carry the same class ln-n (hover links them)."""
    pytest.importorskip("gradio")
    import app
    doc, _ = app_document(tmp_path)
    segmentation = app.segmentation_html(doc, 1.0, 0.8)
    text = app.text_html(doc, 0.8)
    table = app.record_html(doc)
    assert 'class="md-poly ln-5"' in segmentation and 'md-line ln-5 low' in text      # line 5: 50 %
    assert "body:has(.ln-5:hover) .ln-5" in text
    # the father's job is read on line 19, the child's first name on lines 8 (Arabic) and 9 (Latin)
    assert doc["field_lines"]["PROFESSIONPERE"] == [18] and set(doc["field_lines"]["PRENOMAR"]) == {7}
    row = next(r for r in table.split("<tr") if "PROFESSIONPERE" in r)
    assert "ln-19" in row
    row = next(r for r in table.split("<tr") if "PRENOMAR" in r)
    assert "ln-8" in row and "ln-9" in row


def test_app_console(tmp_path):
    pytest.importorskip("gradio")
    import app
    doc, _ = app_document(tmp_path)
    console = app.console_html(doc, 0.8)
    assert console.index("error(s)") < console.index("[LOW CONFIDENCE] line 5")
    assert "low ln-5" in console


def test_app_history_and_files(tmp_path):
    pytest.importorskip("gradio")
    import app
    import zipfile
    doc, options = app_document(tmp_path)
    edited = app.set_text(doc, APP_TEXT.replace(JOB, "نجار", 1), options, "[edit] test")
    assert edited["record"]["PROFESSIONPERE"] == "نجار" and len(edited["history"]) == 2
    back = app.move_in_history(edited, -1, options)
    assert back["record"]["PROFESSIONPERE"] == JOB
    assert app.move_in_history(back, -1, options) is back               # nothing before the first version
    again = app.move_in_history(back, 1, options)
    assert again["record"]["PROFESSIONPERE"] == "نجار"
    names = set(zipfile.ZipFile(again["files"]["zip"]).namelist())
    assert names == {"scan.png", "scan.xml", "scan.txt", "scan.json"}      # all named after the scan
    assert Path(again["files"]["zip"]).name == "scan.zip"
    assert Path(again["files"]["segmentation"]).suffix == ".xml"


def test_app_link_off_when_the_lines_change(tmp_path):
    """A hand edit that adds a line: the text no longer matches the polygons, they are not linked."""
    pytest.importorskip("gradio")
    import app
    doc, options = app_document(tmp_path)
    doc = app.set_text(doc, APP_TEXT + "سطر زائد\n", options)
    assert not app.segmentation_linked(doc)
    assert "ln-1" not in app.segmentation_html(doc, 1.0, 0.8)
    assert "no longer linked" in app.console_html(doc, 0.8)


def test_app_views_without_a_document(tmp_path):
    pytest.importorskip("gradio")
    import app
    assert "Files" in app.image_html(None, 1.0)
    assert "Ready" in app.console_html(None, 0.8)
    doc, _ = app_document(tmp_path)
    deleted = dict(doc, image=str(tmp_path / "gone.png"))
    assert "deleted" in app.image_html(deleted, 1.0)


def test_app_builds():
    pytest.importorskip("gradio")
    import app
    assert app.build_app() is not None


# ---------------------------------------------------------------------------------------------------------
# parser: nationalities
# ---------------------------------------------------------------------------------------------------------


def nationalities(child="", father="جنسيته : مغربية.\n", mother="جنسيتها : مغربية\n"):
    import parser as record_parser
    record, notes = record_parser.parse_with_notes(
        NATIONALITY_DOCUMENT.format(child=child, father=father, mother=mother), None)
    return (record["NATIONALITE"], record["NATIONALITEPERE"], record["NATIONALITEMERE"]), notes


def test_mother_nationality_after_a_masculine_label():
    """The HTR wrote 'جنسيته' in the mother's block: it is still the mother's nationality."""
    values, notes = nationalities(mother="جنسيته : مغربية\n")
    assert values[2] == "مغربية" and any("masculine label" in n for n in notes)


def test_a_masculine_label_in_the_mother_zone_is_the_father_when_he_has_none():
    values, _ = nationalities(father="", mother="جنسيته : جزائرية\n")
    assert values[1:] == ("جزائرية", "")


def test_child_nationality_from_the_parents():
    """No nationality written for the child: both parents are Moroccan -> the child too."""
    values, notes = nationalities()
    assert values == ("مغربية", "مغربية", "مغربية") and any("same for the child" in n for n in notes)


def test_child_nationality_not_guessed_when_the_parents_differ():
    values, _ = nationalities(mother="جنسيتها : فرنسية\n")
    assert values == ("", "مغربية", "فرنسية")


def test_written_child_nationality_is_kept():
    values, _ = nationalities(child="جنسيته : فرنسية\n")
    assert values[0] == "فرنسية"

# ---------------------------------------------------------------------------------------------------------
# languages of the interface
# ---------------------------------------------------------------------------------------------------------
def test_every_text_has_three_languages():
    import ui_text
    for key, versions in ui_text.TEXT.items():
        assert len(versions) == 3 and all(versions), key
        # the same {values} in every language
        import string
        fields = [sorted(f for _, f, _, _ in string.Formatter().parse(v) if f) for v in versions]
        assert fields[0] == fields[1] == fields[2], key
    assert set(ui_text.HELP) == {"en", "fr", "ar"}


def test_tr_writes_the_three_versions():
    import ui_text
    html = ui_text.tr("c_low", number=3, score="72%", text="<b>عام</b>")
    assert 'class="l-en"' in html and 'class="l-fr"' in html and 'class="l-ar"' in html
    assert "[LOW CONFIDENCE] line 3: 72%" in html and "[CONFIANCE FAIBLE] ligne 3" in html and "السطر 3" in html
    assert "&lt;b&gt;" in html                                        # values are escaped
    assert ui_text.text("save", "fr") == "🖫 Enregistrer" and ui_text.text("save", "de") == "🖫 Save"


def test_notification_script_has_every_language():
    import ui_text
    js = ui_text.notification_js()
    assert "Transcription terminée" in js and "انتهى الاستخراج" in js and "document.hidden" in js


def test_app_console_in_three_languages(tmp_path):
    pytest.importorskip("gradio")
    import app
    doc, _ = app_document(tmp_path)
    console = app.console_html(doc, 0.8)
    assert "error(s)" in console and "erreur(s)" in console and "خطأ" in console
    assert "Champ" in app.record_html(doc) and "الحقل" in app.record_html(doc)


# ---------------------------------------------------------------------------------------------------------
# classificator: the French NER model runs locally, on the Latin lines only
# ---------------------------------------------------------------------------------------------------------
import classificator  # noqa: E402

NER_TEXT = APP_TEXT     # samples/2.txt: two Latin lines, "PRENOM AYOUB" and "NOM SBAI"


def fake_ner(text):
    """A pipeline stand-in: every upper-case word that is not a label is a PERSON."""
    return [{"word": m.group(), "entity_group": "PERSON", "start": m.start(), "end": m.end(), "score": 0.98}
            for m in __import__("re").finditer(r"\b(?!PRENOM|NOM)[A-Z]{2,}\b", text)]


def test_latin_lines_offsets():
    lines = classificator.latin_lines(NER_TEXT)
    assert [line for _, line in lines] == ["PRENOM AYOUB", "NOM SBAI"]
    for offset, line in lines:
        assert NER_TEXT[offset:offset + len(line)] == line


def test_french_ner_local(monkeypatch):
    seen = []
    monkeypatch.setattr(classificator, "get_french_pipeline", lambda: lambda t: seen.append(t) or fake_ner(t))
    monkeypatch.setattr(classificator, "classify_french_space", lambda t: pytest.fail("Space called"))
    found = classificator.classify_french(NER_TEXT)
    assert seen == ["PRENOM AYOUB", "NOM SBAI"]          # the Arabic lines are never given to it
    assert [e["entity"] for e in found] == ["AYOUB", "SBAI"]
    for e in found:                                          # offsets in the whole text, for the parser zones
        assert NER_TEXT[e["start"]:e["end"]] == e["entity"]


def test_french_ner_missing_model_is_reported_not_sent(monkeypatch):
    def missing():
        raise classificator.ModelUnavailable("not downloaded")
    monkeypatch.delenv("MADOC_FRENCH_NER_SPACE", raising=False)
    monkeypatch.setattr(classificator, "get_french_pipeline", missing)
    monkeypatch.setattr(classificator, "classify_french_space", lambda t: pytest.fail("Space called"))
    result = classificator.classify(NER_TEXT, use_arabic=False)
    assert result["french_ner"] == [] and "not downloaded" in result["errors"][0]
    assert result["models"]["fr"] == "TypicaAI/magbert-ner"


def test_french_ner_space_only_when_allowed(monkeypatch):
    def missing():
        raise classificator.ModelUnavailable("not downloaded")
    monkeypatch.setenv("MADOC_FRENCH_NER_SPACE", "1")
    monkeypatch.setattr(classificator, "get_french_pipeline", missing)
    monkeypatch.setattr(classificator, "classify_french_space", lambda t: [{"entity": "X", "label": "PERSON"}])
    assert classificator.classify_french(NER_TEXT) == [{"entity": "X", "label": "PERSON"}]


def test_french_ner_skipped_without_latin(monkeypatch):
    monkeypatch.setattr(classificator, "get_french_pipeline", lambda: pytest.fail("model loaded"))
    assert classificator.classify_french("في يوم ثالث\nمن والده\n") == []


def test_french_ner_needs_a_sentence(monkeypatch):
    """A model that only sees "Hicham" inside a sentence: the entity still points at "HICHAM" in the text."""
    import re

    def sentence_only(text):
        if not text.startswith("Son "):
            return []
        return [{"word": m.group(), "entity_group": "PER", "start": m.start(), "end": m.end(), "score": 0.9}
                for m in re.finditer(r"\b(?!Son\b)[A-Z][a-z]+\b", text)]

    monkeypatch.setattr(classificator, "get_french_pipeline", lambda: sentence_only)
    found = classificator.classify_french(NER_TEXT)
    assert [e["entity"] for e in found] == ["AYOUB", "SBAI"]
    for e in found:
        assert NER_TEXT[e["start"]:e["end"]] == e["entity"]


def test_french_ner_drops_labels(monkeypatch):
    def everything(text):
        return [{"word": m.group(), "entity_group": "PER", "start": m.start(), "end": m.end(), "score": 0.5}
                for m in __import__("re").finditer(r"\S+", text)]
    monkeypatch.setattr(classificator, "get_french_pipeline", lambda: everything)
    assert [e["entity"] for e in classificator.classify_french(NER_TEXT)] == ["AYOUB", "SBAI"]


# ---------------------------------------------------------------------------------------------------------
# found in the review of the code
# ---------------------------------------------------------------------------------------------------------
def test_a_misread_number_word_is_not_ignored():
    """"ألفين و ثلاثن" used to be read as 2000 (the unknown word skipped): a wrong date that looks valid."""
    record = converter.convert(dict(common.TEMPLATE, DATENAISSFR="خامس و عشري دجنبر سنة ألفين و ثلاثن"))
    assert not str(record["DATENAISSFR"]).startswith("2000")
    record = converter.convert(dict(common.TEMPLATE, DATENAISSFR="خامس و عشري دجنبر سنة ألفين و ثلاثة"))
    assert record["DATENAISSFR"] == "2003-12-25"


def test_app_files_of_a_run_live_while_it_is_used(tmp_path, monkeypatch):
    pytest.importorskip("gradio")
    import app
    from PIL import Image
    monkeypatch.setattr(app, "RUNS_DIR", tmp_path / "runs")
    folder = app.new_run_folder()
    scan = tmp_path / "upload.jpg"
    Image.new("RGB", (40, 20)).save(scan)
    doc = {"folder": str(folder), "image": app.prepare_image(scan, folder)}
    scan.unlink()                                                      # gradio deletes its copy of the upload
    old = time.time() - 2 * app.KEEP_SECONDS
    os.utime(folder, (old, old))
    app.keep_alive(doc)                                                # used again: kept
    other = app.new_run_folder()                                       # another run cleans the old folders
    assert folder.is_dir() and other.is_dir()
    os.utime(folder, (old, old))
    app.new_run_folder()                                               # not used for too long: deleted
    assert not folder.exists()
    with pytest.raises(Exception, match="import it again"):
        app.keep_alive(doc)


def test_the_models_are_not_asked_twice_for_the_same_text(monkeypatch):
    import classificator
    calls = []
    monkeypatch.setattr(classificator, "classify_arabic", lambda text: calls.append(text) or [])
    monkeypatch.setattr(classificator, "classify_french", lambda text: [])
    classificator.classify("نص")
    classificator.classify("نص")
    assert calls == ["نص"]


@pytest.mark.parametrize("old, new, field", [
    ("على الساعة : السادسة عشرة", "على الساعة :", "HEURENAISS"),
    ("و الدقيقة : الثلاثين.", "و الدقيقة :", "MINNAISS"),
    ("الإسم الشخصي : حفصة", "الإسم الشخصي :", "PRENOMAR"),
    ("الإسم العائلي : الكتاني", "الإسم العائلي :", "NOMAR"),
    ("ولد ب: محج المستشفى مدينة خريبڭة.", "ولد ب:", "LIEUNAISS"),
    ("جنسيته : مغربية.", "جنسيته :", "NATIONALITEPERE"),
    ("حرفته : تقني", "حرفته :", "PROFESSIONPERE"),
    ("حرفتها : ربة البيت.", "حرفتها :", "PROFESSIONMERE"),
    ("من طرفنا نحن فؤاد السعدي موظف جماعي", "من طرفنا نحن", "EMPLOYECREANT"),
])
def test_an_empty_value_stays_empty(old, new, field):
    """A label with nothing after it: the field is empty, the next label is not read as its value, and the
    other fields are read as before (the child's nationality comes from both parents: it may go too)."""
    import parser as record_parser
    text = (ROOT / "samples" / "3.txt").read_text(encoding="utf-8")
    assert old in text
    full = record_parser.parse(text, None)
    empty = record_parser.parse(text.replace(old, new), None)
    assert empty[field] == ""
    assert {k for k in full if full[k] != empty[k]} <= {field, "NATIONALITE"}


def test_app_a_hand_edit_checks_the_spelling_again(tmp_path, monkeypatch):
    """After a hand edit the spelling is checked again on the edited text: the words typed by hand are only
    proposed (orange), a correction refused earlier in the review stays refused."""
    pytest.importorskip("gradio")
    import app
    import spelling_checker as sc
    from PIL import Image
    text = (ROOT / "samples" / "6.txt").read_text(encoding="utf-8")
    lines = [{"text": l, "confidence": 0.9, "polygon": [], "baseline": []} for l in text.rstrip("\n").split("\n")]
    monkeypatch.setattr(app, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(app, "segment_image", lambda *a, **k: "<xml/>")
    monkeypatch.setattr(app, "recognize_lines", lambda *a, **k: [dict(l) for l in lines])
    monkeypatch.setattr(sc, "OLLAMA_URL", "http://127.0.0.1:9")            # rules only
    scan = tmp_path / "scan.jpg"
    Image.new("RGB", (40, 20)).save(scan)
    options = app.options_of(False, False, False, True)
    doc, spelling = app.start_document(str(scan), "scan.jpg", options, 0.8, "cpu", "m1", "m2")
    fix = next(i for i, c in enumerate(spelling["changes"]) if c["old"] == ["البيتت"])
    assert "البيتت" not in doc["history"][-1]                              # applied by the rules
    spelling = {**spelling, "accepted": [i for i in spelling["accepted"] if i != fix]}   # the person refuses it
    doc = app.set_text(doc, app.apply_changes(spelling["text"], spelling["changes"], set(spelling["accepted"])),
                       options, "c_spelling")
    assert "البيتت" in doc["history"][-1]
    # a hand edit on another line, with a typo in a word of the form
    current = doc["history"][-1]
    line = next(l for l in current.split("\n") if l.startswith("موافق"))
    edited = current.replace(line, line.replace("موافق", "موفق", 1), 1)
    doc, spelling = app.edit_text(doc, edited, options, spelling=spelling)
    final = doc["history"][doc["position"]]
    assert "البيتت" in final                                               # still refused
    assert "موفق" in final                                                 # typed by hand: not changed by itself
    typed = next(c for c in spelling["changes"] if c["old"] == ["موفق"])
    assert typed["check"] and not typed["accepted"] and typed["new"] == ["موافق"]
    refused = next(c for c in spelling["changes"] if c["old"] == ["البيتت"])
    assert not refused["accepted"] and "refused" in refused["reason"]
    assert any(entry[1] == "c_respell" for entry in doc["log"])

def test_app_logins():
    pytest.importorskip("gradio")
    import app
    assert app.logins("") == [] and app.logins(None) == []
    assert app.logins("ana:12:34, sara:abcd") == [("ana", "12:34"), ("sara", "abcd")]
    with pytest.raises(SystemExit):
        app.logins("ana")

def test_app_sample_images(tmp_path, monkeypatch):
    pytest.importorskip("gradio")
    import app
    folder = tmp_path / "samples"
    folder.mkdir()
    for name in ("2.jpg", "1.png", "1.txt", "notes.md"):
        (folder / name).write_bytes(b"x")
    monkeypatch.setattr(app, "SAMPLES_DIR", folder)
    assert [Path(p).name for p in app.sample_images()] == ["1.png", "2.jpg"]   # the scans only, in order
    monkeypatch.setattr(app, "SAMPLES_DIR", tmp_path / "none")
    assert app.sample_images() == []                                         # no folder: no Samples button



def test_app_files_are_named_after_the_scan(tmp_path):
    """"C:/Users/.../03010108_1.jpg" -> 03010108_1.xml / .txt / .json, and the PageXML names its image by the
    file name alone (not the path of the server)."""
    pytest.importorskip("gradio")
    import app
    import zipfile
    assert app.file_name("C:\\Users\\me\\Desktop\\03010108_1.jpg") == "03010108_1.jpg"
    assert app.file_name("/tmp/gradio/abc/1.jpg") == "1.jpg" and app.file_name('a?b.jpg') == "a_b.jpg"
    xml = '<PcGts><Page imageFilename="/tmp/madoc_runs/run_x/scan.jpg" imageWidth="10"/></PcGts>'
    assert app.page_xml(xml, "1.jpg") == '<PcGts><Page imageFilename="1.jpg" imageWidth="10"/></PcGts>'
    doc, options = app_document(tmp_path)
    original = tmp_path / "original.jpg"
    original.write_bytes(b"the scan as uploaded")
    doc = dict(doc, name="C:/Users/me/1.jpg", original=str(original),
               xml='<PcGts><Page imageFilename="C:/Users/me/1.jpg"/></PcGts>')
    doc = app.parse_text(doc, doc["htr"], options)
    archive = zipfile.ZipFile(doc["files"]["zip"])
    assert sorted(archive.namelist()) == ["1.jpg", "1.json", "1.txt", "1.xml"]
    assert archive.read("1.jpg") == b"the scan as uploaded"                       # the original, unchanged
    assert 'imageFilename="1.jpg"' in archive.read("1.xml").decode("utf-8")
    assert Path(doc["files"]["zip"]).name == "1.zip"

# --- lines read out of order -------------------------------------------------------------------------------
def test_parser_reads_lines_out_of_order():
    import parser as record_parser
    from documents import APP_TEXT, OUT_OF_ORDER_DOCUMENT
    record, notes = record_parser.parse_with_notes(OUT_OF_ORDER_DOCUMENT)
    assert record == record_parser.parse(APP_TEXT)
    assert record["PRENOMMEREAR"] and record["PROFESSIONMERE"] == "ربة بيت"
    assert "حرفت" not in record["ADRESSEPARENT"] and record["MINNAISS"] == "الصفر"
    assert any("not in the order of the form" in note for note in notes)


def test_parser_same_record_whatever_two_lines_are_swapped():
    import parser as record_parser
    from common import ar_key
    from documents import moved, sample
    for number in range(1, 7):
        text = sample(number)
        expected = record_parser.parse(text)
        for line in range(1, text.count("\n")):
            record = record_parser.parse(moved(text, line, line + 1))
            differ = [f for f in expected if f != "TEXTEDECLA" and ar_key(record[f]) != ar_key(expected[f])]
            assert not differ, f"samples/{number}.txt, lines {line} and {line + 1} swapped: {differ}"


def test_parser_dates_and_nationalities_find_their_parent():
    import parser as record_parser
    from documents import moved, sample
    text = sample(1)
    expected = record_parser.parse(text)
    for line, to in ((18, 20), (17, 20), (20, 17), (11, 14)):   # a date / the mother's line / a nationality
        record = record_parser.parse(moved(text, line, to))
        for field in ("DATENAISSPEREAR", "DATENAISSPEREFR", "DATENAISSMEREAR", "DATENAISSMEREFR",
                      "NATIONALITE", "NATIONALITEPERE", "NATIONALITEMERE"):
            assert record[field] == expected[field], (line, to, field)


def test_parser_text_in_order_is_not_moved():
    import parser as record_parser
    from documents import APP_TEXT
    text, count, new_line = record_parser.restore_order(APP_TEXT)
    assert text == APP_TEXT and count == 0 and all(old == new for old, new in new_line.items())


def test_parser_ner_offsets_move_with_their_lines():
    import parser as record_parser
    from documents import OUT_OF_ORDER_DOCUMENT
    raw = OUT_OF_ORDER_DOCUMENT
    ordered, _, new_line = record_parser.restore_order(raw)
    words = ["زينب", "الصفر", "ربة بيت", "خريبڭة"]
    ner = {"arabic_ner": [{"entity": w, "label": "PERSON", "start": raw.index(w), "end": raw.index(w) + len(w)}
                          for w in words]}
    for item in record_parser.move_ner(ner, raw, new_line)["arabic_ner"]:
        assert ordered[item["start"]:item["end"]] == item["entity"]

# --- the record edited by hand (JSON tab) ------------------------------------------------------------------
def test_app_record_from_json():
    pytest.importorskip("gradio")
    import json
    import app
    record = {key: "" for key in app.TEMPLATE}
    assert list(app.record_from_json(json.dumps(record))) == list(app.TEMPLATE)
    for text, message in (("{", "line 1"), ("[]", "object"),
                          (json.dumps({**record, "AGE": 3}), "unknown field(s): AGE"),
                          (json.dumps({k: v for k, v in record.items() if k != "NOM"}), "missing field(s): NOM"),
                          (json.dumps({**record, "NOM": ["A"]}), "NOM")):
        with pytest.raises(ValueError, match=message.replace("(", r"\(").replace(")", r"\)")):
            app.record_from_json(text)


def test_app_record_edit_is_kept_until_the_text_changes_the_field(tmp_path):
    pytest.importorskip("gradio")
    import json
    import app
    doc, options = app_document(tmp_path)
    before = doc["record"]
    edited = {**before, "PROFESSIONPERE": "نجار", "PRENOM": "AYYOUB"}
    doc, changed = app.edit_record(doc, before, edited, options)
    assert changed == ["PRENOM", "PROFESSIONPERE"] and doc["record"]["PRENOM"] == "AYYOUB"
    assert json.loads(Path(doc["files"]["json"]).read_text(encoding="utf-8"))["PRENOM"] == "AYYOUB"
    assert "[record] edited by hand: PRENOM, PROFESSIONPERE" in app.console_html(doc, 0.8)
    # the text is corrected elsewhere: the hand edits stay
    doc = app.set_text(doc, APP_TEXT.replace("رسم ولادة رقم : 80", "رسم ولادة رقم : 81"), options)
    assert doc["record"]["NUMACTE"] == "81" and doc["record"]["PRENOM"] == "AYYOUB"
    assert doc["record"]["PROFESSIONPERE"] == "نجار"
    # the text now reads another job: the text wins for that field only
    doc = app.set_text(doc, APP_TEXT.replace(JOB, "حداد", 1), options)
    assert doc["record"]["PROFESSIONPERE"] == "حداد" and doc["record"]["PRENOM"] == "AYYOUB"
    assert set(doc["record_edits"]) == {"PRENOM"}
    # written back to the value of the text: no longer an edit
    doc, _ = app.edit_record(doc, doc["record"], {**doc["record"], "PRENOM": doc["auto_record"]["PRENOM"]},
                             options)
    assert doc["record_edits"] == {}


def test_app_record_edit_only_takes_what_was_changed_in_the_editor(tmp_path):
    """The record was parsed again while the editor was open: the other fields keep the new values."""
    pytest.importorskip("gradio")
    import app
    doc, options = app_document(tmp_path)
    before = doc["record"]
    doc = app.set_text(doc, APP_TEXT.replace("رسم ولادة رقم : 80", "رسم ولادة رقم : 81"), options)
    doc, changed = app.edit_record(doc, before, {**before, "PRENOM": "AYYOUB"}, options)
    assert changed == ["PRENOM"] and doc["record"]["NUMACTE"] == "81"


def test_app_translate_again_keeps_the_french_written_by_hand(tmp_path, monkeypatch):
    pytest.importorskip("gradio")
    import app
    import translator
    monkeypatch.setattr(translator, "unavailable_reason", lambda: None)
    monkeypatch.setattr(translator, "translate_batch",
                        lambda items: [f"Latin{i}" for i, _ in enumerate(items)])
    doc, options = app_document(tmp_path)
    doc, _ = app.edit_record(doc, doc["record"], {**doc["record"], "PRENOMMEREAR": "سلمى الحجوجي",
                                                  "LIEUNAISSFR": "Khouribga (à la main)"}, options)
    doc = app.translate_document(doc, options)
    record = doc["record"]
    assert record["LIEUNAISSFR"] == "Khouribga (à la main)"                  # written by hand: kept
    assert record["PRENOMMEREFR"].startswith("Salma Latin")                  # from the Arabic edited by hand
    assert record["NATIONALITEFR"] == "MAROCAINE"                              # from the dictionary again
    assert "[translation] French fields translated again" in app.console_html(doc, 0.8)