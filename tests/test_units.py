"""Unit tests of the helpers around the chain: file reading, kraken calls, translator output, app files.

No model, no kraken, no network: kraken is replaced by a fake command and Ollama by a
small local HTTP server.

    python -m pytest tests/test_units.py
"""

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
    record = dict(common.TEMPLATE, PRENOMPEREAR="محمد", PROFESSIONPERE="حداد التقليدي")
    out = translator.translate(record)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Translation model unavailable" in captured.err
    assert out["PROFESSIONPEREFR"] == ""          # an untranslated job stays empty, never Arabic
    assert out["PRENOMPEREFR"] == "محمد"          # kept, the validator reports it as untranslated


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
        answer = {"name": "Fils de Mohamed", "address": "Khouribga", "job": "forgeron",
                  "employee": "Lahcen Chmiti"}
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
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(spelling_checker, "OLLAMA_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setattr(translator, "MODEL", "qwen2.5:7b")
    yield Fake
    server.shutdown()


def test_translator_local_model(translation_ollama):
    record = dict(common.TEMPLATE, PRENOMPEREAR="بن محمد", PROFESSIONPERE="حداد التقليدي", LIEUNAISS="خريبكة")
    out = translator.translate(record)
    assert out["PRENOMPEREFR"] == "Fils de Mohamed"
    assert out["PROFESSIONPEREFR"] == "FORGERON"           # post-processed like jobs.json
    assert out["LIEUNAISSFR"] == "Khouribga"
    body = translation_ollama.requests[0]
    assert body["model"] == "qwen2.5:7b" and body["format"] == "json"
    assert translator.is_available()


def test_translator_model_not_pulled(translation_ollama, capsys):
    translation_ollama.models = [{"name": "llama3:8b", "model": "llama3:8b"}]
    out = translator.translate(dict(common.TEMPLATE, PRENOMPEREAR="محمد"))
    assert out["PRENOMPEREFR"] == "محمد"
    assert "ollama pull qwen2.5:7b" in capsys.readouterr().err
    assert translation_ollama.requests == []


def test_translator_rejects_arabic_answers(translation_ollama):
    translation_ollama.answer = dict(translation_ollama.answer, name="محمد")
    out = translator.translate(dict(common.TEMPLATE, PRENOMPEREAR="محمد"))
    assert out["PRENOMPEREFR"] == "محمد"                    # the original, never the model's Arabic


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
    item = {"type": "employee", "text": "لحسن شميطي"}
    assert translator.postprocess_output(item, "Lahcen Chmiti").endswith(common.EMPLOYEE_TITLE)


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
    assert app.prepare_image(plain, tmp_path) == str(plain)          # nothing to fix: used as it is
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


APP_TEXT = (
    "رسم ولادة رقم: 42\n"
    "في يوم ثالث شوال عام ألف و أربعمائة و أربعة و عشرين هجرية\n"
    "موافق ثامن عشر دجنبر سنة الفين و ثلاثة ميلادية.\n"
    "PRENOM BILAL\n"
    "الإسم الشخصي : بلال\n"
    "ذكر\n"
    "من والده عبد الجبار قاسم بن محمد\n"
    "حرفته : فلاح\n"
    "و والدته عتيقة ممير بنت عبد العزيز.\n"
    "الساكنان بحي الأمل الرقم: 494 خريبڭة.\n"
)


def app_document(tmp_path, text=APP_TEXT):
    """A document of the interface without kraken: one polygon per line, parsed with the rules only."""
    import app
    from PIL import Image
    scan = tmp_path / "scan.png"
    Image.new("RGB", (400, 30 * 12), "white").save(scan)
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
    # the father's job is read on line 8, the child's first name on lines 4 (Latin) and 5 (Arabic)
    assert doc["field_lines"]["PROFESSIONPERE"] == [7] and set(doc["field_lines"]["PRENOMAR"]) == {4}
    row = next(r for r in table.split("<tr") if "PROFESSIONPERE" in r)
    assert "ln-8" in row
    row = next(r for r in table.split("<tr") if "PRENOMAR" in r)
    assert "ln-4" in row and "ln-5" in row


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
    edited = app.set_text(doc, APP_TEXT.replace("فلاح", "نجار"), options, "[edit] test")
    assert edited["record"]["PROFESSIONPERE"] == "نجار" and len(edited["history"]) == 2
    back = app.move_in_history(edited, -1, options)
    assert back["record"]["PROFESSIONPERE"] == "فلاح"
    assert app.move_in_history(back, -1, options) is back               # nothing before the first version
    again = app.move_in_history(back, 1, options)
    assert again["record"]["PROFESSIONPERE"] == "نجار"
    names = set(zipfile.ZipFile(again["files"]["zip"]).namelist())
    assert names == {"scan.png", "segmentation.png", "segmentation.xml", "transcription.txt",
                     "transcription_htr.txt", "record.json", "report.json", "report.txt"}


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
NATIONALITY_DOCUMENT = (
    "في يوم ثالث شوال\n"
    "الإسم الشخصي : بلال\n"
    "ذكر\n"
    "{child}"
    "من والده عبد الجبار قاسم بن محمد\n"
    "{father}"
    "و والدته عتيقة ممير بنت عبد العزيز.\n"
    "المولودة ب جماعة خريبڭة الحضرية.\n"
    "{mother}"
    "الساكنان بحي الأمل الرقم: 494 خريبڭة.\n"
)


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
    assert ui_text.text("save", "fr") == "💾 Enregistrer" and ui_text.text("save", "de") == "💾 Save"


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

NER_TEXT = "في يوم ثالث\nالاسم الشخصي زكرياء\nPRENOM ZAKARIYA\nNOM KABLI\nمن والده\n"


def fake_ner(text):
    """A pipeline stand-in: every upper-case word that is not a label is a PERSON."""
    return [{"word": m.group(), "entity_group": "PERSON", "start": m.start(), "end": m.end(), "score": 0.98}
            for m in __import__("re").finditer(r"\b(?!PRENOM|NOM)[A-Z]{2,}\b", text)]


def test_latin_lines_offsets():
    lines = classificator.latin_lines(NER_TEXT)
    assert [line for _, line in lines] == ["PRENOM ZAKARIYA", "NOM KABLI"]
    for offset, line in lines:
        assert NER_TEXT[offset:offset + len(line)] == line


def test_french_ner_local(monkeypatch):
    seen = []
    monkeypatch.setattr(classificator, "get_french_pipeline", lambda: lambda t: seen.append(t) or fake_ner(t))
    monkeypatch.setattr(classificator, "classify_french_space", lambda t: pytest.fail("Space called"))
    found = classificator.classify_french(NER_TEXT)
    assert seen == ["PRENOM ZAKARIYA", "NOM KABLI"]          # the Arabic lines are never given to it
    assert [e["entity"] for e in found] == ["ZAKARIYA", "KABLI"]
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
    assert [e["entity"] for e in found] == ["ZAKARIYA", "KABLI"]
    for e in found:
        assert NER_TEXT[e["start"]:e["end"]] == e["entity"]


def test_french_ner_drops_labels(monkeypatch):
    def everything(text):
        return [{"word": m.group(), "entity_group": "PER", "start": m.start(), "end": m.end(), "score": 0.5}
                for m in __import__("re").finditer(r"\S+", text)]
    monkeypatch.setattr(classificator, "get_french_pipeline", lambda: everything)
    assert [e["entity"] for e in classificator.classify_french(NER_TEXT)] == ["ZAKARIYA", "KABLI"]