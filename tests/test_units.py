"""Unit tests of the helpers around the chain: file reading, kraken calls, translator output, app files.

No model, no kraken, no network: kraken is replaced by a fake command and Groq is not called.

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
    monkeypatch.setattr(translator, "get_client", lambda: None)
    record = dict(common.TEMPLATE, PRENOMPEREAR="محمد", PROFESSIONPERE="حداد التقليدي")
    out = translator.translate(record)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Groq client unavailable" in captured.err
    assert out["PROFESSIONPEREFR"] == ""          # an untranslated job stays empty, never Arabic


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


def test_app_view(tmp_path):
    pytest.importorskip("gradio")
    import app
    from PIL import Image
    scan = tmp_path / "scan.png"
    Image.new("RGB", (100, 50), "white").save(scan)
    lines = [{"text": "رسم <ولادة>", "confidence": 0.5, "polygon": [[0, 0], [10, 0], [10, 5]], "baseline": []}]
    html = app.render_view(str(scan), lines, 0.8)
    assert "bv-poly-1" in html and "&lt;ولادة&gt;" in html and "1 under 80%" in html
    assert app.rerender(str(tmp_path / "deleted.png"), lines, 0.8).startswith("<p>The scan")