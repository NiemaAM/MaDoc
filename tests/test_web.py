"""The web application (src/server.py, src/madoc_web): its routes with kraken and the local model replaced.

kraken is never loaded: the segmentation gives fixed lines and the recognition gives the text of a sample."""
import base64
import io
import json
import zipfile

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multipart")

from pathlib import Path  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAGE_TWO = (ROOT / "samples" / "6.txt").read_text(encoding="utf-8").rstrip("\n").split("\n")
PAGE_ONE = ["الصفحة رقم : 7", "البيانات الهامشية", PAGE_TWO[0], "الإسم الشخصي : حمزة", "PRENOM HAMZA",
            "الإسم العائلي : الساحلي", "NOM SAHLI", "تزوج بتاريخ فاتح يناير", "Fils de NABIL SAHLI et de RIM CHAOUI"]


def scan(size=(400, 800), fmt="JPEG"):
    data = io.BytesIO()
    Image.new("RGB", size, "white").save(data, fmt)
    return data.getvalue()


def boxes(count):
    return [{"id": f"k{i}", "text": "", "confidence": None,
             "polygon": [(10, 10 + 20 * i), (390, 10 + 20 * i), (390, 26 + 20 * i), (10, 26 + 20 * i)],
             "baseline": [(10, 24 + 20 * i), (390, 24 + 20 * i)]} for i in range(count)]


@pytest.fixture
def web(tmp_path, monkeypatch):
    """(client, calls): the application on an empty data folder; calls counts what kraken was asked."""
    import spelling_checker as sc
    from madoc_web import api, config, jobs, monitor, process
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    config.reset_cache()
    monkeypatch.setattr(sc, "OLLAMA_URL", "http://127.0.0.1:9")            # no local model: rules only
    monkeypatch.setattr(monitor, "_gpu_tool", [None])
    calls = {"segment": 0, "recognize": [], "texts": {"1": PAGE_ONE, "2": PAGE_TWO}}

    def fake_segment(image, model, device):
        calls["segment"] += 1
        return "page" + ("1" if "page1" in Path(image).name else "2")

    def fake_recognize(xml, image, model, device):
        page = "1" if "page1" in Path(image).name else "2"
        ids = __import__("re").findall(r'<TextLine id="([^"]+)"', xml)
        calls["recognize"].append((page, len(ids)))
        texts = calls["texts"][page]
        return [{"id": identifier, "text": texts[i % len(texts)], "confidence": 0.9, "polygon": [], "baseline": []}
                for i, identifier in enumerate(ids)]

    monkeypatch.setattr(process, "segment_image", fake_segment)
    monkeypatch.setattr(process, "parse_pagexml", lambda xml: boxes(len(calls["texts"][xml[-1]])))
    monkeypatch.setattr(process, "recognize_lines", fake_recognize)
    with TestClient(api.create_app()) as client:
        config.save({"use_ner": False, "use_french_ner": False, "use_translation": False})
        yield client, calls
        jobs.wait()
    config.reset_cache()


def run(client, doc, **body):
    from madoc_web import jobs
    answer = client.post(f"/api/docs/{doc}/run", json=body)
    assert answer.status_code == 200, answer.text
    assert jobs.wait()
    result = client.get(f"/api/docs/{doc}").json()
    assert (result.get("job") or {}).get("state") in (None, "done"), result["job"]
    return result


def new_document(client, pages=("2",)):
    doc = client.post("/api/docs").json()["id"]
    for page in pages:
        answer = client.put(f"/api/docs/{doc}/pages/{page}", files={"file": (f"acte {page}.jpg", scan(), "image/jpeg")})
        assert answer.status_code == 200, answer.text
    return doc


def test_web_runs_the_two_pages_and_fills_the_marginal_notes(web):
    client, calls = web
    doc = new_document(client, ("2", "1"))
    result = run(client, doc, kind="auto")
    assert result["job"]["steps"] == ["segment 2", "recognize 2", "segment 1", "recognize 1", "extract"]
    assert [line["text"] for line in result["pages"]["2"]["lines"]] == PAGE_TWO
    record = result["record"]
    assert record["NUMACTE"] and record["TEXTEMENTIONS"] == "تزوج بتاريخ فاتح يناير"
    assert record["TEXTEMENTIONSFR"] == "Fils de NABIL SAHLI et de RIM CHAOUI"
    assert record["PRENOMPEREFR"] == "NABIL SAHLI" and record["PRENOMMEREFR"] == "RIM CHAOUI"
    assert [(c["field"], c["same"]) for c in result["comparisons"]] == [
        ("NUMACTE", True), ("PRENOMAR", True), ("PRENOM", True), ("NOMAR", True), ("NOM", True)]
    assert result["field_lines"]["TEXTEMENTIONS"]["page"] == "1"
    assert "auto_record" not in result and not result["stale"]
    # the downloads
    assert json.loads(client.get(f"/api/docs/{doc}/download/json").content)["TEXTEMENTIONSFR"]
    archive = zipfile.ZipFile(io.BytesIO(client.get(f"/api/docs/{doc}/download/zip").content))
    assert sorted(Path(name).suffix for name in archive.namelist()) == [".jpg", ".jpg", ".json", ".txt", ".xml", ".xml"]
    assert "<Unicode>" in client.get(f"/api/docs/{doc}/download/xml2").text


def test_web_a_difference_between_the_pages_is_a_warning(web):
    client, calls = web
    calls["texts"]["1"] = [line.replace("حمزة", "حمزي") if "الشخصي" in line else line for line in PAGE_ONE]
    result = run(client, new_document(client, ("2", "1")), kind="auto")
    differing = [c for c in result["comparisons"] if c["same"] is False]
    assert [c["field"] for c in differing] == ["PRENOMAR"] and result["record"]["PRENOMAR"] == "حمزة"
    assert any("page 1 and page 2 differ" in issue["message"] for issue in result["report"]["issues"])


def test_web_lines_can_be_moved_deleted_and_drawn_then_read_again(web):
    client, calls = web
    doc = new_document(client)
    result = run(client, doc, kind="auto")
    lines = result["pages"]["2"]["lines"]
    kept = [{"id": line["id"], "polygon": line["polygon"], "baseline": line["baseline"]} for line in lines[1:]]
    kept[0]["polygon"][0] = [12, kept[0]["polygon"][0][1]]                            # moved
    kept.append({"polygon": [[10, 760], [390, 760], [390, 780], [10, 780]], "baseline": [[10, 778], [390, 778]]})
    answer = client.put(f"/api/docs/{doc}/pages/2/lines", json={"lines": kept})
    assert answer.status_code == 200, answer.text
    after = answer.json()["pages"]["2"]["lines"]
    assert len(after) == len(lines) and [line["dirty"] for line in after].count(True) == 2
    assert after[0]["text"] == lines[1]["text"] and after[-1]["text"] == "" and answer.json()["stale"]
    calls["recognize"].clear()
    result = run(client, doc, kind="recognize", pages=["2"], only_dirty=True)
    assert calls["recognize"] == [("2", 2)]                                           # only the two changed lines
    assert not any(line["dirty"] for line in result["pages"]["2"]["lines"])
    # a polygon outside the scan is brought back inside it; a broken one is refused
    wild = [{"polygon": [[-50, -5], [9999, 0], [20, 9999]], "baseline": [[0, 5], [500, 5]]}]
    moved = client.put(f"/api/docs/{doc}/pages/2/lines", json={"lines": wild}).json()["pages"]["2"]["lines"][0]
    assert moved["polygon"] == [[0, 0], [400, 0], [20, 800]] and moved["baseline"] == [[0, 5], [400, 5]]
    assert client.put(f"/api/docs/{doc}/pages/2/lines", json={"lines": [{"polygon": [[1, 1]], "baseline": []}]}
                      ).status_code == 400


def test_web_text_edit_spelling_on_demand_and_record_edit(web):
    client, calls = web
    doc = new_document(client)
    result = run(client, doc, kind="auto")
    assert result["spelling"] is None                                                # not checked by the run
    line = next(line for line in result["pages"]["2"]["lines"] if "البيتت" in line["text"])
    result = run(client, doc, kind="spelling")
    fix = next(i for i, c in enumerate(result["spelling"]["changes"]) if c["old"] == ["البيتت"])
    assert result["spelling"]["changes"][fix]["line_id"] == line["id"] and result["spelling"]["changes"][fix]["accepted"]
    result = client.post(f"/api/docs/{doc}/spelling/apply", json={"accepted": [fix]}).json()
    assert "البيتت" not in "".join(l["text"] for l in result["pages"]["2"]["lines"]) and result["stale"]
    # a hand edit of a line, then the record again
    answer = client.patch(f"/api/docs/{doc}/pages/2/lines/{line['id']}", json={"text": "  سطر   مصحح "})
    assert next(l for l in answer.json()["pages"]["2"]["lines"] if l["id"] == line["id"])["text"] == "سطر مصحح"
    result = run(client, doc, kind="extract")
    assert not result["stale"]
    # a field written by hand is kept when the record is read again
    result = client.patch(f"/api/docs/{doc}/record", json={"PRENOM": "HICHAME"}).json()
    assert result["record"]["PRENOM"] == "HICHAME" and "PRENOM" in result["record_edits"]
    assert run(client, doc, kind="extract")["record"]["PRENOM"] == "HICHAME"
    assert client.patch(f"/api/docs/{doc}/record", json={"HEURENAISS": "7"}).json()["record"]["HEURENAISS"] == 7
    assert client.patch(f"/api/docs/{doc}/record", json={"NOPE": "x"}).status_code == 400
    assert client.patch(f"/api/docs/{doc}/record", json={"SEXE": 3}).status_code == 400


def test_web_refuses_bad_uploads_unknown_documents_and_busy_documents(web, monkeypatch):
    client, calls = web
    doc = client.post("/api/docs").json()["id"]
    assert client.put(f"/api/docs/{doc}/pages/2", files={"file": ("x.jpg", b"not an image", "image/jpeg")}
                      ).status_code == 400
    assert client.put(f"/api/docs/{doc}/pages/3", files={"file": ("x.jpg", scan(), "image/jpeg")}).status_code == 404
    assert client.get("/api/docs/" + "0" * 32).status_code == 404
    assert client.get("/api/docs/..%2F..%2Fetc").status_code == 404
    assert client.post(f"/api/docs/{doc}/run", json={"kind": "nope"}).status_code == 400
    # a rotated phone photo is stored upright
    data = io.BytesIO()
    image = Image.new("RGB", (600, 400), "white")
    exif = image.getexif()
    exif[0x0112] = 6
    image.save(data, "JPEG", exif=exif)
    page = client.put(f"/api/docs/{doc}/pages/2", files={"file": ("../../évil name.jpg", data.getvalue(), "image/jpeg")}
                      ).json()["pages"]["2"]
    assert (page["width"], page["height"]) == (400, 600) and "/" not in page["name"]
    # another scan of the same size: same stored name, but a new rev (the address of its image in the browser)
    again = client.put(f"/api/docs/{doc}/pages/2", files={"file": ("b.jpg", data.getvalue(), "image/jpeg")}
                       ).json()["pages"]["2"]
    assert again["image"] == page["image"] and again["rev"] and again["rev"] != page["rev"]
    assert client.get(f"/api/docs/{doc}/pages/2/image").headers["content-type"] == "image/png"
    # while a run works on the document, changes are refused
    from madoc_web import store
    store.update(doc, lambda d: d.update(job={"id": "x", "state": "running"}))
    assert client.delete(f"/api/docs/{doc}").status_code == 409
    assert client.put(f"/api/docs/{doc}/pages/2/lines", json={"lines": []}).status_code == 409
    assert client.post(f"/api/docs/{doc}/run", json={"kind": "auto"}).status_code == 409


def test_web_a_failed_step_is_reported_and_does_not_stop_the_worker(web, monkeypatch):
    client, calls = web
    from madoc_web import jobs, process
    doc = new_document(client)

    def broken(*args):
        raise RuntimeError("kraken exploded")
    monkeypatch.setattr(process, "segment_image", broken)
    client.post(f"/api/docs/{doc}/run", json={"kind": "auto"})
    assert jobs.wait()
    job = client.get(f"/api/docs/{doc}/status").json()["job"]
    assert job["state"] == "error" and "segment 2 failed (RuntimeError" in job["error"]
    monkeypatch.undo()
    monitoring = client.get("/api/monitor").json()
    assert monitoring["summary"]["segment"]["errors"] >= 1 and "madoc_step_seconds_count" in client.get("/metrics").text


def test_web_settings_are_checked_kept_and_applied(web):
    client, calls = web
    import kraken_engine
    from madoc_web import config
    answer = client.put("/api/settings", json={"cpu_threads": 3, "device": "cpu", "keep_models_seconds": 60})
    assert answer.status_code == 200 and answer.json()["values"]["cpu_threads"] == 3
    assert kraken_engine.THREADS == 3 and kraken_engine.KEEP_SECONDS == 60 and config.device() == "cpu"
    config.reset_cache()
    assert config.load()["cpu_threads"] == 3                                          # read back from the file
    for bad in ({"cpu_threads": 0}, {"device": "gpu"}, {"nope": 1}, {"use_ner": "yes"}, {"retention_hours": 9999}):
        assert client.put("/api/settings", json=bad).status_code == 400, bad
    assert config.device({"device": "cuda:7"}) == "cpu"                               # a GPU that is not there


def test_web_login_and_old_documents(tmp_path, monkeypatch):
    from madoc_web import api, config, monitor, store
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(monitor, "_gpu_tool", [None])
    config.reset_cache()
    with TestClient(api.create_app(api.logins("ana:1234"))) as client:
        assert client.get("/healthz").status_code == 200                              # the healthcheck: no login
        assert client.get("/api/docs").status_code == 401 and client.get("/").status_code == 401
        wrong = {"Authorization": "Basic " + base64.b64encode(b"ana:0000").decode()}
        assert client.get("/api/docs", headers=wrong).status_code == 401
        good = {"Authorization": "Basic " + base64.b64encode(b"ana:1234").decode()}
        assert client.get("/api/docs", headers=good).status_code == 200
        doc = client.post("/api/docs", headers=good).json()
        assert store.purge(now=doc["updated"] + 3599) == 0 and store.purge(now=doc["updated"] + 3601) == 1
        assert client.get(f"/api/docs/{doc['id']}", headers=good).status_code == 404
    config.reset_cache()
    with pytest.raises(SystemExit):
        api.logins("no-password")


def test_page_one_reads_the_header_and_keeps_the_rest_as_notes():
    from madoc_web import pageone
    found = pageone.parse(PAGE_ONE + ["رسم ولادة رقم 44 في هامش آخر"])
    assert found["fields"]["PRENOM"] == "HAMZA" and found["fields"]["NOMAR"] == "الساحلي"
    assert found["fields"]["TEXTEMENTIONS"].split("\n") == ["تزوج بتاريخ فاتح يناير", "رسم ولادة رقم 44 في هامش آخر"]
    assert found["sources"]["TEXTEMENTIONSFR"] == [8]
    record, comparisons, notes = pageone.merge({"NUMACTE": "", "PRENOM": "hamza", "PRENOMAR": "حمزة", "NOMAR": "x",
                                                "NOM": ""}, found)
    assert record["NUMACTE"] == found["fields"]["NUMACTE"] and record["NOM"] == "SAHLI"
    assert {c["field"]: c["same"] for c in comparisons} == {"NUMACTE": None, "PRENOMAR": True, "PRENOM": True,
                                                           "NOMAR": False, "NOM": None}
    empty = pageone.parse([])
    assert not any(empty["fields"].values()) and pageone.merge({"NOM": "A"}, empty)[1] == []