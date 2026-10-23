"""Stage 0b - recognition: PageXML file (from segmentation.py) -> txt file, one line of text per text line.

    python src/recognition.py scan.xml                    # writes scan.txt next to the xml
    python src/recognition.py scan.xml -o out/scan.txt
    python src/recognition.py scan.xml --image scan.jpg   # when the image moved since the segmentation
    python src/recognition.py scan.xml --model models/htr.mlmodel --device cuda:0

It runs the kraken command line the same way it was tested by hand:

    kraken -f page -i scan.xml scan.txt ocr -m models/htr.mlmodel

kraken writes every line in visual order (the letters of an Arabic line come out reversed: "ةدالو مسر").
Every line then goes through python-bidi's get_display (pip install python-bidi), which gives the normal
text ("رسم ولادة") the parser and the NER models need. --raw keeps the visual order kraken wrote.
"""

import argparse
import re
import sys
import tempfile
from pathlib import Path
from xml.sax.saxutils import quoteattr

from common import resolve_device, run_kraken

ROOT = Path(__file__).resolve().parents[1]
MODEL_RECOGNITION = ROOT / "models" / "htr.mlmodel"


def find_image(xml_text, base_dir, override=None):
    """The image of the xml: the --image option, the imageFilename written in the xml, or a file named like it."""
    candidates = [Path(override)] if override else []
    match = re.search(r'imageFilename="([^"]+)"', xml_text)
    if match:
        name = Path(match.group(1))
        candidates += [name, Path(base_dir) / name, Path(base_dir) / name.name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"image of the xml not found (tried {[str(c) for c in candidates]}), use --image")


def prepare_xml(xml, image_path=None):
    """PageXML (text, or the Path of an xml file) -> xml text naming its image with a full path.

    kraken reads the image from the xml, so the copy given to kraken names it with its full path (escaped:
    a path may contain "&" or quotes, which would break the xml).
    """
    if isinstance(xml, Path):
        xml_text, base_dir = xml.read_text(encoding="utf-8"), xml.parent
    else:
        xml_text, base_dir = str(xml), Path.cwd()
    image = find_image(xml_text, base_dir, image_path)
    return re.sub(r'imageFilename="[^"]*"', lambda m: f"imageFilename={quoteattr(str(image))}", xml_text, count=1)


def to_logical_order(texts):
    """kraken's visual order ("ةدالو مسر") -> normal text ("رسم ولادة"), with python-bidi."""
    from bidi.algorithm import get_display  # pip install python-bidi

    return [get_display(text) for text in texts]


def run_recognition(xml_path, output_txt, model=MODEL_RECOGNITION, device="cpu", pagexml=False):
    """Run the kraken ocr command on the segmentation xml, write the text (or a PageXML) to output_txt."""
    run_kraken([
        "-d", resolve_device(device),
        "-f", "page",           # the input is a PageXML segmentation, the lines come from it
        *(["-x"] if pagexml else []),   # PageXML output: polygons, baselines, text and confidence per line
        "-i", str(xml_path), str(output_txt),
        "ocr", "-m", str(model),
    ], output_txt, "recognition")


def recognize_xml(xml, image_path=None, model=MODEL_RECOGNITION, device="cpu", fix_order=True):
    """PageXML (text, or the Path of an xml file) -> text, one line per text line.

    fix_order=True turns the visual order kraken writes ("ةدالو مسر") into the normal text ("رسم ولادة") with
    python-bidi, which is what the parser needs. fix_order=False returns kraken's raw lines.
    image_path replaces the image named in the xml (when the image moved).
    """
    xml_text = prepare_xml(xml, image_path)
    with tempfile.TemporaryDirectory() as folder:
        xml_file, output_txt = Path(folder) / "segmentation.xml", Path(folder) / "text.txt"
        xml_file.write_text(xml_text, encoding="utf-8")
        run_recognition(xml_file, output_txt, model, device)
        lines = [line.strip() for line in output_txt.read_text(encoding="utf-8").splitlines()]
    if fix_order:
        lines = to_logical_order(lines)
    return "\n".join(line for line in lines if line) + "\n"


def _points(text):
    """"x,y x,y ..." -> [[x, y], ...]"""
    points = []
    for pair in (text or "").split():
        try:
            x, y = pair.split(",")[:2]
            points.append([round(float(x)), round(float(y))])
        except ValueError:
            continue
    return points


def parse_pagexml(xml_text):
    """PageXML written by kraken -> [{"id", "text", "confidence", "polygon", "baseline"}] in reading order.

    confidence is the kraken score of the line (0..1), or None when the xml has none (it is then averaged from
    the characters when those have one). The text is kraken's own, in visual order.
    """
    from xml.etree import ElementTree

    root = ElementTree.fromstring(xml_text.encode("utf-8") if isinstance(xml_text, str) else xml_text)
    lines = []
    for line in root.iter():
        if not line.tag.endswith("}TextLine") and line.tag != "TextLine":
            continue
        entry = {"id": line.get("id", f"line_{len(lines) + 1}"), "text": "", "confidence": None,
                 "polygon": [], "baseline": []}
        for child in line:
            name = child.tag.rsplit("}", 1)[-1]
            if name == "Coords":
                entry["polygon"] = _points(child.get("points"))
            elif name == "Baseline":
                entry["baseline"] = _points(child.get("points"))
            elif name == "TextEquiv" and not entry["text"]:
                unicode_node = next((n for n in child if n.tag.rsplit("}", 1)[-1] == "Unicode"), None)
                entry["text"] = (unicode_node.text or "").strip() if unicode_node is not None else ""
                if child.get("conf"):
                    entry["confidence"] = float(child.get("conf"))
        if entry["confidence"] is None:
            glyphs = [float(n.get("conf")) for n in line.iter() if n.tag.rsplit("}", 1)[-1] == "TextEquiv"
                      and n.get("conf") and n is not line]
            if glyphs:
                entry["confidence"] = sum(glyphs) / len(glyphs)
        lines.append(entry)
    return lines


def recognize_lines(xml, image_path=None, model=MODEL_RECOGNITION, device="cpu", fix_order=True):
    """PageXML (text or Path) -> list of lines with text, confidence, polygon and baseline (see parse_pagexml).

    One kraken run gives everything: the text is converted to normal order like recognize_xml (fix_order),
    lines without text are dropped. If kraken's PageXML has no text, the plain text run is used instead and
    the lines keep the polygons of the segmentation (without confidence).
    """
    xml_text = prepare_xml(xml, image_path)
    with tempfile.TemporaryDirectory() as folder:
        xml_file, output = Path(folder) / "segmentation.xml", Path(folder) / "recognized.xml"
        xml_file.write_text(xml_text, encoding="utf-8")
        run_recognition(xml_file, output, model, device, pagexml=True)
        lines = parse_pagexml(output.read_text(encoding="utf-8"))
        if not any(line["text"] for line in lines):
            # no text in the PageXML: read it from the plain text run, same order as the segmentation
            segmented = parse_pagexml(xml_text)
            text_file = Path(folder) / "text.txt"
            run_recognition(xml_file, text_file, model, device)
            texts = text_file.read_text(encoding="utf-8").splitlines()
            if len(texts) == len(segmented):
                lines = [{**seg, "text": text.strip()} for seg, text in zip(segmented, texts)]
            else:  # the lines cannot be paired with their polygons: text only
                lines = [{"id": f"line_{i}", "text": text.strip(), "confidence": None, "polygon": [], "baseline": []}
                         for i, text in enumerate(texts, 1)]
    if fix_order:
        for line, text in zip(lines, to_logical_order([line["text"] for line in lines])):
            line["text"] = text
    return [line for line in lines if line["text"]]


def main():
    parser = argparse.ArgumentParser(description="Recognize the text lines of a PageXML file (kraken HTR).")
    parser.add_argument("xml", type=Path, help="Path to the PageXML written by segmentation.py")
    parser.add_argument("-o", "--output", type=Path, help="txt file to write (default: next to the xml)")
    parser.add_argument("--image", type=Path, help="the image, if it is not where the xml says")
    parser.add_argument("--model", type=Path, default=MODEL_RECOGNITION, help="recognition model")
    parser.add_argument("--device", default="cpu", help="cpu, cuda:0 ...")
    parser.add_argument("--raw", action="store_true", help="keep the visual order kraken writes (no python-bidi)")
    args = parser.parse_args()
    if not args.xml.is_file():
        parser.error(f"Input file not found: {args.xml}")
    if not args.model.is_file():
        parser.error(f"Model not found: {args.model}")
    output = args.output or args.xml.with_suffix(".txt")
    text = recognize_xml(args.xml, args.image, args.model, args.device, fix_order=not args.raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"{len(text.splitlines())} lines written to {output}", file=sys.stderr)


if __name__ == "__main__":
    main()