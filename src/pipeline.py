"""pipeline.py - runs the whole chain on one scan, one PageXML or one text file.

    segmentation > recognition > [spelling checker] > classificator > parser > converter > mapper > translator > validator
    image -> xml     xml -> txt     txt -> txt           txt -> NER json -> record

The input decides where the chain starts: an image (.jpg .png .tif ...) runs everything, a PageXML
(.xml, from segmentation.py) starts at recognition, a .txt starts at the classificator.

    python src/pipeline.py scan.jpg
    python src/pipeline.py scan.xml --image scan.jpg          # the xml is already segmented
    python src/pipeline.py samples/6.txt
    python src/pipeline.py samples/6.txt -o result.json --report report.json --save-stages out/
    python src/pipeline.py samples/6.txt --no-ner            # rules only, no model is loaded
    python src/pipeline.py samples/6.txt --ner ner.json      # reuse a saved classificator output
    python src/pipeline.py scan.jpg --spellcheck              # HTR spelling fixes by a local model (Ollama)

The record goes to stdout (or -o), the validation report and the NER notes go to stderr.
--save-stages DIR writes the output of every stage (0a_segmentation.xml, 0b_text.txt, 0c_spellchecked.txt, 1_ner.json ...
5_translated.json, 6_report.json), which is the easiest way to see which stage changed what.
The recognized text is the normal Arabic text (kraken's visual order is converted back), as the parser expects.
"""

import argparse
import sys
from pathlib import Path

from classificator import classify
from common import read_json_file, read_text_file, resolve_device, write_json
from recognition import MODEL_RECOGNITION, recognize_xml
from segmentation import MODEL_SEGMENTATION, segment_image
from spelling_checker import correct_text
from converter import convert
from mapper import map_fields
from parser import parse_with_notes
from translator import is_available, translate
from validator import format_report, validate

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def text_from_source(path, image=None, save_dir=None, device="cpu",
                     segmentation_model=MODEL_SEGMENTATION, recognition_model=MODEL_RECOGNITION):
    """Stages 0a / 0b: image -> xml -> text. A .xml starts at recognition, a .txt is read as it is."""
    path = Path(path)
    save_dir = Path(save_dir) if save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        # 0a. segmentation: image -> PageXML (an object handed to recognition, no file needed)
        xml = segment_image(path, segmentation_model, device)
        if save_dir:
            (save_dir / "0a_segmentation.xml").write_text(xml, encoding="utf-8")
        image = path
    elif suffix == ".xml":
        xml = path
    else:
        return read_text_file(path)
    # 0b. recognition: PageXML -> text. kraken writes visual order, recognize_xml gives the normal text back
    text = recognize_xml(xml, image, recognition_model, device, fix_order=True)
    if save_dir:
        (save_dir / "0b_text.txt").write_text(text, encoding="utf-8")
    return text


def run_pipeline(raw_text, ner=None, use_ner=True, use_translation=True, save_dir=None, use_french_ner=True,
                 stages=None):
    """raw text -> (record, validation report, notes). `ner` skips the models with a saved JSON.

    use_french_ner=False runs only the Arabic NER model (both run on this computer, see classificator.py).
    stages: a dict that receives the output of every stage ("ner", "parsed", "converted", "mapped").
    """
    save_dir = Path(save_dir) if save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    def save(name, obj):
        if save_dir:
            write_json(obj, save_dir / name)

    # 1. classificator: NER entities
    if ner is None and use_ner:
        ner = classify(raw_text, use_french=use_french_ner)
        for error in ner.get("errors", []):
            print(f"Warning: NER unavailable, rules only for it ({error})", file=sys.stderr)
    if ner is not None:
        save("1_ner.json", ner)
    # 2. parser: rules + NER entities
    parsed, notes = parse_with_notes(raw_text, ner)
    save("2_parsed.json", parsed)
    if stages is not None:
        stages.update({"ner": ner, "parsed": parsed})
    # 3. converter: dates, time, sex
    converted = convert(parsed)
    save("3_converted.json", converted)
    # 4. mapper: jobs, nationalities
    mapped = map_fields(converted)
    save("4_mapped.json", mapped)
    # 5. translator: names and places (local model, Ollama)
    translated = translate(mapped) if use_translation else mapped
    save("5_translated.json", translated)
    # 6. validator: report
    report = validate(translated, check_translation=use_translation and is_available())
    save("6_report.json", report)
    return translated, report, notes


def main():
    parser = argparse.ArgumentParser(
        description="Run the full birth certificate pipeline on a text file."
    )
    parser.add_argument("input_file", type=Path, help="scan (.jpg .png .tif), PageXML (.xml) or text (.txt)")
    parser.add_argument("-o", "--output", type=Path, help="write the record here instead of stdout")
    parser.add_argument("--report", type=Path, help="also write the validation report to this file")
    parser.add_argument("--save-stages", type=Path, metavar="DIR", help="write the JSON of every stage in DIR")
    parser.add_argument("--ner", type=Path, help="use this classificator JSON instead of running the models")
    parser.add_argument("--no-ner", action="store_true", help="do not run the NER models (rules only)")
    parser.add_argument("--no-translation", action="store_true", help="skip the translation stage (local model)")
    parser.add_argument("--image", type=Path, help="the scan of a .xml input, if it is not where the xml says")
    parser.add_argument("--segmentation-model", type=Path, default=MODEL_SEGMENTATION, help="kraken blla model")
    parser.add_argument("--recognition-model", type=Path, default=MODEL_RECOGNITION, help="kraken HTR model")
    parser.add_argument("--device", default="cpu", help="kraken device: cpu, cuda:0 ...")
    parser.add_argument("--strict", action="store_true", help="exit with code 1 when the report has errors")
    parser.add_argument("--spellcheck", action="store_true",
                        help="correct the HTR spelling mistakes with a local model (Ollama, nothing leaves the computer)")
    args = parser.parse_args()

    if not args.input_file.is_file():
        parser.error(f"Input file not found: {args.input_file}")
    if args.ner and not args.ner.is_file():
        parser.error(f"NER file not found: {args.ner}")

    # the same string goes to the classificator and to the parser: the NER offsets must match it
    raw_text = text_from_source(
        args.input_file, args.image, args.save_stages, resolve_device(args.device),
        args.segmentation_model, args.recognition_model,
    )
    # 0c. spelling checker (optional): only known words replace misspelt words, names are never touched
    if args.spellcheck:
        if args.ner:
            print("Warning: --ner was computed on the uncorrected text, its offsets may not match", file=sys.stderr)
        raw_text, spelling_notes = correct_text(raw_text)
        for note in spelling_notes:
            print(f"[spelling] {note}", file=sys.stderr)
        if args.save_stages:
            Path(args.save_stages).mkdir(parents=True, exist_ok=True)
            (Path(args.save_stages) / "0c_spellchecked.txt").write_text(raw_text, encoding="utf-8")
    record, report, notes = run_pipeline(
        raw_text,
        ner=read_json_file(args.ner) if args.ner else None,
        use_ner=not args.no_ner,
        use_translation=not args.no_translation,
        save_dir=args.save_stages,
    )
    write_json(record, args.output)
    if args.report:
        write_json(report, args.report)
    for note in notes:
        print(f"[parser] {note}", file=sys.stderr)
    print(format_report(report), file=sys.stderr)
    if args.strict and not report["valid"]:
        sys.exit(1)


if __name__ == "__main__":
    main()