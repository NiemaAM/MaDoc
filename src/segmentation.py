"""Stage 0a - segmentation: image -> PageXML file with the text lines found by the kraken baseline model.

    python src/segmentation.py scan.jpg                   # writes scan.xml next to the image
    python src/segmentation.py scan.jpg -o out/scan.xml
    python src/segmentation.py scan.jpg --model models/blla.mlmodel --device cuda:0

It runs the kraken command line the same way it was tested by hand:

    kraken -x -i scan.jpg scan.xml segment -bl -i models/blla.mlmodel

The PageXML keeps the image name so recognition.py can find the image again.
"""

import argparse
import sys
import tempfile
from pathlib import Path

from common import resolve_device, run_kraken

ROOT = Path(__file__).resolve().parents[1]
MODEL_SEGMENTATION = ROOT / "models" / "blla.mlmodel"


def run_segmentation(image_file, output_xml, model=MODEL_SEGMENTATION, device="cpu"):
    """Run kraken BLLA segmentation, PageXML output."""
    run_kraken([
        "-d", resolve_device(device),
        "-x",                   # PageXML output
        "-i", str(image_file), str(output_xml),
        "segment", "-bl", "-i", str(model),
    ], output_xml, "segmentation")


def segment_image(image_file, model=MODEL_SEGMENTATION, device="cpu"):
    """image file -> PageXML text (what recognition.recognize_xml takes). The image is named with its full
    path inside the xml, so recognition finds it wherever the xml is stored."""
    image_file = Path(image_file).resolve()
    with tempfile.TemporaryDirectory() as folder:
        output_xml = Path(folder) / "segmentation.xml"
        run_segmentation(image_file, output_xml, model, device)
        return output_xml.read_text(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Segment a scanned document into text lines (kraken blla).")
    parser.add_argument("image", type=Path, help="Path to the input image")
    parser.add_argument("-o", "--output", type=Path, help="xml file to write (default: next to the image)")
    parser.add_argument("--model", type=Path, default=MODEL_SEGMENTATION, help="segmentation model")
    parser.add_argument("--device", default="cpu", help="cpu, cuda:0 ...")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Input image not found: {args.image}")
    if not args.model.is_file():
        parser.error(f"Model not found: {args.model}")
    output = args.output or args.image.with_suffix(".xml")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(segment_image(args.image, args.model, args.device), encoding="utf-8")
    print(f"PageXML written to {output}", file=sys.stderr)


if __name__ == "__main__":
    main()