"""The lines of a page as PageXML: what kraken's recognition reads, and what the PageXML download gives.

Same shape as the PageXML kraken writes (one region for the page, a Coords polygon and a Baseline per line), so
a line drawn or corrected by hand is read exactly like a line found by the segmentation model."""

from datetime import datetime, timezone
from xml.sax.saxutils import escape, quoteattr

NAMESPACE = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"


def points(values):
    return " ".join(f"{int(round(x))},{int(round(y))}" for x, y in values)


def build(lines, image_name, width, height, with_text=False):
    """lines: [{"id", "polygon", "baseline", "text"}] -> PageXML text naming `image_name`."""
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for line in lines:
        if len(line.get("polygon") or []) < 3 or len(line.get("baseline") or []) < 2:
            continue    # not a line kraken can read
        text = ""
        if with_text and line.get("text"):
            confidence = line.get("confidence")
            conf = f' conf="{confidence:.4f}"' if isinstance(confidence, (int, float)) else ""
            text = f"<TextEquiv{conf}><Unicode>{escape(line['text'])}</Unicode></TextEquiv>"
        rows.append(f'<TextLine id={quoteattr(str(line["id"]))} custom="type {{type:default;}}">'
                    f'<Coords points="{points(line["polygon"])}"/>'
                    f'<Baseline points="{points(line["baseline"])}"/>{text}</TextLine>')
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n<PcGts xmlns="{NAMESPACE}">'
            f"<Metadata><Creator>MaDoc</Creator><Created>{now}</Created><LastChange>{now}</LastChange></Metadata>"
            f'<Page imageFilename={quoteattr(str(image_name))} imageWidth="{int(width)}" imageHeight="{int(height)}">'
            f'<TextRegion id="textblock_1"><Coords points="0,0 0,{int(height)} {int(width)},{int(height)} '
            f'{int(width)},0"/>{"".join(rows)}</TextRegion></Page></PcGts>\n')
