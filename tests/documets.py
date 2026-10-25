"""The documents of the tests: the anonymized samples of samples/, with the reading mistakes of each test.

No test writes a certificate of its own: every document is a sample with some words replaced, so the tests
contain no name, address or number of a real certificate. Each replacement must find its text in the sample
(a sample edited later makes the test fail here, not silently test something else).

The anonymization shifted the Gregorian dates of the samples: where a test needs the Hijri and the Gregorian
date of a pair to agree (to the day), the replacement also writes the Gregorian day the Hijri date gives.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sample(number):
    """The text of samples/<number>.txt, ending with a line end (as a transcription of the HTR does)."""
    text = (ROOT / "samples" / f"{number}.txt").read_text(encoding="utf-8")
    return text if text.endswith("\n") else text + "\n"


def from_sample(number, *replacements):
    """samples/<number>.txt with (old, new) replaced, in order; `old` must be in the text (first one replaced,
    or every one with a third element "all")."""
    text = sample(number)
    for old, new, *every in replacements:
        assert old in text, f"samples/{number}.txt has no {old!r}"
        text = text.replace(old, new) if every else text.replace(old, new, 1)
    return text


# --- sample 3 -------------------------------------------------------------------------------------------
# a misread year: "ألفين و ثمانين" (2080, impossible) next to the Hijri years 1428 / 1429; the Gregorian days are
# those of the Hijri dates (26 Dhu al-Hijja 1428 = 5 January 2008, 4 Muharram 1429 = 13 January 2008)
DATE_DOCUMENT = from_sample(
    3,
    ("موافق: سادس عشر يناير سنة ألفين و ثمانية", "موافق: خامس يناير سنة ألفين و ثمانين"),
    ("موافق رابع و عشري يناير سنة ألفين و ثمانية", "موافق ثالث عشر يناير سنة ألفين و ثمانين"),
)

# --- sample 2 -------------------------------------------------------------------------------------------
# the child's Arabic first name misread: "أيرب" / PRENOM AYOUB
NAME_DOCUMENT = from_sample(2, ("الإسم الشخصي : أيوب", "الإسم الشخصي : أيرب"))

# --- sample 5 -------------------------------------------------------------------------------------------
# the certificate read as a whole: a misread year word (the Gregorian day set to the Hijri one: 12 Rabi I 1436
# = 3 January 2015), the minute's label cut to "وة", a doubled letter in an address word, the child's first name
# (names.json) and family name (the father's line), a place written twice on its line, the mother's father
# (written as the child's father), two places of cities.json, the house number of the declarant's address
COHERENCE_DOCUMENT = from_sample(
    5,
    ("موافق رابع و عشري يناير سنة ألفين و خمسة عشرة", "موافق ثالث يناير سنة ألفين و خمسن عشرة"),
    ("و الدقيقة : الثلاثين.", "وة : الثلاثين."),
    ("ولد ب جماعة خريبڭة الحضرية اقليم خريبڭة.", "ولد ب محج المستتشفى مدينة خريبڭة."),
    ("الإسم الشخصي : إدريس", "الإسم الشخصي : إدريش"),
    ("الإسم العائلي : الودغيري", "الإسم العائلي : الودعيري"),
    ("جماعة سيدي رحال الشاطئ الحضرية اقليم برشيد", "جماعة سيدي رحال الشاطئ دائرة برشيد اقليم برشبد"),
    ("بنت خالد.", "بنت حالد."),
    ("المولودة بجماعة الأولاد اقليم سطات.", "المولودة بجماعة اليوسفة الحضرية اقليم أسي."),
    ("الساكنة بحي القدس الأول الرقم : 591", "الساكنة بحي القدس الأول الرقم : 519"),
)

# --- sample 3 -------------------------------------------------------------------------------------------
# the declarant is the mother: her name misread in the header (3 letters), the day and the month of the child's
# Gregorian date misread (4 January 2008 read "رامع بير"; the Hijri date gives the 5th), a compound place
# ("اولاد فارس", not Fès), a word of the form with a letter more, the label of the house number cut, two
# different house numbers
DECLARANT_DOCUMENT = from_sample(
    3,
    ("موافق: سادس عشر يناير سنة ألفين و ثمانية", "موافق: رامع بير سنة ألفين و ثمانية"),
    ("و والدتها غزلان الزروالي", "و والدتها عرلاب الزروالي"),
    ("المولودة بجماعة اولاد عزوز اقليم خريبڭة.", "المولودة بجماعة اولاد فارس اقليم سطات."),
    ("عمرها ثلاثون سنة", "عمرهار ثلاثون سنة"),
    ("الشطر الأول الرقم: 57 خريبڭة.", "الشطر الأول الرق : : 75 خريبڭة."),
)
# the same, with the year of the child's Gregorian date on the next line
DECLARANT_TWO_LINES = DECLARANT_DOCUMENT.replace("موافق: رامع بير سنة ألفين و ثمانية",
                                                 "موافق: رامع بير\nسنة ألفين و ثمانية")

# --- sample 1 -------------------------------------------------------------------------------------------
# the declarant is the father: the child's family name two letters from the father's ("الإيريشي" / IDRISSI),
# the declarant's reading of the father misread ("الإدريسن": the child's Latin name decides), the age, the
# father's Gregorian year misread with the day and the month of his Hijri date (3 Rabi I 1407 = 6 November
# 1986), the child's Gregorian day and month misread with the year on the next line (11 Safar 1438 = 12
# November 2016)
FATHER_DECLARANT_DOCUMENT = from_sample(
    1,
    ("موافق خامس عشر دجنبر سنة ألفين و ستة عشرة", "موافق : ثاني عشر تير\nسنة ألفين و ستة عشرة"),
    ("الإسم العائلي : الإدريسي", "الإسم العائلي : الإيريشي"),
    ("السيد نبيل الإدريسي (والده)", "السيد نبيل الإدريسن (والده)"),
    ("عمره احدى و ثلاثون سنة", "عمره ثمانة و ثلاثون سنة"),
    ("موافق تاسع دجنبر سنة ألف و تسعمائة و ستة و ثمانين", "موافق سادس نونبر سنة ألف و تسعمائة و تسعة و ثمانين"),
)

# --- sample 6 -------------------------------------------------------------------------------------------
# a minute's label with no value, a neighbourhood misread three ways, the parents' address going on after the
# mother's job, "بنت" glued to the mother's family name, the child's family name one letter off
EMPTY_MINUTE_DOCUMENT = from_sample(
    6,
    ("على الساعة: السادسة عشرة و الدقيقة: الصفر.", "و الدقيقة\nعلى الساعة: السادسة عشرة"),
    ("ولد ب محج المستشفى مدينة خريبڭة.", "ولد ب حي المناومة مدينة خريبڭة."),
    ("الإسم العائلي: الساحلي", "الإسم العائلي: الساحني"),
    ("نادية العمراني بنت رضا.", "نادية العمرانيبست رضا."),
    ("حرفتها: ربة البيتت.\nالساكنان بتجزئة الفتح الرقم : 32 خريبڭة",
     "الساكنان: حي المعاومة.\nحرفتها: ربة البيت.\nالزنقة: 19 الرقم : 32 خريبڭة"),
    ("الساكن بتجزئة الفيق الرقم : 32", "الساكن بحي القاومة الزنقة: 1 الرقم : 32"),
)

# --- sample 2, for the interface and the parser ---------------------------------------------------------
APP_TEXT = sample(2)

# sample 1 without its three nationality lines: {child}, {father}, {mother} put them back (or others)
NATIONALITY_DOCUMENT = from_sample(
    1,
    ("ذكر\nجنسيته مغربية\n", "ذكر\n{child}"),
    ("جنسيته : مغربية.\n", "{father}"),
    ("جنسيتها : مغربية\n", "{mother}"),
)