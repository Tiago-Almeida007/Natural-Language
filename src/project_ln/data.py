import re
import unicodedata
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
TRAIN_PATH = BASE_DIR / "train.csv"
TEST_PATH = BASE_DIR / "test_no_labels.csv"
DATA_DIR = BASE_DIR / "data"

SPECIALTIES = {
    "Cardiovascular-Pulmonary",
    "Dermatology",
    "Gastroenterology",
    "General Medicine",
    "Neurology",
    "Neurosurgery",
    "Obstetrics-Gynecology",
    "Ophthalmology",
    "Orthopedic",
    "Psychiatry-Psychology",
    "Radiology",
    "Surgery",
}

TEXT_FIELDS = ["description", "sample_name", "transcription", "keywords"]

# A physical line starting with a known specialty opens a new train record; any
# following line that does not is a continuation of the previous record's text.
_RECORD_RE = re.compile(r"^(" + "|".join(re.escape(s) for s in sorted(SPECIALTIES)) + r");")

# Typographic characters normalised to ASCII. Meaningful symbols (degree, micro
# sign, accent, middot, fraction) are deliberately kept: they carry clinical
# units and names.
_TYPOGRAPHY = str.maketrans({
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-",
    "\u2026": "...", "\u00a0": " ",
})


def clean_text(value):
    """Normalise one field: NFC, ASCII punctuation, drop stray quotes, tidy spaces.

    Applied identically to train and test (the PDF requires the same
    preprocessing on both). Idempotent: clean_text(clean_text(x)) == clean_text(x).
    """
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFC", value)
    text = text.translate(_TYPOGRAPHY)
    text = text.replace('"', "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _read_lines(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        content = f.read()
    return content.splitlines()


def load_rows(path=TRAIN_PATH):
    """Parse train.csv into one dict per record (semicolon-delimited).

    The raw file has broken quoting: stray `"` characters and records split
    across physical lines by a spreadsheet. We stitch a split record back
    together (the continuation is always a mid-word cut of `transcription`),
    normalise every field with `clean_text`, and keep the 7 incomplete rows
    (label + partial description only) by padding their missing fields with "".
    Rows whose label is not one of the 12 specialties are dropped.
    """
    lines = _read_lines(path)
    records = []
    for line in lines[1:]:  # skip header: medical_specialty;description;sample_name;transcription;keywords
        if _RECORD_RE.match(line):
            records.append([line, ""])
        elif records:
            records[-1][1] += line.replace('"', "").rstrip(";")

    rows = []
    for raw, tail in records:
        fields = raw.replace('"', "").split(";")
        while len(fields) > 5 and fields[-1] == "":
            fields.pop()
        if tail:
            while len(fields) < 4:
                fields.append("")
            fields[3] = fields[3].rstrip(";") + tail
        while len(fields) < 5:
            fields.append("")
        fields = [clean_text(f) for f in fields]
        label = fields[0]
        if label not in SPECIALTIES:
            continue
        rows.append({
            "label": label,
            "description": fields[1],
            "sample_name": fields[2],
            "transcription": fields[3],
            "keywords": fields[4],
        })
    return rows


def load_test(path=TEST_PATH):
    """Parse test_no_labels.csv into one dict per record, in file order.

    No header, 4 fields (description;sample_name;transcription;keywords), same
    cleaning as `load_rows`. The returned order defines the order of
    `results.txt`.
    """
    rows = []
    for line in _read_lines(path):
        if not line.strip():
            continue
        fields = line.replace('"', "").split(";")
        while len(fields) > 4 and fields[-1] == "":
            fields.pop()
        while len(fields) < 4:
            fields.append("")
        fields = [clean_text(f) for f in fields]
        rows.append({
            "description": fields[0],
            "sample_name": fields[1],
            "transcription": fields[2],
            "keywords": fields[3],
        })
    return rows


def texts_and_labels(rows, field):
    """Return (texts, labels) for one field, dropping rows with an empty field.

    field="all" keeps every row and returns the row dicts themselves as X
    (consumed by FieldSelector, which vectorizes each field separately);
    for a single field the list of that field's strings is returned instead.
    """
    if field == "all":
        texts = rows
        labels = [r["label"] for r in rows]
    else:
        keep = [r for r in rows if r[field]]
        texts = [r[field] for r in keep]
        labels = [r["label"] for r in keep]
    return texts, labels


def encode_labels(labels):
    names = sorted(set(labels))
    index = {name: i for i, name in enumerate(names)}
    return np.array([index[label] for label in labels]), names
