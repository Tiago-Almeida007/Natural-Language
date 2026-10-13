import csv
from pathlib import Path

import numpy as np

TRAIN_PATH = Path(__file__).resolve().parents[2] / "train.csv"
DATA_DIR = TRAIN_PATH.parent / "data"

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


def load_rows(path=TRAIN_PATH):
    """Parse train.csv into one dict per record (semicolon-delimited).

    Rows whose label is not one of the 12 known specialties are dropped; that
    discards the malformed rows where an unquoted newline split a record.
    """
    rows = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)  # medical_specialty;description;sample_name;transcription;keywords
        for row in reader:
            if len(row) < 5:
                continue
            label = row[0].strip()
            if label not in SPECIALTIES:
                continue
            rows.append({
                "label": label,
                "description": row[1].strip(),
                "sample_name": row[2].strip(),
                "transcription": row[3].strip(),
                "keywords": row[4].strip(),
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
