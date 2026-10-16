#!/usr/bin/env python
"""Reproduce our best model (EXPERIMENTS.md entry 6d) and write results.txt.

Self-contained on purpose: it inlines the whole pipeline (data cleaning,
back-translation, embedding-space SMOTE, frozen-encoder features) so that this
one file can be copied anywhere and re-run. It imports nothing from
`project_ln` - only numpy, scikit-learn and sentence-transformers.

The recipe, chosen by 5-fold cross-validation on train.csv
(acc 0.6427 / macro-F1 0.6397 / min-F1 0.4659, clearing both graded bars of
>0.49 accuracy and no per-class F1 below 0.25):

  1. Features: frozen Bio_ClinicalBERT (`emilyalsentzer/Bio_ClinicalBERT`, no
     fine-tuning) encodes `sample_name`, `description` and `keywords`
     *separately*; the three 768-d vectors are hstacked into 2304-d.
     `transcription` is deliberately left out: it is too long for the 512-token
     window (median 651 tokens), and feeding it either truncated (entry 8) or
     whole through a 4096-token Clinical-Longformer (entry 7) changed nothing
     beyond noise.
  2. Data augmentation (two stages, both mandatory for the paper):
     a. back-translation EN->es/de/fr->EN tops every minority class up to 200
        rows (only the first non-empty field of each source row is translated);
     b. SMOTE *in embedding space* then tops every class up to 500 rows,
        interpolating each synthetic point toward one of its k=5 same-class
        nearest neighbours.
  3. Classifier: LogisticRegression(max_iter=2000, class_weight="balanced"),
     fit on the whole 2669-row training set.
  4. Prediction: applied to test_no_labels.csv in file order -> results.txt.

Runtime: ~1.5-2 h from a cold start - that is ~9000 texts to encode (~1.5 h of
CPU) plus ~667 back-translations over the network (~10 min). Progress is
written through to `data/embeddings.npz` and `data/augmented_cache.json` every
100 / every translation, so an interrupted run resumes where it stopped and a
second run is near-instant. Those caches are an internal optimisation only:
`data/` is gitignored, no cache file is required, and a fresh checkout of the
project (code + train.csv + test_no_labels.csv) runs on its own.

Usage:  uv run python make_results.py [--out results.txt]
"""

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

# --------------------------------------------------------------------------
# paths (everything is resolved relative to this file, so the script works
# from any working directory)
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "train.csv"
TEST_PATH = BASE_DIR / "test_no_labels.csv"
DATA_DIR = BASE_DIR / "data"
EMBED_CACHE = DATA_DIR / "embeddings.npz"
BT_CACHE = DATA_DIR / "augmented_cache.json"

BERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
BERT_FIELDS = ["sample_name", "description", "keywords"]  # transcription excluded
BT_TARGET = 200     # stage 2a: back-translation rows per minority class
SMOTE_TARGET = 500  # stage 2b: embedding-space SMOTE rows per minority class
SMOTE_K = 5
SMOTE_SEED = 0

# --------------------------------------------------------------------------
# 1. data loading (inlined from project_ln/data.py)
# --------------------------------------------------------------------------
SPECIALTIES = {
    "Cardiovascular-Pulmonary", "Dermatology", "Gastroenterology",
    "General Medicine", "Neurology", "Neurosurgery", "Obstetrics-Gynecology",
    "Ophthalmology", "Orthopedic", "Psychiatry-Psychology", "Radiology", "Surgery",
}

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

    Applied identically to train and test. Idempotent.
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
    together, normalise every field with `clean_text`, and keep the incomplete
    rows (label + partial description only) by padding missing fields with "".
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
    cleaning as `load_rows`. This order defines the order of results.txt.
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


# --------------------------------------------------------------------------
# 2. stage 2a: back-translation augmentation (inlined from project_ln/augment.py)
# --------------------------------------------------------------------------
# Order in which fields are considered for translation; the first non-empty one wins.
AUGMENT_FIELD_ORDER = ["description", "transcription", "keywords", "sample_name"]
_BT_LANGS = ["es", "de", "fr"]
# Errors that mean "translation failed" (network down, bad response): we fall
# back to the original text instead of crashing the run.
_BT_ERRORS = (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError)

_BT_STATS = {"cached": 0, "fetched": 0, "failed": 0}
_BT_CACHE = None  # loaded once, written through on every successful translation


def _load_bt_cache() -> dict:
    global _BT_CACHE
    if _BT_CACHE is None:
        if not BT_CACHE.exists():
            _BT_CACHE = {}
        else:
            try:
                with open(BT_CACHE, "r", encoding="utf-8") as f:
                    _BT_CACHE = json.load(f)
            except _BT_ERRORS:
                _BT_CACHE = {}
    return _BT_CACHE


def _save_bt_cache() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(BT_CACHE, "w", encoding="utf-8") as f:
        json.dump(_BT_CACHE, f, ensure_ascii=False, indent=2)


def _bt_key(text, lang):
    return hashlib.sha256(f"{lang}|{text}".encode()).hexdigest()


def _bt_fetch_once(text, lang, timeout=15):
    """One EN->lang->EN round trip. Returns the translation or None on failure.

    Google's translate endpoint occasionally answers with a body whose payload
    is null (rate limiting), which used to crash with TypeError; anything
    unexpected is treated as a failed attempt so the caller can retry.
    """
    try:
        url1 = (
            "https://translate.googleapis.com/translate_a/single"
            f"?client=gtx&sl=en&tl={lang}&dt&q=" + urllib.parse.quote(text)
        )
        req1 = urllib.request.Request(url1, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req1, timeout=timeout) as resp:
            res1 = json.loads(resp.read().decode("utf-8"))
        mid = "".join(item[0] for item in res1[0] if item[0])

        url2 = (
            "https://translate.googleapis.com/translate_a/single"
            f"?client=gtx&sl={lang}&tl=en&dt&q=" + urllib.parse.quote(mid)
        )
        req2 = urllib.request.Request(url2, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req2, timeout=timeout) as resp:
            res2 = json.loads(resp.read().decode("utf-8"))
        back = "".join(item[0] for item in res2[0] if item[0])
    except _BT_ERRORS:
        return None
    return back if back else None


def back_translate(text, lang="es"):
    """Round-trip translate EN -> lang -> EN, cached by (lang, text).

    Tries the requested language, then the other two, with a couple of retries
    per language and a short pause between calls. Returns the original text if
    every attempt fails (counted, and reported loudly at the end), so the run
    always produces a model even with no network.
    """
    if not text or not isinstance(text, str):
        return text
    cache = _load_bt_cache()
    key = _bt_key(text, lang)
    if key in cache:
        _BT_STATS["cached"] += 1
        return cache[key]

    for attempt_lang in [lang] + [l for l in _BT_LANGS if l != lang]:
        for retry in range(3):
            back = _bt_fetch_once(text, attempt_lang)
            if back is not None:
                time.sleep(0.25)  # be gentle with the endpoint
                cache[_bt_key(text, attempt_lang)] = back
                _save_bt_cache()
                _BT_STATS["fetched"] += 1
                return back
            time.sleep(1.0 + 2.0 * retry)

    _BT_STATS["failed"] += 1
    return text


def augment_train_fold(X_train, y_train, target_count=BT_TARGET, languages=None):
    """Top every class up to `target_count` rows via back-translation.

    No-op for classes already at/above the target, so only minority classes are
    augmented. Each synthetic row is a copy of a class original with exactly one
    field back-translated; languages cycle round-robin.
    """
    if languages is None:
        languages = list(_BT_LANGS)
    if target_count <= 0:
        return X_train, y_train

    counts = Counter(y_train)
    X_aug = list(X_train)
    y_aug = list(y_train)
    done = 0

    for cls, count in counts.items():
        if count >= target_count:
            continue
        cls_indices = [i for i, label in enumerate(y_train) if label == cls]
        for added in range(target_count - count):
            orig_row = X_train[cls_indices[added % len(cls_indices)]]
            lang = languages[added % len(languages)]
            aug_row = dict(orig_row)
            for field in AUGMENT_FIELD_ORDER:
                if aug_row.get(field):
                    aug_row[field] = back_translate(aug_row[field], lang=lang)
                    break
            X_aug.append(aug_row)
            y_aug.append(cls)
            done += 1
            if done % 50 == 0:
                print(f"    [bt] {done} synthetic rows "
                      f"(fetched {_BT_STATS['fetched']}, failed {_BT_STATS['failed']})",
                      flush=True)
    return X_aug, y_aug


# --------------------------------------------------------------------------
# 3. stage 2b: embedding-space SMOTE (inlined from project_ln/embed_augment.py)
# --------------------------------------------------------------------------
def _renorm(X):
    """Scale each row back to unit length (zero rows stay zero)."""
    if len(X) == 0:
        return X
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (X / norms).astype(np.float32)


def _minority_slots(y, target):
    """(classes needing synthetics, how many each) for counts below target."""
    counts = np.bincount(y, minlength=int(y.max()) + 1)
    return [(c, int(target - counts[c])) for c in range(len(counts)) if counts[c] < target]


def _noise_block(X, members, need, rng, strength=0.1):
    """Noise fallback used when a class has a single member."""
    sigma = X.std(axis=0)
    pick = rng.choice(members, size=need, replace=True)
    return X[pick] + rng.normal(0.0, strength, size=(need, 1)) * sigma


def smote(X, y, target=SMOTE_TARGET, k=SMOTE_K, seed=SMOTE_SEED):
    """SMOTE: interpolate each minority sample toward one of its k same-class
    nearest neighbours (self-implemented; imbalanced-learn is not a dependency).

    k is clamped to n_c - 1, so tiny classes simply use fewer neighbours.
    Synthetic vectors are re-normalised to the unit hypersphere, matching the
    L2-normalised real embeddings.
    """
    rng = np.random.default_rng(seed)
    slots = _minority_slots(y, target)
    if not slots:
        return X, y
    synth, labels = [], []
    for cls, need in slots:
        members = np.flatnonzero(y == cls)
        if len(members) < 2:
            synth.append(_noise_block(X, members, need, rng))
        else:
            points = X[members]
            kk = min(k, len(points) - 1)
            # squared distances via the identity ||a-b||^2 = |a|^2+|b|^2-2a.b
            sq = np.sum(points * points, axis=1)
            d2 = sq[:, None] + sq[None, :] - 2.0 * (points @ points.T)
            np.fill_diagonal(d2, np.inf)
            nn = np.argsort(d2, axis=1)[:, :kk]
            base = rng.choice(len(points), size=need, replace=True)
            step = rng.integers(0, kk, size=need)
            other = nn[base, step]
            lam = rng.uniform(0.0, 1.0, size=(need, 1))
            synth.append(points[base] + lam * (points[other] - points[base]))
        labels.append(np.full(need, cls, dtype=y.dtype))
    return (
        np.vstack([X, _renorm(np.vstack(synth))]),
        np.concatenate([y, np.concatenate(labels)]),
    )


# --------------------------------------------------------------------------
# 4. frozen-encoder features (inlined from project_ln/models/bert_embeddings.py)
# --------------------------------------------------------------------------
_EMBEDDER = None


def get_embedder(model_name=BERT_MODEL):
    """Load (once, on demand) the frozen encoder on CPU."""
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer

        print(f"  loading {model_name} ...", flush=True)
        _EMBEDDER = SentenceTransformer(model_name, device="cpu")
        # Fields can be longer than the 512-token window; we only need the
        # encoder's pooled output, and without this it warns on every long text.
        _EMBEDDER.tokenizer.model_max_length = 1_000_000
    return _EMBEDDER


def _load_embed_cache():
    if not EMBED_CACHE.exists():
        return {}
    try:
        data = np.load(EMBED_CACHE, allow_pickle=False)
        return dict(zip(data["keys"].tolist(), data["vectors"]))
    except (OSError, ValueError, KeyError):
        return {}


def _save_embed_cache(cache):
    if not cache:
        return
    keys = np.array(list(cache.keys()))
    vectors = np.vstack(list(cache.values())).astype(np.float32)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EMBED_CACHE, keys=keys, vectors=vectors)


def encode_texts(texts, model_name=BERT_MODEL):
    """Embed each text, using the on-disk cache; returns an (n, dim) array.

    Cache keys are sha256(model|text), so duplicated/augmented texts are stored
    once and only cache misses touch the encoder. The cache is written every
    100 new texts so an interruption only loses the current chunk.
    """
    keys = [hashlib.sha256(f"{model_name}|{t}".encode()).hexdigest() for t in texts]
    cache = _load_embed_cache()

    pending = {}
    for key, text in zip(keys, texts):
        if key not in cache and key not in pending:
            pending[key] = text
    if pending:
        embedder = get_embedder(model_name)
        pending_keys = list(pending)
        chunk_size = 100
        for start in range(0, len(pending_keys), chunk_size):
            chunk_keys = pending_keys[start : start + chunk_size]
            vectors = embedder.encode(
                [pending[k] for k in chunk_keys],
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            for key, vector in zip(chunk_keys, vectors):
                cache[key] = vector.astype(np.float32)
            _save_embed_cache(cache)
            done = min(start + chunk_size, len(pending_keys))
            print(f"    [cache] {done}/{len(pending_keys)} newly encoded and saved", flush=True)

    return np.vstack([cache[key] for key in keys])


def build_features(rows, model_name=BERT_MODEL):
    """Per-field feature matrix: each field gets its own 768-d embedding, the
    blocks are hstacked into (n, 3*dim). Empty fields contribute an all-zeros
    block so they add no signal."""
    dim = get_embedder(model_name).get_sentence_embedding_dimension()
    blocks = []
    for field in BERT_FIELDS:
        texts = [r[field] for r in rows]
        unique = list(dict.fromkeys(t for t in texts if t))
        vectors = encode_texts(unique, model_name) if unique else np.empty((0, dim))
        by_text = dict(zip(unique, vectors))
        block = np.zeros((len(rows), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            if text:
                block[i] = by_text[text]
        blocks.append(block)
    return np.hstack(blocks)


# --------------------------------------------------------------------------
# 5. the run
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(prog="make_results.py")
    parser.add_argument("--out", default=str(BASE_DIR / "results.txt"), help="output path")
    args = parser.parse_args()
    out_path = Path(args.out)

    print("== 1/5 load data ==")
    rows = load_rows()
    test_rows = load_test()
    names = sorted({r["label"] for r in rows})
    label_to_idx = {name: i for i, name in enumerate(names)}
    y = np.array([label_to_idx[r["label"]] for r in rows])
    print(f"train {len(rows)} rows / {len(names)} specialties, test {len(test_rows)} rows")
    print("train class counts:", dict(Counter(r["label"] for r in rows)))

    print("== 2/5 encode training texts (per-field Bio_ClinicalBERT) ==")
    X_all = build_features(rows)
    print(f"X_all {X_all.shape}")

    print(f"== 3/5 augmentation 2a: back-translation to {BT_TARGET}/class ==")
    aug_rows, aug_labels = augment_train_fold(rows, y.tolist(), target_count=BT_TARGET)
    y_aug = np.array(aug_labels)
    added_bt = len(aug_rows) - len(rows)
    print(f"added {added_bt} back-translated rows "
          f"(cache hits {_BT_STATS['cached']}, fetched {_BT_STATS['fetched']}, "
          f"failed {_BT_STATS['failed']})")
    X_bt = build_features(aug_rows) if added_bt else X_all

    print(f"== 4/5 augmentation 2b: embedding-space SMOTE to {SMOTE_TARGET}/class ==")
    X_syn, y_syn = smote(X_bt, y_aug, target=SMOTE_TARGET, k=SMOTE_K, seed=SMOTE_SEED)
    added_smote = len(y_syn) - len(y_aug)
    print(f"added {added_smote} SMOTE rows; final training set {X_syn.shape[0]} rows")

    print("== 5/5 fit on ALL training rows and predict the test set ==")
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_syn, y_syn)
    print(f"fitted on {X_syn.shape[0]} rows x {X_syn.shape[1]} features")

    X_test = build_features(test_rows)
    pred = clf.predict(X_test)
    predicted = [names[i] for i in pred]

    if len(predicted) != len(test_rows):
        raise SystemExit(f"expected {len(test_rows)} predictions, got {len(predicted)}")
    bad = sorted({p for p in predicted if p not in SPECIALTIES})
    if bad:
        raise SystemExit(f"predicted labels outside the 12 specialties: {bad}")

    out_path.write_text("\n".join(predicted) + "\n", encoding="utf-8")
    dist = Counter(predicted)
    print(f"\nwrote {out_path} ({len(predicted)} lines, no header)")
    print("predicted class counts:")
    for name in names:
        print(f"  {name:<24} {dist[name]:>4}  ({dist[name] / len(predicted):.1%})")
    if _BT_STATS["failed"]:
        print(f"\nWARNING: {_BT_STATS['failed']} back-translations failed and fell back "
              f"to the original text - the final model may differ slightly from the "
              f"cross-validated recipe. Re-run with network access to fix this.")
    return out_path


if __name__ == "__main__":
    sys.exit(main())
