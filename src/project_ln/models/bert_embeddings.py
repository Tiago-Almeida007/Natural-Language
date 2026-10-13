"""Frozen sentence-embedding model: Bio_ClinicalBERT -> LogisticRegression.

This is transfer learning WITHOUT fine-tuning: the encoder is used purely as a
feature extractor (weights frozen), its dense output feeds a linear classifier.
Encoding is the only expensive step and its output is cached, so re-runs are
instant and offline.

Evaluation mode for this model: `--cv 0` (default) is a single stratified
train/validation split (90/10); `--cv K` is K-fold stratified CV (out-of-fold
predictions). Both are repeated over `--seeds` and aggregated as mean +/- std.
CV is affordable here because encoding is done once and cached — only the cheap
linear classifier is refit per fold. With augmentation, each fold's training
partition is augmented and its new texts are encoded (also cached).

Flags (model-specific): see main.py for --bert-model / --bert-fields / --seeds / --cv.
"""

import hashlib

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

from ..augment import augment_train_fold
from ..data import DATA_DIR, encode_labels, load_rows
from ..utils import report

CACHE_FILE = DATA_DIR / "embeddings.npz"
DEFAULT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
FIELD_ORDER = ["sample_name", "description", "keywords"]

_EMBEDDER = None


def get_embedder(model_name):
    """Load (once, on demand) the frozen encoder on CPU."""
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER = SentenceTransformer(model_name, device="cpu")
    return _EMBEDDER


def make_text(row):
    """Concatenate the concise, high-signal fields for the encoder."""
    parts = [row["sample_name"], row["description"], row["keywords"]]
    return " ".join(p for p in parts if p)


def _load_cache():
    if not CACHE_FILE.exists():
        return {}
    try:
        data = np.load(CACHE_FILE, allow_pickle=False)
        return dict(zip(data["keys"].tolist(), data["vectors"]))
    except (OSError, ValueError, KeyError):
        return {}


def _save_cache(cache):
    if not cache:
        return
    keys = np.array(list(cache.keys()))
    vectors = np.vstack(list(cache.values())).astype(np.float32)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_FILE, keys=keys, vectors=vectors)


def encode_texts(model_name, texts):
    """Embed each text, using the on-disk cache; returns an (n, dim) array.

    Cache keys are sha256(model|text), so augmented/duplicated texts are stored
    once. Only cache misses touch the encoder.
    """
    keys = [hashlib.sha256(f"{model_name}|{t}".encode()).hexdigest() for t in texts]
    cache = _load_cache()

    pending = {}
    for key, text in zip(keys, texts):
        if key not in cache and key not in pending:
            pending[key] = text
    if pending:
        embedder = get_embedder(model_name)
        pending_keys = list(pending)
        vectors = embedder.encode(
            [pending[k] for k in pending_keys],
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        for key, vector in zip(pending_keys, vectors):
            cache[key] = vector.astype(np.float32)
        _save_cache(cache)

    return np.vstack([cache[key] for key in keys])


def encode_fields(model_name, rows):
    """Encode each field separately and hstack the vectors (no text join).

    Each of FIELD_ORDER gets its own 768-d embedding; the three blocks are
    concatenated into a (n, 3*dim) feature matrix. Empty fields contribute an
    all-zeros block so they add no signal.
    """
    dim = get_embedder(model_name).get_sentence_embedding_dimension()
    blocks = []
    for field in FIELD_ORDER:
        texts = [r[field] for r in rows]
        unique = list(dict.fromkeys(t for t in texts if t))
        vectors = encode_texts(model_name, unique) if unique else np.empty((0, dim))
        by_text = dict(zip(unique, vectors))
        block = np.zeros((len(rows), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            if text:
                block[i] = by_text[text]
        blocks.append(block)
    return np.hstack(blocks)


def build_features(model_name, rows, field_mode):
    """Feature matrix for rows: joined text (concat) or per-field vectors."""
    if field_mode == "per_field":
        return encode_fields(model_name, rows)
    return encode_texts(model_name, [make_text(r) for r in rows])


def _train_matrix(rows, X_all, idx_train, labels, args, model_name):
    """Embeddings + labels for a training partition (augmented if requested)."""
    if args.augment == "back_translation":
        train_rows = [rows[i] for i in idx_train]
        aug_rows, aug_labels = augment_train_fold(
            train_rows, labels[idx_train].tolist(), target_count=args.augment_target
        )
        X_train = build_features(model_name, aug_rows, args.bert_fields)
        return X_train, np.array(aug_labels)
    return X_all[idx_train], labels[idx_train]


def _fit_predict(X_train, y_train, X_val):
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train, y_train)
    return clf.predict(X_val)


def run_experiment(args):
    rows = load_rows()
    labels, names = encode_labels([r["label"] for r in rows])
    model_name = args.bert_model
    seeds = [int(s) for s in args.seeds.split(",")]
    n_splits = args.cv
    aug_str = (
        f", augment={args.augment}(target={args.augment_target})"
        if args.augment != "none"
        else ""
    )
    mode_str = f"{n_splits}-fold CV" if n_splits > 1 else "90/10 split"
    fields_str = (
        "per_field(sample_name+description+keywords)"
        if args.bert_fields == "per_field"
        else "concat(sample_name+description+keywords)"
    )
    print(
        f"loaded {len(rows)} rows, {len(names)} specialties "
        f"(model={model_name}, fields={fields_str}, {mode_str}{aug_str})"
    )

    X_all = build_features(model_name, rows, args.bert_fields)

    scores = []
    for seed in seeds:
        if n_splits > 1:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            oof_pred = np.empty(len(rows), dtype=int)
            for idx_train, idx_val in skf.split(np.arange(len(rows)), labels):
                X_train, y_train = _train_matrix(
                    rows, X_all, idx_train, labels, args, model_name
                )
                oof_pred[idx_val] = _fit_predict(X_train, y_train, X_all[idx_val])
            acc, macro, min_f1 = report(
                labels, oof_pred, names, f"{n_splits}-fold CV (seed={seed})"
            )
        else:
            idx_train, idx_val = train_test_split(
                np.arange(len(rows)),
                test_size=0.10,
                stratify=labels,
                random_state=seed,
            )
            X_train, y_train = _train_matrix(
                rows, X_all, idx_train, labels, args, model_name
            )
            pred = _fit_predict(X_train, y_train, X_all[idx_val])
            acc, macro, min_f1 = report(
                labels[idx_val], pred, names, f"90/10 split (seed={seed})"
            )
        scores.append((acc, macro, min_f1))

    arr = np.array(scores)
    print(
        f"\n=== {len(seeds)} seeds, {mode_str} — mean +/- std "
        f"=== acc {arr[:, 0].mean():.4f} +/- {arr[:, 0].std():.4f}  "
        f"macroF1 {arr[:, 1].mean():.4f} +/- {arr[:, 1].std():.4f}  "
        f"minF1 {arr[:, 2].mean():.4f} +/- {arr[:, 2].std():.4f}"
    )
    return arr
