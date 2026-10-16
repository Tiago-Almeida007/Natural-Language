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

Augmentation: `--augment back_translation` adds new *texts* (encoded and cached);
`--augment embed_*` and `bt_embed_smote` work directly on the feature matrix in
`src/project_ln/embed_augment.py` and cost nothing extra.
"""

import hashlib

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

from ..augment import augment_train_fold
from ..data import DATA_DIR, encode_labels, load_rows
from ..embed_augment import augment_embeddings
from ..longformer_embeddings import encode_transcriptions
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
        # tokenize() is only used to measure/slice long transcriptions, never
        # to feed the model; without this it warns on every >512-token note.
        _EMBEDDER.tokenizer.model_max_length = 1_000_000
    return _EMBEDDER


def head_tail_truncate(text, tokenizer, max_tokens=508, head_tokens=254, tail_tokens=254):
    """Keep the first and last tokens of a long text instead of dropping the tail.

    Bio_ClinicalBERT only reads 512 tokens (510 after CLS/SEP). Clinical notes
    put the chief complaint / procedure at the top and the impression / plan /
    postoperative diagnosis at the bottom, with mostly routine narrative in
    between, so head+tail captures both diagnostic poles and drops only the
    middle: 254 + 254 + 2 = 510 <= 512, nothing overflows.
    """
    if not text:
        return ""
    tokens = tokenizer.tokenize(text)
    if len(tokens) <= max_tokens:
        return text
    # convert_tokens_to_string + tokenize is not an exact round trip (wordpieces
    # can re-split), so the re-encoded slice can be a few tokens longer than
    # requested. Shave the tail until it fits, worst case fall back to head-only.
    head, tail = head_tokens, tail_tokens
    for _ in range(4):
        candidate = tokenizer.convert_tokens_to_string(tokens[:head] + tokens[-tail:])
        if len(tokenizer.tokenize(candidate)) <= max_tokens:
            return candidate
        tail = max(0, tail - 16)
    return tokenizer.convert_tokens_to_string(tokens[:head])


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
        # Encode in chunks and save after each one, so an interruption only
        # loses the current chunk instead of the whole run.
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
            _save_cache(cache)
            done = min(start + chunk_size, len(pending_keys))
            print(f"    [cache] {done}/{len(pending_keys)} newly encoded and saved", flush=True)

    return np.vstack([cache[key] for key in keys])


def encode_fields(model_name, rows, transcription_encoder="none", transcription_mode="none"):
    """Encode each field separately and hstack the vectors (no text join).

    Each of FIELD_ORDER gets its own 768-d embedding; the three blocks are
    concatenated into a (n, 3*dim) feature matrix. Empty fields contribute an
    all-zeros block so they add no signal.

    Two mutually exclusive ways to add a fourth `transcription` block:
    - `transcription_mode="head_tail"`: Bio_ClinicalBERT reads the first+last
      254 tokens of the note (cached in the same embeddings.npz as everything
      else, keyed by the truncated text).
    - `transcription_encoder="longformer"`: Clinical-Longformer reads the whole
      note (cached in data/longformer_embeddings.npz).
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
    if transcription_mode == "head_tail":
        tokenizer = get_embedder(model_name).tokenizer
        texts = [r["transcription"] for r in rows]
        # dedupe on the *truncated* text, then map back to the raw row text
        sliced = {t: head_tail_truncate(t, tokenizer) for t in texts if t}
        unique_enc = list(dict.fromkeys(sliced.values()))
        vectors = encode_texts(model_name, unique_enc) if unique_enc else np.empty((0, dim))
        by_enc = dict(zip(unique_enc, vectors))
        block = np.zeros((len(rows), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            if text:
                block[i] = by_enc[sliced[text]]
        blocks.append(block)
    elif transcription_encoder == "longformer":
        texts = [r["transcription"] for r in rows]
        unique = list(dict.fromkeys(t for t in texts if t))
        vectors = encode_transcriptions(unique) if unique else np.empty((0, dim))
        by_text = dict(zip(unique, vectors))
        block = np.zeros((len(rows), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            if text:
                block[i] = by_text[text]
        blocks.append(block)
    return np.hstack(blocks)


def build_features(model_name, rows, field_mode, transcription_encoder="none", transcription_mode="none"):
    """Feature matrix for rows: joined text (concat) or per-field vectors."""
    if field_mode == "per_field":
        return encode_fields(model_name, rows, transcription_encoder, transcription_mode)
    return encode_texts(model_name, [make_text(r) for r in rows])


def _train_matrix(rows, X_all, idx_train, labels, args, model_name):
    """Embeddings + labels for a training partition (augmented if requested)."""
    if args.augment == "back_translation":
        train_rows = [rows[i] for i in idx_train]
        aug_rows, aug_labels = augment_train_fold(
            train_rows, labels[idx_train].tolist(), target_count=args.augment_target
        )
        X_train = build_features(model_name, aug_rows, args.bert_fields, args.transcription_encoder, args.transcription_mode)
        return X_train, np.array(aug_labels)
    if args.augment.startswith("embed_"):
        # embedding-space augmentation: X_all is already encoded and cached
        return augment_embeddings(
            X_all[idx_train],
            labels[idx_train],
            args.augment,
            target=args.augment_target,
            strength=args.augment_strength,
            alpha=args.augment_alpha,
            k=args.augment_k,
            seed=args.augment_seed,
        )
    if args.augment == "bt_embed_smote":
        # stage 1: back-translation (new texts, encoded+cached); stage 2: SMOTE
        # in embedding space on the result.
        train_rows = [rows[i] for i in idx_train]
        aug_rows, aug_labels = augment_train_fold(
            train_rows, labels[idx_train].tolist(), target_count=args.augment_target
        )
        X_bt = build_features(model_name, aug_rows, args.bert_fields, args.transcription_encoder, args.transcription_mode)
        second = args.augment_target_2 or args.augment_target + 100
        return augment_embeddings(
            X_bt,
            np.array(aug_labels),
            "embed_smote",
            target=second,
            k=args.augment_k,
            seed=args.augment_seed,
        )
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
    aug_str = ""
    if args.augment == "back_translation":
        aug_str = f", augment=back_translation(target={args.augment_target})"
    elif args.augment.startswith("embed_"):
        detail = {"embed_noise": f"strength={args.augment_strength}",
                  "embed_mixup": f"alpha={args.augment_alpha}",
                  "embed_smote": f"k={args.augment_k}"}[args.augment]
        aug_str = f", augment={args.augment}(target={args.augment_target}, {detail})"
    elif args.augment == "bt_embed_smote":
        second = args.augment_target_2 or args.augment_target + 100
        aug_str = (
            f", augment=bt+embed_smote(bt_target={args.augment_target}, "
            f"smote_target={second}, k={args.augment_k})"
        )
    mode_str = f"{n_splits}-fold CV" if n_splits > 1 else "90/10 split"
    lf_str = (
        "+transcription(longformer)"
        if args.transcription_encoder == "longformer"
        else ("+transcription(head_tail)" if args.transcription_mode == "head_tail" else "")
    )
    fields_str = (
        f"per_field(sample_name+description+keywords{lf_str})"
        if args.bert_fields == "per_field"
        else "concat(sample_name+description+keywords)"
    )
    print(
        f"loaded {len(rows)} rows, {len(names)} specialties "
        f"(model={model_name}, fields={fields_str}, {mode_str}{aug_str})"
    )

    X_all = build_features(
        model_name, rows, args.bert_fields, args.transcription_encoder, args.transcription_mode
    )

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
