"""Clinical-Longformer frozen encoder for the long `transcription` field.

Bio_ClinicalBERT caps at 512 tokens, which throws away the diagnostic tail of
the median 651-token note. `yikuan8/Clinical-Longformer` is Clinical-BERT
extended to a 4096-token window (sliding-window attention + global attention
on the CLS token), so it reads the whole note in one pass.

Same frozen-feature design as `bert_embeddings`: mean-pool the last hidden
state over non-pad tokens, L2-normalize, cache on disk. Encoding is the only
expensive step (~9 s/note on this CPU, ~6.6 h for all 2638 unique notes), so
the cache is checkpointed to disk every `CHECKPOINT_EVERY` notes and an
interrupted run resumes instead of restarting.
"""

import hashlib
from pathlib import Path

import numpy as np
import torch

from .data import DATA_DIR

MODEL_ID = "yikuan8/Clinical-Longformer"
CACHE_FILE = DATA_DIR / "longformer_embeddings.npz"
MAX_LENGTH = 4096
BATCH_SIZE = 4
CHECKPOINT_EVERY = 25

_MODEL = None
_TOKENIZER = None


def get_longformer():
    """Load (once) the frozen Longformer on CPU, using all physical cores."""
    global _MODEL, _TOKENIZER
    if _MODEL is None:
        from transformers import AutoModel, AutoTokenizer

        # the default thread count can leave cores idle on this 8-thread CPU
        torch.set_num_threads(8)
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
        _MODEL = AutoModel.from_pretrained(MODEL_ID).eval()
    return _MODEL, _TOKENIZER


def _key(text):
    return hashlib.sha256(f"{MODEL_ID}|{text}".encode()).hexdigest()


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
    # savez appends ".npz" unless the name already ends with it
    tmp = CACHE_FILE.parent / (CACHE_FILE.stem + ".tmp.npz")
    np.savez_compressed(tmp, keys=keys, vectors=vectors)
    tmp.replace(CACHE_FILE)


def _encode_batch(model, tokenizer, texts):
    """Encode one batch of notes -> (batch, 768) L2-normalized float32.

    Inputs are padded to the batch max and then up to a multiple of 512
    (Longformer's attention window) so the sliding-window kernel is happy.
    """
    encoded = [
        tokenizer(t, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        for t in texts
    ]
    width = max(e["input_ids"].shape[1] for e in encoded)
    width = ((width + 511) // 512) * 512
    input_ids = torch.zeros(len(encoded), width, dtype=torch.long)
    attention = torch.zeros(len(encoded), width, dtype=torch.long)
    for i, e in enumerate(encoded):
        length = e["input_ids"].shape[1]
        input_ids[i, :length] = e["input_ids"][0]
        attention[i, :length] = e["attention_mask"][0]

    global_mask = torch.zeros_like(input_ids)
    global_mask[:, 0] = 1  # CLS attends globally
    with torch.inference_mode():
        hidden = model(
            input_ids=input_ids,
            attention_mask=attention,
            global_attention_mask=global_mask,
        ).last_hidden_state
    mask = attention.bool().unsqueeze(-1)
    pooled = (hidden * mask).sum(1) / mask.sum(1)
    pooled = torch.nn.functional.normalize(pooled, dim=-1)
    return pooled.float().numpy()


def encode_transcriptions(texts, batch_size=BATCH_SIZE):
    """(n, 768) L2-normalized vectors for `texts`, cached on disk.

    Only cache misses hit the encoder. Batches are formed over a length-sorted
    ordering so padding waste is small (notes run 128-3753 tokens), and the
    cache is flushed every `CHECKPOINT_EVERY` newly-encoded notes.
    """
    keys = [_key(t) for t in texts]
    cache = _load_cache()
    pending = {}
    for key, text in zip(keys, texts):
        if key not in cache and key not in pending:
            pending[key] = text

    if pending:
        model, tokenizer = get_longformer()
        # sort by length: cheap batches first, minimal padding per batch
        items = sorted(
            pending.items(),
            key=lambda kv: len(
                tokenizer(kv[1], truncation=True, max_length=MAX_LENGTH)["input_ids"]
            ),
        )
        done = 0
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            vectors = _encode_batch(model, tokenizer, [text for _, text in batch])
            for (key, _), vector in zip(batch, vectors):
                cache[key] = vector
            done += len(batch)
            if done % CHECKPOINT_EVERY < batch_size or done == len(items):
                _save_cache(cache)
                print(f"  [lf-cache] {done}/{len(items)} newly encoded and saved", flush=True)

    return np.vstack([cache[key] for key in keys]).astype(np.float32)
