import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter

from .data import DATA_DIR

CACHE_FILE = DATA_DIR / "augmented_cache.json"

# Order in which fields are considered for translation; the first non-empty one wins.
AUGMENT_FIELD_ORDER = ["description", "transcription", "keywords", "sample_name"]

# Errors that mean "translation failed" (network down, bad response): we fall
# back to the original text instead of crashing the training run.
_TRANSLATE_ERRORS = (OSError, ValueError, KeyError, IndexError)


def _load_cache() -> dict:
    if not CACHE_FILE.exists():
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except _TRANSLATE_ERRORS:
        return {}


def _save_cache(cache: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def back_translate(text: str, lang: str = "es", timeout: int = 15) -> str:
    """Round-trip translate EN -> lang -> EN, cached by (lang, text).

    Returns the original text on any translation error, so the caller always
    gets usable (if not augmented) data.
    """
    if not text or not isinstance(text, str):
        return text
    cache = _load_cache()
    key = hashlib.sha256(f"{lang}|{text}".encode()).hexdigest()
    if key in cache:
        return cache[key]

    try:
        url1 = (
            "https://translate.googleapis.com/translate_a/single"
            f"?client=gtx&sl=en&tl={lang}&dt=t&q=" + urllib.parse.quote(text)
        )
        req1 = urllib.request.Request(url1, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req1, timeout=timeout) as resp:
            res1 = json.loads(resp.read().decode("utf-8"))
        mid = "".join(item[0] for item in res1[0] if item[0])

        url2 = (
            "https://translate.googleapis.com/translate_a/single"
            f"?client=gtx&sl={lang}&tl=en&dt=t&q=" + urllib.parse.quote(mid)
        )
        req2 = urllib.request.Request(url2, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req2, timeout=timeout) as resp:
            res2 = json.loads(resp.read().decode("utf-8"))
        back = "".join(item[0] for item in res2[0] if item[0])
    except _TRANSLATE_ERRORS:
        return text

    cache[key] = back
    _save_cache(cache)
    return back


def augment_train_fold(
    X_train: list[dict],
    y_train: list[int],
    target_count: int = 100,
    languages: list[str] | None = None,
) -> tuple[list[dict], list[int]]:
    """Top every class up to `target_count` rows via back-translation.

    No-op for classes already at/above the target, so only minority classes are
    augmented. Each synthetic row is a copy of a class original with exactly one
    field back-translated; languages cycle round-robin. Must be called on a
    training partition only (never validation/test).
    """
    if languages is None:
        languages = ["es", "de", "fr"]
    if target_count <= 0:
        return X_train, y_train

    counts = Counter(y_train)
    X_aug = list(X_train)
    y_aug = list(y_train)

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
    return X_aug, y_aug
