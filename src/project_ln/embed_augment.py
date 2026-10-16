"""Embedding-space augmentation for the frozen-BERT + LogReg model.

Operates directly on the already-encoded feature matrix (no re-encoding, no
network), so each experiment is seconds instead of minutes. Only minority
classes are topped up, mirroring `augment.augment_train_fold`: every class
below `target` gets `target - n_c` synthetic vectors.

All synthetic points are re-normalized to unit norm: the real embeddings are
L2-normalized (they live on the unit hypersphere) but interpolations and noise
perturbations do not, and leaving them off-scale shifts the feature magnitude
the logistic regression sees.
"""

import numpy as np

METHODS = ("embed_noise", "embed_mixup", "embed_smote", "bt_embed_smote")


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


def _class_indices(y, c):
    return np.flatnonzero(y == c)


def add_gaussian_noise(X, y, target=200, strength=0.1, seed=0):
    """Scatter each minority class around its members with isotropic noise.

    sigma is the global per-feature std of the fold: a per-class std is
    unusable for the ~25-row classes (it estimates nothing useful and can be
    ~0, which would freeze the augmentation).
    """
    rng = np.random.default_rng(seed)
    slots = _minority_slots(y, target)
    if not slots:
        return X, y
    sigma = X.std(axis=0)
    synth, labels = [], []
    for cls, need in slots:
        members = _class_indices(y, cls)
        pick = rng.choice(members, size=need, replace=True)
        noise = rng.normal(0.0, strength, size=(need, 1)) * sigma
        synth.append(X[pick] + noise)
        labels.append(np.full(need, cls, dtype=y.dtype))
    return (
        np.vstack([X, _renorm(np.vstack(synth))]),
        np.concatenate([y, np.concatenate(labels)]),
    )


def mixup_within_class(X, y, target=200, alpha=0.4, seed=0):
    """Convex combinations of two same-class points (within-class MixUp).

    lambda ~ Beta(alpha, alpha). With alpha < 1 the mass sits near 0 and 1,
    so most synthetics stay close to a real sample rather than collapsing
    toward the class centroid. Single-member classes fall back to noise.
    """
    rng = np.random.default_rng(seed)
    slots = _minority_slots(y, target)
    if not slots:
        return X, y
    synth, labels = [], []
    for cls, need in slots:
        members = _class_indices(y, cls)
        if len(members) < 2:
            synth.append(_noise_block(X, members, need, rng))
        else:
            i = rng.choice(members, size=need, replace=True)
            j = rng.choice(members, size=need, replace=True)
            lam = rng.beta(alpha, alpha, size=(need, 1))
            synth.append(lam * X[i] + (1.0 - lam) * X[j])
        labels.append(np.full(need, cls, dtype=y.dtype))
    return (
        np.vstack([X, _renorm(np.vstack(synth))]),
        np.concatenate([y, np.concatenate(labels)]),
    )


def _noise_block(X, members, need, rng, strength=0.1):
    """Noise fallback used when a class has a single member."""
    sigma = X.std(axis=0)
    pick = rng.choice(members, size=need, replace=True)
    return X[pick] + rng.normal(0.0, strength, size=(need, 1)) * sigma


def smote(X, y, target=200, k=5, seed=0):
    """SMOTE: interpolate each minority sample toward one of its k same-class
    nearest neighbours. Self-implemented (imbalanced-learn is not a dependency).

    k is clamped to n_c - 1 (a class cannot have more neighbours than it has
    other members), so tiny classes simply use fewer neighbours. Single-member
    classes fall back to noise.
    """
    rng = np.random.default_rng(seed)
    slots = _minority_slots(y, target)
    if not slots:
        return X, y
    synth, labels = [], []
    for cls, need in slots:
        members = _class_indices(y, cls)
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


def augment_embeddings(X, y, method, target=200, strength=0.1, alpha=0.4, k=5, seed=0):
    """Dispatch one embedding-space augmentation over a training fold."""
    if method == "embed_noise":
        return add_gaussian_noise(X, y, target, strength, seed)
    if method == "embed_mixup":
        return mixup_within_class(X, y, target, alpha, seed)
    if method == "embed_smote":
        return smote(X, y, target, k, seed)
    raise ValueError(f"unknown embedding augmentation {method!r}")
