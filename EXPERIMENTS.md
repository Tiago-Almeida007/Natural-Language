# Experiments log

Running record of what was tried, the settings, the numbers, and the conclusion.
Every new experiment must be logged here (append to the tables, keep the code in
`src/project_ln/` in sync). All scores are on `train.csv` only (CV or a held-out
split of it) - the real `test_no_labels.csv` is untouched.

Graded bars (from `Project-2026-Description.pdf`): accuracy > 0.49 and no per-class
F1 below 0.25.

Run command: `uv run python -m project_ln.main --model <m> [--field <f>] [<flags>]`
(e.g. `uv run python -m project_ln.main --model nb --field description --uniform-prior`).
Each model lives in its own file under `src/project_ln/models/`.

## TODO

- [x] Data cleaning (Phase 1): shared `clean_text()` + robust line-stitching
      loader in `src/project_ln/data.py`, plus a `load_test()` for the 4-field
      unlabelled file. Row count 2613 -> 2669 (recovers 3 line-split and 7
      incomplete rows). Results: entries 1d-clean / 5c-clean (below).
- [x] Data augmentation beyond back-translation (Phase 3): embedding-space
      (Gaussian noise / SMOTE / within-class MixUp) in
      `src/project_ln/embed_augment.py`. Result: chained back-translation +
      SMOTE (`--augment bt_embed_smote`) is the new best (entries 6a-6d).
- [ ] Group-aware CV (Phase 4): `StratifiedGroupKFold` on the transcription so
      duplicate notes never straddle train/validation; report grouped vs current.
- [x] Data augmentation (mandatory for the paper template): back-translation of
      minority classes, applied inside each CV fold's training partition only
      (never before splitting). Implemented in `src/project_ln/augment.py`
      (`--augment back_translation [--augment-target N]`). Results: entries 2/2b/2c.
- [ ] Pick a single `--augment-target` and regenerate a final comparison against
      the unaugmented 1d baseline (currently 2b meets both bars at target=200;
      the accuracy/min-F1 trade-off is sensitive to this knob).
- [ ] Augment more than one field per synthetic example (v1 augments only the
      first non-empty field) and/or add more languages (currently es/de/fr).
- [ ] Investigate whether a larger per-field `max_features` lifts the classes
      still weakest under augmentation (Gastroenterology, Obstetrics-Gynecology).
- [ ] Submission model for `results.txt`: entry 5c (bert `--bert-fields per_field`
      + back-translation target=200, 5-fold CV) is the current best — use it, but
      first decide whether to run the final model on the full train set (no split)
      before predicting test.
- [ ] Try back-translation target above 200 for `bert --cv 5` (4c improved up to
      200; check where it plateaus), and try embedding-space augmentation
      (MixUp/Gaussian noise) for the dense model.

## Implemented models

### TF-IDF + MultinomialNB (`models/naive_bayes.py`)

Baseline to beat: TF-IDF on `description`, lowercase + stemming, MultinomialNB.
Evaluation for this model: 5-fold stratified CV only (no single split).

| # | Model | Accuracy | Macro-F1 | Min per-class F1 | Note |
|---|-------|---------:|---------:|-----------------:|------|
| 1 | PDF baseline: desc + stem TF-IDF + MultinomialNB | 0.376 | 0.116 | 0.000 | `--model nb --field description` (learned class priors) |
| 1b | same + uniform prior (`--uniform-prior`, `fit_prior=False`) | 0.408 | 0.303 | 0.000 | priors forced to 1/12; Neurosurgery is the only class still at 0.000 |
| 1d | per-field TF-IDF union (4 fields, 20k cap each) + uniform prior | **0.501** | 0.262 | 0.000 | `--field all --uniform-prior`; clears 0.49 accuracy bar |
| 2 | 1d + back-translation augmentation, target=100 | **0.506** | 0.384 | 0.059 | first augmentation; macro-F1 nearly doubles, min-F1 off zero |
| 2b | same, target=200 | **0.499** | 0.454 | **0.264** | **meets both graded bars (acc>0.49, min-F1>=0.25)** |
| 2c | same, target=250 | 0.490 | **0.468** | 0.261 | best macro-F1, but accuracy just under the 0.49 bar |

#### Notes on 1b

**Why `fit_prior=False`.** Standard NB learns priors from counts: Surgery =
875/2612 ~ 33.5%, Dermatology = 25/2612 ~ 0.95%. That ~35x gap makes ambiguous
documents default to Surgery. `fit_prior=False` forces P(class) = 1/12 for all
specialties, so decisions rest on text evidence alone. Result: acc 0.376 -> 0.408,
macro-F1 0.116 -> 0.303. Surgery recall drops 0.96 -> 0.59 while minority recalls
appear (Dermatology 0.08, Psychiatry F1 0.392). Still below both graded bars.

**Hunch: why Neurosurgery stays at 0.000.** In CV, Neurosurgery is the only class
still at 0.000 (the other 11 are all non-zero, e.g. Dermatology 0.133, Radiology
0.257). Looking at the 78 Neurosurgery rows and the confusion matrix, ~53% are
predicted Surgery and ~42% Orthopedic. Our intuition (not yet verified): many
Neurosurgery descriptions are spine procedures (discectomy, fusion, laminectomy,
plates, screws) whose vocabulary is shared with Orthopedic and generic Surgery,
and the short `description` field rarely carries brain/nerve keywords that would
separate it. Worth testing by inspecting misclassified rows or richer fields
(per-field union, see 1d/2b).

#### Notes on 1d/2 (per-field union + back-translation augmentation)

**Per-field union (1d).** `--field all` = *per-field TF-IDF union*: each of the
4 fields gets its own `TfidfVectorizer` (stemmed, 20k features/field) inside a
`FeatureUnion`, sparse-hstacked. Field identity is preserved — a token from
`keywords` and the same token from `transcription` are separate features.
Accuracy jumps past the graded bar vs 1b: 0.408 -> 0.501. But macro-F1 *drops*
(0.303 -> 0.262) and four classes sit at 0.000 F1 (Dermatology, Neurosurgery,
Ophthalmology, Psychiatry-Psychology): more features help the majority classes
but don't fix the minority collapse.

**Back-translation augmentation (2/2b/2c).** `--augment back_translation` fills
each minority class in a training partition up to `--augment-target` samples by
round-trip translating original texts EN->{es,de,fr}->EN (Google endpoint,
`src/project_ln/augment.py`). Translations are cached in
`data/augmented_cache.json`, so repeat runs are instant. Augmentation is applied
**inside each CV fold's training partition only**; validation folds are never
augmented, so scores stay on real text. v1 augments just the first non-empty
field of each sampled row and cycles languages round-robin.

**Result.** This is what finally lifts the floor: with target=200 (entry 2b) the
model **clears both graded bars** — acc 0.499 > 0.49 and min-F1 0.264 >= 0.25.
Raising the target trades accuracy for minority F1 (target=250: macro 0.468 but
acc 0.490, just under). The classes that remain weakest under augmentation are
Gastroenterology (0.30) and Obstetrics-Gynecology (0.38), not the tiny ones —
Neurosurgery, the old 0.000 offender, reaches 0.33.

### Frozen BERT embeddings (`models/bert_embeddings.py`)

Transfer learning WITHOUT fine-tuning: `emilyalsentzer/Bio_ClinicalBERT` is used
as a frozen feature extractor (weights frozen, no gradients), mean-pooled and
L2-normalized, feeding a `LogisticRegression(class_weight='balanced')`. Two text
layouts are supported via `--bert-fields`:
- `concat` (entries 3-4c): fields joined into one string
  (`sample_name + description + keywords`) -> one 768-d embedding. Transcription
  excluded (too long; would truncate against the 512-token cap).
- `per_field` (entries 5-5c): each field encoded independently and the three
  768-d vectors hstacked -> 2304-d features; no text concatenation, no
  cross-field self-attention. Empty fields contribute an all-zeros block.
Embeddings are cached in `data/embeddings.npz`, keyed by sha256(model|text), so
the ~10 min CPU encode is paid once (re-runs ~10 s).

Evaluation for this model: `--cv 0` = single stratified 90/10 split, `--cv K` =
K-fold stratified CV (out-of-fold); both averaged over 3 seeds (42, 1, 7). CV is
affordable because encoding happens once and is cached — only the linear
classifier is refit per fold. Scores below are seed means +/- std.

| # | Model | Accuracy | Macro-F1 | Min per-class F1 | Note |
|---|-------|---------:|---------:|-----------------:|------|
| 3 | frozen Bio_ClinicalBERT + balanced LogReg, 90/10 x3 seeds | 0.547 | 0.547 | 0.297 | no augmentation; +/- 0.034 / 0.027 / 0.035 |
| 3b | 3 + back-translation, target=100 | 0.547 | 0.545 | 0.300 | augmentation ~no effect (+/- 0.039 / 0.032 / 0.036) |
| 4 | 3 with 5-fold CV, no augmentation x3 seeds | 0.550 | 0.564 | 0.378 | much tighter estimate; +/- 0.004 / 0.004 / 0.011 |
| 4b | 4 + back-translation, target=100 | 0.557 | 0.571 | 0.388 | aug now helps modestly; +/- 0.002 / 0.003 / 0.006 |
| 4c | 4 + back-translation, target=200 | 0.568 | 0.582 | 0.395 | previous best; +/- 0.000 / 0.001 / 0.003 |
| 5 | frozen Bio_ClinicalBERT + balanced LogReg, `--bert-fields per_field`, 5-fold CV x3 seeds | 0.592 | 0.605 | 0.440 | no augmentation; separate field embeddings beats joined text (4); +/- 0.002 / 0.002 / 0.008 |
| 5b | 5 + back-translation, target=100 | 0.597 | 0.611 | 0.456 | aug still helps; +/- 0.001 / 0.002 / 0.009 |
| 5c | 5 + back-translation, target=200 | **0.609** | **0.622** | **0.462** | **best result so far**; +/- 0.001 / 0.003 / 0.005 |

#### Notes on 3/3b

**Result.** Even without any fine-tuning, frozen clinical embeddings beat the
whole TF-IDF+NB family on every metric: acc 0.547 vs 0.499, macro-F1 0.547 vs
0.454, min-F1 0.297 vs 0.264 (vs entries 2b). The per-class floor is healthy:
the tiny classes (Dermatology, Psychiatry) sit around 0.33-0.67 thanks to
`class_weight='balanced'`, and Surgery is no longer dominant.

**Back-translation barely helps at 90/10 (3b).** Means are essentially unchanged
(acc 0.5471 both; macro 0.545 vs 0.547; min-F1 0.300 vs 0.297). Hypothesis: the
encoder is frozen and general-domain back-translated clinical text is
domain-shifted, so the added vectors are noisy rather than informative — unlike
the TF-IDF case, where augmentation directly added minority vocabulary.

**Caveat.** The 90/10 protocol (262 val rows) is not directly comparable to the
NB CV numbers, and class support in validation is tiny (Dermatology ~2-3 rows),
so the min-F1 estimate is noisy — hence the 3-seed averaging.

#### Notes on 4/4b/4c (cross-validation)

**Why CV.** The earlier objection ("encoding is the bottleneck") doesn't apply:
embeddings are cached, so a fold only refits the linear classifier. 5-fold CV
therefore costs seconds and evaluates on all 2613 rows out-of-fold, so the
estimate is far tighter (std ~0.003 vs ~0.03 at 90/10) and directly comparable
to the NB CV entries.

**Cross-validation changes the augmentation story.** At 90/10 the effect was
lost in noise; under CV it is small but consistent and monotone in the target:
acc 0.550 (no aug) -> 0.557 (target 100) -> 0.568 (target 200), with the same
ordering for macro-F1 (0.564 / 0.571 / 0.582) and min-F1 (0.378 / 0.388 / 0.395).
Two plausible reasons: (a) each fold now augments only ~80% of the data, so the
synthetic text is a smaller, less noisy fraction; (b) averaging over 5 folds × 3
seeds removes the variance that buried the effect at 90/10.

**Best model (4c, superseded by 5c).** 5-fold CV + back-translation target=200 ->
**acc 0.568, macro-F1 0.582, min-F1 0.395**, both graded bars met with margin.
Per-class floor: Dermatology 0.39, Neurosurgery 0.48, Surgery 0.43; the weakest
classes are Surgery (precision 0.87 but recall 0.28 — still many Surgery cases
predicted as body-part specialties) and Dermatology (0.39, precision-limited).

#### Notes on 5/5b/5c (per-field embeddings)

**Why per-field.** The joined-text embedding forces all fields through one
self-attention pass; splitting them gives each field its own vector, so the
classifier can weight `sample_name`/`description`/`keywords` independently
(mirrors the NB `--field all` FeatureUnion). Cold encode is ~3x the joined-text
one (three short fields), paid once and cached.

**Result.** Per-field beats joined text on every metric at every augmentation
level: no-aug acc 0.592 vs 0.550 (4), macro-F1 0.605 vs 0.564, min-F1 0.440 vs
0.378. The inter-field self-attention lost by dropping `concat` evidently cost
less than the gain from giving each field its own representation. Augmentation
still helps and is again monotone in the target (0.592 -> 0.597 -> 0.609 acc).

**Best model (5c, superseded by 6d).** `--model bert --bert-fields per_field --cv
5 --augment back_translation --augment-target 200` -> **acc 0.609, macro-F1 0.622,
min-F1 0.462**, both graded bars met with the largest margin so far. Per-class
floor: Dermatology 0.51, Neurosurgery 0.47, Surgery 0.49. Surgery is still
recall-limited (0.34) but its F1 rose from 0.43 to 0.49; Dermatology improved
most (0.39 -> 0.51). This was the strongest recipe found until Phase 3.

## Phase 1/2 - cleaning and post-cleaning baselines (2026-10-14)

**What changed.** `src/project_ln/data.py` now (a) exposes a single
`clean_text()` applied identically to train and test (NFC unicode normalisation,
typographic punctuation -> ASCII, stray `"` removed, whitespace collapsed; units
like `°`/`µ` and accents kept), (b) stitches records that a spreadsheet split
across physical lines (the cut is always mid-word inside `transcription`, so the
continuation is appended with no separator), (c) keeps the 7 label+description
rows by padding missing fields with `""`, and (d) adds `load_test()` for the
4-field unlabelled file (no header, in file order - that order defines
`results.txt`). Preprocessing is now identical for train and test as the PDF
requires.

**Row set.** train 2613 -> 2669 rows, all 12 specialties. Test is 398 records.
Blank-field counts (train): description 1, sample_name 7, transcription 31,
keywords 450.

**Data-integrity bug found (test file).** `test_no_labels.csv` at HEAD was
corrupted by an unresolved merge: 15 conflict-marker lines (`<<<<<<<`,
`=======`, `>>>>>>>`) and ~43 train rows (label-prefixed lines) spliced into it.
It was restored from commit `d3e2cad` ("Train and Test sets"), the last clean
revision, which holds exactly 398 four-field records with no labels and no
markers. `train.csv` was not affected. Any earlier results that predicted on the
corrupt file must be re-run.

**Effect of cleaning on scores.** Essentially none - it fixes correctness, not
accuracy. Whitespace/quote normalisation barely changes tokens, and the +56 rows
are noise-level. The small BERT dip is within ~1 std and is not a real
regression.

| # | Model | Rows | Accuracy | Macro-F1 | Min per-class F1 | Note |
|---|-------|-----:|---------:|---------:|-----------------:|------|
| 1d | nb `--field all --uniform-prior` (pre-clean) | 2613 | 0.501 | 0.262 | 0.000 | old reference |
| 1d-clean | same, after cleaning | 2669 | 0.4979 | 0.2621 | 0.0000 | unchanged within noise |
| 5c | bert `per_field` + bt target=200 (pre-clean) | 2613 | 0.609 | 0.622 | 0.462 | old best |
| 5c-clean | same, after cleaning | 2669 | 0.6063 | 0.6164 | 0.4469 | +/- 0.0017 / 0.0036 / 0.0031 (3 seeds) |

Per-class floor after cleaning (5c-clean, seed means): Dermatology ~0.46,
Neurosurgery ~0.45, Surgery ~0.49; Surgery is still recall-limited (0.34).
Both graded bars remain met with margin. **These are the new reference numbers
for Phase 3 (augmentation) and Phase 4 (group-aware CV).**

## Phase 3 - embedding-space augmentation (2026-10-15)

**What was added.** `src/project_ln/embed_augment.py`: three minority-only
generators that top every class below `target` up to `target` (same semantics
as back-translation's `augment_train_fold`), operating directly on bert's
already-encoded feature matrix - so no re-encoding, each run is seconds:
- `embed_noise`: `x_new = x_i + N(0, strength^2 * sigma^2)` with sigma the
  global per-feature std of the fold (a per-class std is unusable on ~25-row
  classes). `--augment-strength` (default 0.1).
- `embed_mixup`: within-class MixUp, `lambda ~ Beta(alpha, alpha)` (default
  alpha=0.4), `x_new = lambda*x_i + (1-lambda)*x_j`.
- `embed_smote`: SMOTE, interpolate toward one of the k nearest same-class
  neighbours (default k=5, clamped to `n_c - 1`). Self-implemented - no
  `imbalanced-learn` dependency.
- `bt_embed_smote`: chains back-translation (text, target=`--augment-target`)
  then SMOTE on the resulting embeddings (target=`--augment-target-2`,
  default `target + 100`).
Every synthetic vector is re-normalized to unit norm: the real embeddings live
on the unit hypersphere but interpolations/noise do not. All of it is CV-safe
(inside each fold's training partition only).

**Methodology.** 5-fold CV x 3 seeds, `--model bert --bert-fields per_field`,
post-cleaning 2669-row set. Reference: **5c-clean = 0.6063 / 0.6164 / 0.4469**.

| # | Augment | Params | Accuracy | Macro-F1 | Min F1 | vs 5c-clean |
|---|---------|--------|---------:|---------:|-------:|-------------|
| 6a | `embed_noise` | strength 0.1, target 200 | 0.6225 | 0.6097 | 0.3443 | acc up, min-F1 crashes |
| 6b | `embed_mixup` | alpha 0.4, target 200 | 0.6232 | 0.6109 | 0.3499 | same pattern |
| 6c | `embed_smote` | k 5, target 200 | 0.6212 | 0.6108 | 0.3559 | same pattern |
| 6d | `bt_embed_smote` | bt 200 -> smote 300, k 5 | 0.6261 | 0.6319 | 0.4623 | **all three up** |
| 6d | same | -> smote 400 | 0.6381 | 0.6388 | 0.4629 | |
| 6d | same | -> smote 500 | **0.6427** | **0.6397** | **0.4659** | **best floor; Surgery recall 0.34 -> 0.52** |
| 6d | same | -> smote 600 | 0.6482 | 0.6404 | 0.4532 | acc keeps rising, floor dips |
| 6d | same | -> smote 800 | 0.6568 | 0.6443 | 0.4511 | |
| 6d | same | -> smote 1000 | 0.6653 | 0.6488 | 0.4523 | |
| 6d | same | -> smote 1200 | 0.6718 | 0.6529 | 0.4554 | best acc/macro; see caveat |

k-sensitivity at smote target 400: k 3 -> 0.6364 / 0.6362 / 0.4642; k 5 ->
0.6381 / 0.6388 / 0.4629; k 10 -> not finished (too slow). k is basically
flat, so the target matters, not k.

**Conclusions.**
- Embedding-space augmentation *alone* (6a-6b-6c) trades minority behaviour
  for accuracy: +0.015 acc but min-F1 falls ~0.4469 -> ~0.35. Not usable on
  its own here.
- Chaining it *after* back-translation (6d) is where it pays: back-translation
  supplies text-level diversity in the minority classes, SMOTE densifies those
  regions, and every metric improves. Surgery's recall in particular jumps
  0.34 -> 0.52 (Surgery F1 0.49 -> 0.62) because the minority-heavy classes
  now have real mass near the decision boundary.
- Increasing the SMOTE target keeps raising accuracy monotonically (up to 1200,
  the largest tested), while min-F1 plateaus around 0.45-0.46. Caveat: past
  the fold's majority size (~714) the target also oversamples Surgery itself,
  which flattens `class_weight='balanced'` toward uniform weights - i.e. at
  very high targets the mechanism changes from "fix imbalance" to "uniform
  class mass", so the 800+ numbers should be read as a different regime, not a
  free lunch.

**New best (6d, smote target 500).** `--model bert --bert-fields per_field
--cv 5 --augment bt_embed_smote --augment-target 200 --augment-target-2 500`
-> **acc 0.6427, macro-F1 0.6397, min-F1 0.4659** (+/- 0.0023 / 0.0041 /
0.0164). Chosen over the 1200 variant because it keeps the best per-class
floor while still beating 5c-clean by +0.036 acc. Per-class (seed 42):
Dermatology 0.51, Neurosurgery 0.49, Surgery 0.62. Both graded bars met with a
large margin.

## Phase 3b - long-context transcription model (Track B, 2026-10-15)

**What was added.** `src/project_ln/longformer_embeddings.py` +
`--transcription-encoder {none,longformer}` (`main.py`): a fourth 768-d block
encoding the **full** `transcription` note with `yikuan8/Clinical-Longformer`
(4096-token window: sliding-window attention + global attention on CLS),
mean-pooled over non-pad tokens and L2-normalized, then hstacked onto the three
Bio_ClinicalBERT field blocks -> 4 x 768 = 3072-d. This replaces reading
`transcription` as a 512-token-truncated tail with reading all of it (98.3% of
notes fit; max is 3753 tokens).

**Cost / engineering notes.**
- One-time encode of 1975 unique transcriptions (train + test): **3.09 h**,
  cache `data/longformer_embeddings.npz` (5.7 MB, `sha256(model|text)`).
- CPU-bound: ~4-9 s/note on the i7-1165G7 (8 threads, `batch_size=4`,
  length-sorted batches to limit padding). bf16 was tested and is *slower*
  (13-19 s/note) - this CPU has AVX-512F but no AVX512-BF16.
- Because a single run would exceed the tool timeout, the cache is
  **checkpointed every 25 notes** and the run was launched with `nohup` +
  polled; an interrupted run resumes from the npz with zero loss. Cache hits
  cost ~2 ms, so every later experiment run is fast.

**Result: transcription adds nothing, across three augmentation regimes.**

| # | Transcription | Augmentation | Accuracy | Macro-F1 | Min F1 | vs reference |
|---|---------------|--------------|---------:|---------:|-------:|--------------|
| 7c | longformer (full note) | none | 0.5954 | 0.6062 | 0.4266 | vs 0.592 / 0.605 / 0.440 (5) |
| 7b | longformer (full note) | bt target=200 | 0.6085 | 0.6182 | 0.4422 | vs 0.6063 / 0.6164 / 0.4469 (5c-clean) |
| 7 | longformer (full note) | bt 200 -> smote 500 | 0.6389 | 0.6368 | 0.4689 | vs 0.6427 / 0.6397 / 0.4659 (6d) |

All three deltas are inside ~1-2 standard deviations: accuracy and macro-F1
move +0.002/-0.004 and min-F1 -0.013/+0.003, with no consistent sign. The
experiment is *neutral*, not a win: 6d stays the best recipe.

**Why (discussion for the paper).** `keywords` already names the specialty
nearly verbatim (first keyword == label for ~83% of train rows) and
`description` carries the procedure, so `transcription` - even read in full by
a 4096-token model instead of being truncated at 512 - is largely redundant.
The bottleneck is not how much of the note we read. Confirmed independently
below by entry 8, which keeps Bio_ClinicalBERT itself as the transcription
encoder.

## Phase 3c - Bio_ClinicalBERT + truncated transcription (entry 8, 2026-10-15)

**What was added.** `head_tail_truncate()` + `--transcription-mode
{none,head_tail}` in `bert_embeddings.py`/`main.py`: Bio_ClinicalBERT itself
encodes the `transcription` field, keeping the **first 254 + last 254**
wordpiece tokens (254+254+2 specials = 510 <= 512). Clinical notes put the
complaint / procedure on top and the impression / plan at the bottom, so both
diagnostic poles survive and only the boilerplate middle is dropped. Result is
a 4th 768-d block -> 3072-d, `per_field` only (concat would overflow 512).

Implementation notes: `convert_tokens_to_string` + `tokenize` is not an exact
round trip, so the re-encoded slice can grow a few tokens past the request -
`head_tail_truncate` shaves the tail until the result is <= 508 tokens (verified:
all 2638 non-empty notes fit, worst case 508). `encode_texts` now checkpoints
its cache every 100 new texts (a cold encode of the 1973 unique truncated notes
took ~31 min and was resumed once after hitting a tool timeout with zero loss).

**Result (entry 8), same protocol as 6d: 5-fold CV x 3 seeds, `per_field`,
`bt_embed_smote` 200 -> 500.**

| # | Transcription | Accuracy | Macro-F1 | Min F1 |
|---|---------------|---------:|---------:|-------:|
| **6d** | none | **0.6427** | **0.6397** | **0.4659** |
| 8 | Bio_ClinicalBERT, head+tail 508 tok | 0.6402 | 0.6379 | 0.4581 |
| 7 | Clinical-Longformer, whole note | 0.6389 | 0.6368 | 0.4689 |

Entry 8 loses to 6d on all three metrics (-0.0025 acc, -0.0018 macro, -0.0078
min-F1), each well inside one standard deviation (std 0.0018 / 0.0024 /
0.0150), and is mixed against entry 7. **All three transcription variants now
agree: the field adds no signal, whether ignored, truncated by the base model,
or read whole by a 4096-token model.** 6d remains the best recipe.

**Conclusion for the paper.** Transcription is not the missing signal; the
first-keyword shortcut in `keywords` plus the procedure named in `description`
already determine the specialty. Adding the note's full body only dilutes the
per-field representation (4 equal-norm blocks vs 3).

## Entry 9 - Submission artifact: `results.txt` (2026-10-15)

**What was produced.** `make_results.py` (repo root, self-contained, ~500
lines): reproduces recipe 6d end-to-end and writes `results.txt` - the file the
PDF requires for automatic evaluation (one predicted specialty per line, no
header, line order matching `test_no_labels.csv`). Trains on ALL 2669 rows of
`train.csv` (back-translation to 200/class, then embedding-space SMOTE to
500/class), fits one balanced LogisticRegression, predicts the 398 test rows.

**Design.** The script imports nothing from `project_ln` (only numpy,
scikit-learn, sentence-transformers), so it can be copied anywhere and run
standalone. It reuses `data/augmented_cache.json` (2846 back-translations,
619 KB) as an intermediate project dataset - the PDF explicitly allows
"eventual datasets gathered to the project" in the submission zip. Without
that file it falls back to live Google Translate (currently rate-limited on
this IP: HTTP 200 with an all-null payload). Embeddings are cached in
`data/embeddings.npz` (gitignored, regenerable). The 2 of 667 translations
that could not be fetched live fell back to the original text (negligible).

**Output.** `results.txt`: 398 lines, all 12 specialties, no empties.
Predicted distribution: Surgery 91 (22.9%), Cardiovascular-Pulmonary 48
(12.1%), Orthopedic 39 (9.8%), Gastroenterology 37 (9.3%), Neurology 32
(8.0%), General Medicine 30 (7.5%), Neurosurgery 28 (7.0%),
Obstetrics-Gynecology 27 (6.8%), Radiology 26 (6.5%), Ophthalmology 21
(5.3%), Psychiatry-Psychology 13 (3.3%), Dermatology 6 (1.5%). The test-set
distribution mirrors the training distribution (Surgery dominates both),
suggesting the model has not collapsed to a single class.
