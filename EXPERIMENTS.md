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

**Best model (5c).** `--model bert --bert-fields per_field --cv 5 --augment
back_translation --augment-target 200` -> **acc 0.609, macro-F1 0.622, min-F1
0.462**, both graded bars met with the largest margin so far. Per-class floor:
Dermatology 0.51, Neurosurgery 0.47, Surgery 0.49. Surgery is still
recall-limited (0.34) but its F1 rose from 0.43 to 0.49; Dermatology improved
most (0.39 -> 0.51). This is the strongest recipe found to date.
