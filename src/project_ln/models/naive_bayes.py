from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline

from ..augment import augment_train_fold
from ..data import TEXT_FIELDS, encode_labels, load_rows, texts_and_labels
from ..utils import make_stem_tokenizer, report

# Cap per field so 5-fold CV stays fast on the long transcription text.
MAX_FEATURES_PER_FIELD = 20000


class FieldSelector(BaseEstimator, TransformerMixin):
    """Pick one text field out of a list of row dicts."""

    def __init__(self, field="description"):
        self.field = field

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [row[self.field] for row in X]


def build_pipeline(fit_prior=True, per_field=False):
    if per_field:
        union = FeatureUnion(
            [
                (
                    name,
                    Pipeline(
                        [
                            ("select", FieldSelector(name)),
                            (
                                "tfidf",
                                TfidfVectorizer(
                                    analyzer=make_stem_tokenizer(),
                                    max_features=MAX_FEATURES_PER_FIELD,
                                ),
                            ),
                        ]
                    ),
                )
                for name in TEXT_FIELDS
            ]
        )
    else:
        union = TfidfVectorizer(analyzer=make_stem_tokenizer())
    return Pipeline([("tfidf", union), ("nb", MultinomialNB(fit_prior=fit_prior))])


def run_experiment(args):
    rows = load_rows()
    field = args.field
    fit_prior = not args.uniform_prior
    per_field = field == "all"
    texts, raw_labels = texts_and_labels(rows, field)
    labels, names = encode_labels(raw_labels)
    prior_str = "uniform prior" if args.uniform_prior else "learned prior"
    mode_str = "per-field union" if per_field else field
    aug_str = f", augment={args.augment}(target={args.augment_target})" if args.augment != "none" else ""
    print(f"loaded {len(texts)} rows, {len(names)} specialties (field={mode_str}, {prior_str}{aug_str})")

    if args.augment == "back_translation":
        # CV with augmentation inside each training fold only
        y = labels
        X = texts
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        all_preds = []
        all_trues = []
        for train_idx, val_idx in cv.split(X, y):
            X_train = [X[i] for i in train_idx]
            y_train = [y[i] for i in train_idx]
            X_val = [X[i] for i in val_idx]
            y_val = [y[i] for i in val_idx]
            X_train_aug, y_train_aug = augment_train_fold(X_train, y_train, target_count=args.augment_target)
            pipe = build_pipeline(fit_prior=fit_prior, per_field=per_field)
            pipe.fit(X_train_aug, y_train_aug)
            preds = pipe.predict(X_val)
            all_preds.extend(preds)
            all_trues.extend(y_val)
        from sklearn import metrics

        acc = metrics.accuracy_score(all_trues, all_preds)
        report_str = f"5-fold CV [TF-IDF + NB ({prior_str}), {mode_str}{aug_str}]"
        print("=====", report_str, "=====")
        print(f"accuracy={acc:.4f}  macroF1={metrics.f1_score(all_trues, all_preds, average='macro'):.4f}  minF1={metrics.f1_score(all_trues, all_preds, average=None).min():.4f}")
        print(metrics.classification_report(all_trues, all_preds, target_names=names, zero_division=0))
        return acc
    else:
        cv_pred = cross_val_predict(
            build_pipeline(fit_prior=fit_prior, per_field=per_field),
            texts,
            labels,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        )
        return report(labels, cv_pred, names, f"5-fold CV [TF-IDF + NB ({prior_str}), {mode_str}{aug_str}]")
