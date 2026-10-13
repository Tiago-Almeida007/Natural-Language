import re

from sklearn.metrics import accuracy_score, classification_report, f1_score


def make_stem_tokenizer():
    """Build the tokenizer that matches the PDF baseline.

    Each document is lowercased, split into words, and every word is
    Porter-stemmed (running/runs -> run) so different forms share one
    vocabulary entry.
    """
    from nltk.stem import PorterStemmer

    stemmer = PorterStemmer()

    def stem_tokenizer(text):
        return [stemmer.stem(word) for word in re.findall(r"[a-z0-9]+", text.lower())]

    return stem_tokenizer


def report(y_true, y_pred, names, header):
    acc = accuracy_score(y_true, y_pred)
    per_class = f1_score(y_true, y_pred, average=None, labels=range(len(names)), zero_division=0)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"\n===== {header} =====")
    print(f"accuracy={acc:.4f}  macroF1={macro:.4f}  minF1={per_class.min():.4f}")
    print(classification_report(y_true, y_pred, labels=range(len(names)),
                                target_names=names, digits=3, zero_division=0))
    return acc, macro, per_class.min()
