import re

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


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


def report(y_true, y_pred, names, header, out_path=None, plot_path=None):
    acc = accuracy_score(y_true, y_pred)
    per_class = f1_score(y_true, y_pred, average=None, labels=range(len(names)), zero_division=0)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"\n===== {header} =====")
    print(f"accuracy={acc:.4f}  macroF1={macro:.4f}  minF1={per_class.min():.4f}")
    print(classification_report(y_true, y_pred, labels=range(len(names)),
                                target_names=names, digits=3, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=range(len(names)))
    if out_path is not None:
        _write_confusion_report(out_path, cm, y_true, y_pred, names, header, acc, macro, per_class)
        print(f"[confusion] written to {out_path}")
    if plot_path is not None:
        _plot_confusion_png(plot_path, cm, names, header, acc, macro, per_class)
        print(f"[confusion] written to {plot_path}")
    return acc, macro, per_class.min()


def _write_confusion_report(path, cm, y_true, y_pred, names, header, acc, macro, per_class):
    """Write the per-label metrics and the confusion matrix to a text file.

    Rows are true labels, columns are predicted labels, both in the alphabetical
    `names` order. Compact column headers (c0..c11) keep the 12x12 grid readable;
    the legend maps each index back to a specialty.
    """
    n = len(names)
    report_dict = classification_report(
        y_true, y_pred, labels=range(n), target_names=names, digits=3,
        zero_division=0, output_dict=True,
    )
    lines = [f"===== {header} =====",
             f"accuracy={acc:.4f}  macroF1={macro:.4f}  minF1={per_class.min():.4f}",
             "",
             "Per-label metrics:",
             f"{'label':<26}{'prec':>8}{'recall':>8}{'f1':>8}{'support':>9}",
             "-" * 59]
    for name in names:
        m = report_dict[name]
        lines.append(f"{name:<26}{m['precision']:>8.3f}{m['recall']:>8.3f}{m['f1-score']:>8.3f}{int(m['support']):>9}")
    lines.append("")
    lines.append("Confusion matrix (rows=true, cols=predicted):")
    lines.append(f"{'':<26}" + "".join(f"{f'c{i}':>6}" for i in range(n)))
    for i, name in enumerate(names):
        lines.append(f"{name:<26}" + "".join(f"{cm[i, j]:>6}" for j in range(n)))
    lines.append("")
    lines.append("Legend:")
    for i, name in enumerate(names):
        lines.append(f"  c{i:<2} = {name}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _plot_confusion_png(path, cm, names, header, acc, macro, per_class):
    """Render a publication-quality 12x12 confusion matrix heatmap as a PNG.

    Rows are true labels, columns are predicted labels, both in alphabetical
    `names` order. Saved at 300 DPI so it can be dropped into the paper PDF.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(names)
    fig, ax = plt.subplots(figsize=(11, 9), dpi=300)

    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=range(n),
        yticks=range(n),
        xticklabels=names,
        yticklabels=names,
        ylabel="True label",
        xlabel="Predicted label",
        title=f"Confusion matrix (rows=true, cols=predicted)\n"
              f"{header} — accuracy={acc:.4f}, macro-F1={macro:.4f}, min-F1={per_class.min():.4f}",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate each cell with its count, flipping text colour on dark cells.
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center", fontsize=7,
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
