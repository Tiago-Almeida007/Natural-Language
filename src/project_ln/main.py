import argparse

from . import models


def main():
    parser = argparse.ArgumentParser(prog="project_ln.main")
    parser.add_argument(
        "--model",
        choices=["nb", "naive_bayes", "bert"],
        default="nb",
        help="model to run (naive_bayes/nb = TF-IDF+NB; bert = frozen Bio_ClinicalBERT embeddings + logistic regression)",
    )
    parser.add_argument(
        "--field",
        choices=["description", "sample_name", "transcription", "keywords", "all"],
        default="description",
        help="text field fed to the model; 'all' vectorizes every field separately via a feature union (no concatenation)",
    )
    parser.add_argument(
        "--uniform-prior",
        action="store_true",
        help="use uniform class priors (fit_prior=False) instead of learned priors",
    )
    parser.add_argument(
        "--augment",
        choices=["none", "back_translation", "embed_noise", "embed_mixup", "embed_smote", "bt_embed_smote"],
        default="none",
        help="data augmentation mode (applied to training partitions only, CV-safe). "
        "'back_translation' works on text (nb + bert); 'embed_*' operate in embedding "
        "space on bert's feature matrix only (no re-encoding); 'bt_embed_smote' chains "
        "the two (back-translation first, then SMOTE on the resulting embeddings).",
    )
    parser.add_argument(
        "--augment-target",
        type=int,
        default=100,
        help="target count per class when using augmentation",
    )
    parser.add_argument(
        "--augment-target-2",
        type=int,
        default=0,
        help="second-stage target for --augment bt_embed_smote (0 = augment-target + 100)",
    )
    parser.add_argument(
        "--augment-strength",
        type=float,
        default=0.1,
        help="embed_noise: noise scale, in units of the per-feature std of the fold",
    )
    parser.add_argument(
        "--augment-alpha",
        type=float,
        default=0.4,
        help="embed_mixup: Beta(alpha, alpha) shape for the mixup coefficient",
    )
    parser.add_argument(
        "--augment-k",
        type=int,
        default=5,
        help="embed_smote: number of same-class neighbours to interpolate toward",
    )
    parser.add_argument(
        "--augment-seed",
        type=int,
        default=0,
        help="seed for embedding-space augmentation (independent of the split seeds)",
    )
    parser.add_argument(
        "--bert-model",
        default="emilyalsentzer/Bio_ClinicalBERT",
        help="Hugging Face encoder id used by --model bert (frozen feature extractor)",
    )
    parser.add_argument(
        "--bert-fields",
        choices=["concat", "per_field"],
        default="concat",
        help="--model bert text layout: 'concat' joins sample_name+description+keywords "
        "into one text (one embedding); 'per_field' encodes each field separately "
        "(768-d each) and hstacks the vectors, no text concatenation, transcription excluded",
    )
    parser.add_argument(
        "--transcription-encoder",
        choices=["none", "longformer"],
        default="none",
        help="--model bert: how to encode the long `transcription` field. 'none' (default) "
        "excludes it (Bio_ClinicalBERT only reads 512 tokens). 'longformer' encodes the full "
        "note (4096-token window) with yikuan8/Clinical-Longformer and appends a 4th 768-d "
        "block, for --bert-fields per_field only. First run encodes ~2638 notes in ~6h "
        "(cached in data/longformer_embeddings.npz, resumable).",
    )
    parser.add_argument(
        "--transcription-mode",
        choices=["none", "head_tail"],
        default="none",
        help="--model bert: read the long `transcription` field with Bio_ClinicalBERT itself. "
        "'none' (default) excludes it. 'head_tail' keeps the first 254 and last 254 tokens "
        "(clinical notes put complaint/procedure on top, impression/plan at the bottom) so a "
        "~650-token note fits the 512-token window; adds a 4th 768-d block. "
        "for --bert-fields per_field only, and not together with --transcription-encoder.",
    )
    parser.add_argument(
        "--seeds",
        default="42,1,7",
        help="comma-separated split seeds for --model bert (single-split evaluation)",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=0,
        help="--model bert: 0 = single 90/10 split; K>1 = K-fold stratified CV (cheap: embeddings are cached)",
    )
    parser.add_argument(
        "--confusion-out",
        default=None,
        help="path to write the per-label metrics and the confusion matrix (rows=true, cols=predicted) as a text file; default off",
    )
    parser.add_argument(
        "--confusion-plot",
        default=None,
        help="path to write the confusion matrix as a 300-DPI PNG heatmap; default off",
    )
    parser.add_argument(
        "--error-dump",
        default=None,
        help="path to write misclassified out-of-fold rows (tsv: true, pred, sample_name, description, keywords); --model bert only",
    )
    args = parser.parse_args()

    if args.model in ("nb", "naive_bayes") and args.augment.startswith("embed"):
        parser.error(f"--augment {args.augment} needs --model bert (it works on embeddings)")
    if args.model in ("nb", "naive_bayes") and args.augment == "bt_embed_smote":
        parser.error("--augment bt_embed_smote needs --model bert")
    if args.transcription_encoder == "longformer":
        if args.model in ("nb", "naive_bayes"):
            parser.error("--transcription-encoder needs --model bert")
        if args.bert_fields != "per_field":
            parser.error("--transcription-encoder longformer needs --bert-fields per_field")
    if args.transcription_mode == "head_tail":
        if args.model in ("nb", "naive_bayes"):
            parser.error("--transcription-mode needs --model bert")
        if args.bert_fields != "per_field":
            parser.error("--transcription-mode head_tail needs --bert-fields per_field")
    if args.transcription_mode == "head_tail" and args.transcription_encoder != "none":
        parser.error("--transcription-mode and --transcription-encoder are mutually exclusive")

    if args.model in ("nb", "naive_bayes"):
        models.naive_bayes.run_experiment(args)
    elif args.model == "bert":
        models.bert_embeddings.run_experiment(args)
    else:
        raise ValueError(f"unknown model {args.model}")


if __name__ == "__main__":
    main()
