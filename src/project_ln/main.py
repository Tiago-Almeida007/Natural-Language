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
        choices=["none", "back_translation"],
        default="none",
        help="data augmentation mode (applied to training partitions only, CV-safe)",
    )
    parser.add_argument(
        "--augment-target",
        type=int,
        default=100,
        help="target count per class when using augmentation",
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
    args = parser.parse_args()

    if args.model in ("nb", "naive_bayes"):
        models.naive_bayes.run_experiment(args)
    elif args.model == "bert":
        models.bert_embeddings.run_experiment(args)
    else:
        raise ValueError(f"unknown model {args.model}")


if __name__ == "__main__":
    main()
