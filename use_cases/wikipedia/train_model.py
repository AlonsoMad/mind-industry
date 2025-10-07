import argparse
import pathlib
import time

from mind.topic_modeling.polylingual_tm import PolylingualTM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Polylingual Topic Models with different numbers of topics",
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to the input parquet file containing the polylingual dataset"
    )

    parser.add_argument(
        "--lang1",
        type=str,
        default="EN",
        help="First language code (e.g., EN, ES, DE)"
    )

    parser.add_argument(
        "--lang2",
        type=str,
        default="DE",
        help="Second language code (e.g., EN, ES, DE)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Base directory where models will be saved"
    )

    parser.add_argument(
        "--num-topics",
        type=int,
        nargs="+",
        default=[5, 10, 15, 20, 25, 30, 40, 50],
        help="List of topic numbers to train models for (space-separated)"
    )

    parser.add_argument(
        "--model-prefix",
        type=str,
        default="poly",
        help="Prefix for model folder names"
    )

    parser.add_argument(
        "--date-suffix",
        action="store_true",
        help="Add date suffix to model folder names"
    )

    return parser.parse_args()


def main():
    """Main function to train polylingual topic models."""
    args = parse_args()

    print(f"Training Polylingual Topic Models")
    print(f"Input data: {args.input}")
    print(f"Languages: {args.lang1} - {args.lang2}")
    print(f"Output directory: {args.output_dir}")
    print(f"Topic numbers: {args.num_topics}")

    # Create date suffix if requested
    date_suffix = f"_{time.strftime('%d_%m_%y')}" if args.date_suffix else ""

    # Train models for each number of topics
    for k in args.num_topics:
        print(f"\n-- Training model with {k} topics --")

        # Create model folder name
        lang_pair = f"{args.lang1.lower()}_{args.lang2.lower()}"
        model_folder = pathlib.Path(
            args.output_dir) / f"{args.model_prefix}_{lang_pair}{date_suffix}_{k}"

        print(f"Model folder: {model_folder}")

        # Initialize and train model
        model = PolylingualTM(
            lang1=args.lang1,
            lang2=args.lang2,
            model_folder=model_folder,
            num_topics=k
        )

        model.train(args.input)
        print(f"Model with {k} topics completed")

    print(f"\nAll models trained successfully!")


if __name__ == "__main__":
    main()
