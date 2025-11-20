import argparse
import json
from typing import List

from mind.pipeline.pipeline import MIND


def comma_separated_ints(value: str) -> List[int]:
    try:
        return [int(v.strip()) for v in value.split(",") if v.strip() != ""]
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Topics must be comma-separated integers.")


def build_parser() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser(
        description="CLI for running MIND pipeline."
    )

    # Common options
    argparser.add_argument(
        "--llm_model", default="qwen:32b", help="LLM model (default: qwen:32b)")
    argparser.add_argument(
        "--llm_server", default=None, type=str,
        help="LLM server URL, e.g. http://localhost:11434")
    argparser.add_argument(
        "--topics", type=comma_separated_ints, required=True,
        help="Comma-separated topic IDs, e.g. '17' or '15,17'.")
    argparser.add_argument(
        "--sample_size", type=int, default=None,
        help="Optional sample size to subsample passages.")
    argparser.add_argument(
        "--path_save", type=str, required=True,
        help="Directory where results will be saved.")
    argparser.add_argument(
        "--dry_run", action="store_true",
        help="If set, runs without writing outputs.")
    argparser.add_argument(
        "--no_entailment", action="store_true",
        help="Disable entailment checking.")

    # Source corpus
    argparser.add_argument("--src_corpus_path", required=True, type=str)
    argparser.add_argument("--src_thetas_path", required=True, type=str)
    argparser.add_argument("--src_id_col", default="doc_id")
    argparser.add_argument("--src_passage_col", default="text")
    argparser.add_argument("--src_full_doc_col", default="full_doc")
    argparser.add_argument("--src_lang_filter", default="EN")
    argparser.add_argument("--src_filter_ids_path", default=None, type=str, help="Path to a file with IDs to filter out from the source corpus.")
    argparser.add_argument("--previous_check", default=None, type=str, help="Path to a file with IDs to filter out from the source corpus based on previous checks.")

    # Target corpus
    argparser.add_argument("--tgt_corpus_path", required=True, type=str)
    argparser.add_argument("--tgt_thetas_path", required=True, type=str)
    argparser.add_argument("--tgt_id_col", default="doc_id")
    argparser.add_argument("--tgt_passage_col", default="text")
    argparser.add_argument("--tgt_full_doc_col", default="full_doc")
    argparser.add_argument("--tgt_lang_filter",
                           required=True, help="e.g. ES or DE")
    argparser.add_argument("--tgt_index_path", default=None, type=str,
                           help="Path for saving the indexes.")
    argparser.add_argument("--tgt_filter_ids_path", default=None, type=str,
                           help="Path to a file with IDs to filter out from the target corpus.")
    argparser.add_argument("--load_thetas", action="store_true",
                           help="If set, loads the thetas from file (otherwise, assumes they are already in memory).")
    argparser.add_argument("--print_config", action="store_true",
                           help="Print resolved configuration before running.")

    return argparser


def main():
    args = build_parser().parse_args()
    
    src_filter_ids, tgt_filter_ids = None, None
    if args.src_filter_ids_path:
        with open(args.src_filter_ids_path, "r") as f:
            src_filter_ids = [line.strip() for line in f if line.strip() != ""]
    if args.tgt_filter_ids_path:
        with open(args.tgt_filter_ids_path, "r") as f:
            tgt_filter_ids = [line.strip() for line in f if line.strip() != ""]

    source_corpus = {
        "corpus_path": args.src_corpus_path,
        "thetas_path": args.src_thetas_path,
        "id_col": args.src_id_col,
        "passage_col": args.src_passage_col,
        "full_doc_col": args.src_full_doc_col,
        "language_filter": args.src_lang_filter,
        "filter_ids": src_filter_ids,
        "load_thetas": args.load_thetas
    }

    target_corpus = {
        "corpus_path": args.tgt_corpus_path,
        "thetas_path": args.tgt_thetas_path,
        "id_col": args.tgt_id_col,
        "passage_col": args.tgt_passage_col,
        "full_doc_col": args.tgt_full_doc_col,
        "language_filter": args.tgt_lang_filter,
        "filter_ids": tgt_filter_ids,
        "load_thetas": args.load_thetas
    }
    if args.tgt_index_path:
        target_corpus["index_path"] = args.tgt_index_path

    cfg = {
        "llm_model": args.llm_model,
        "llm_server": args.llm_server,
        "source_corpus": source_corpus,
        "target_corpus": target_corpus,
        "dry_run": bool(args.dry_run),
        "do_check_entailement": not args.no_entailment,
    }

    if args.print_config:
        print(json.dumps(cfg, indent=2))

    mind = MIND(**cfg)

    run_kwargs = {
        "topics": args.topics, 
        "path_save": args.path_save,
        "previous_check": args.previous_check
    }
    
    if args.sample_size is not None:
        run_kwargs["sample_size"] = args.sample_size

    mind.run_pipeline(**run_kwargs)


if __name__ == "__main__":
    main()
