import argparse
import time
from pathlib import Path

import pandas as pd
from mind.corpus_building.segmenter import Segmenter
from mind.corpus_building.translator import Translator
from mind.corpus_building.data_preparer import DataPreparer
from mind.utils.utils import init_logger
from wikipedia.retriever import WikiRetriever

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Wikipedia dataset construction.")
    parser.add_argument(
        "--output-path", type=str, required=True,
        help="Directory where all files are saved (created if missing).")
    parser.add_argument(
        "--seed-query", type=str, default="George Washington")
    parser.add_argument(
        "--seed-lang", type=str, default="en")
    parser.add_argument(
        "--target-lang", type=str, default="de")
    parser.add_argument(
        "--ndocs", type=int, default=2)
    parser.add_argument(
        "--alignment", type=int, default=1)
    parser.add_argument(
        "--max-tries", type=int, default=8)
    parser.add_argument(
        "--base-name", type=str, default="wiki_corpus",
        help="Base filename for all artifacts.")
    parser.add_argument(
        "--text-col", type=str, default="text")
    parser.add_argument(
        "--lang-col", type=str, default="lang")
    parser.add_argument(
        "--min-length", type=int, default=100)
    parser.add_argument(
        "--sep", type=str, default="\n")
    parser.add_argument(
        "-v", "--verbose", action="count", default=1,
        help="Increase logging verbosity (-v, -vv).")
    parser.add_argument(
        "--config_path", type=str, default="config/config.yaml",
        help="Path to the configuration file.")
    args = parser.parse_args()

    # logger and output dir
    logger = init_logger(Path(args.config_path), "WikipediaDataset")
    output_dir = Path(args.output_path).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    # retrieval (raw)
    raw_parquet_path = output_dir / \
        f"{args.base_name}_raw_s{args.seed_lang}_t{args.target_lang}_n{args.ndocs}.parquet.gzip"
    if raw_parquet_path.exists():
        logger.info(
            "Raw file already exists, skipping retrieval: %s", raw_parquet_path)
        df_raw = pd.read_parquet(raw_parquet_path)
    else:
        retriever = WikiRetriever(
            file_path=str(output_dir),
            trgt_lan=args.target_lang,
            seed_lan=args.seed_lang,
            seed_query=args.seed_query,
            ndocs=args.ndocs,
        )
        attempt = 0
        while True:
            try:
                logger.info(
                    "Starting Wikipedia retrieval (attempt %d)…", attempt + 1)
                retriever.retrieval(alignment=args.alignment)
                break
            except Exception as e:
                attempt += 1
                logger.warning("Retrieval failed: %s", e)
                if attempt >= args.max_tries:
                    logger.error("Max retries reached. Exiting.")
                    raise
                sleep_s = min(60, 2 ** attempt)
                logger.info("Retrying in %d seconds…", sleep_s)
                time.sleep(sleep_s)

        file_out = retriever.df_to_parquet()
        Path(file_out).rename(raw_parquet_path)
        logger.info("Saved RAW: %s", raw_parquet_path)
        df_raw = pd.read_parquet(raw_parquet_path)

    # segmentation (both languages)
    seg_all_path = output_dir / \
        f"{args.base_name}_seg_all_s{args.seed_lang}_t{args.target_lang}_n{args.ndocs}.parquet.gzip"
    if seg_all_path.exists():
        logger.info("Segmented file already exists, skipping: %s", seg_all_path)
    else:
        logger.info("Segmenting the dataset…")
        seg = Segmenter(config_path=args.config_path)
        seg.segment(
            path_df=raw_parquet_path,
            path_save=seg_all_path,
            text_col=args.text_col,
            min_length=args.min_length,
            sep=args.sep,
        )
        logger.info("Saved SEGMENTED (all langs): %s", seg_all_path)

    seg_df = pd.read_parquet(seg_all_path)

    # split segmented files per language
    seg_seed_path = output_dir / \
        f"{args.base_name}_seg_{args.seed_lang}_s{args.seed_lang}_t{args.target_lang}_n{args.ndocs}.parquet.gzip"
    seg_trgt_path = output_dir / \
        f"{args.base_name}_seg_{args.target_lang}_s{args.seed_lang}_t{args.target_lang}_n{args.ndocs}.parquet.gzip"

    if args.lang_col in seg_df.columns:
        if not seg_seed_path.exists():
            seg_df.query(f"{args.lang_col} == @args.seed_lang").to_parquet(
                seg_seed_path, compression="gzip", index=False)
            logger.info("Saved SEGMENTED: %s", seg_seed_path)
        else:
            logger.info(
                "Per-language segmented (seed) already exists: %s", seg_seed_path)

        if not seg_trgt_path.exists():
            seg_df.query(f"{args.lang_col} == @args.target_lang").to_parquet(
                seg_trgt_path, compression="gzip", index=False)
            logger.info("Saved SEGMENTED: %s", seg_trgt_path)
        else:
            logger.info(
                "Per-language segmented (target) already exists: %s", seg_trgt_path)
    else:
        logger.warning(
            "Column '%s' not found; skipping per-language segmented saves.", args.lang_col)

    # translation (seed->target and target->seed)
    trans = Translator(config_path=args.config_path)

    for lang, path in [(args.seed_lang, Path(seg_seed_path)), (args.target_lang, Path(seg_trgt_path))]:
        src = lang
        tgt = args.target_lang if lang == args.seed_lang else args.seed_lang

        out_path = output_dir / \
            f"{args.base_name}_trans_s{src}_t{tgt}_n{args.ndocs}.parquet.gzip"

        translated_concat = trans.translate(
            path_df=path,
            src_lang=src,
            tgt_lang=tgt,
            text_col=args.text_col,
            lang_col=args.lang_col,
            save_path=out_path,  # we control filenames outside
        )
        logger.info("Saved TRANSLATED (%s→%s): %s", src, tgt, out_path)

    # create final PLTM dataset
    prep = DataPreparer(
        preproc_script="externals/NLPipe/src/nlpipe/cli.py",
        config_path="config.json",
        stw_path="externals/NLPipe/src/nlpipe/stw_lists",
        spacy_models={
            "en": "en_core_web_sm",
            "de": "de_core_news_sm",
            "es": "es_core_news_sm"},
        schema={
            "chunk_id": "id_preproc",
            "doc_id": "id",
            "text": "text",
            "full_doc": "summary",
            "lang": "lang",
            "title": "title",
            "url": "url",
            "equivalence": "equivalence",
        },
    )

    prep.format_dataframes(
        anchor_path=str(output_dir / f"{args.base_name}_trans_s{args.seed_lang}_t{args.target_lang}_n{args.ndocs}.parquet.gzip"),
        comparison_path=str(output_dir / f"{args.base_name}_trans_s{args.target_lang}_t{args.seed_lang}_n{args.ndocs}.parquet.gzip"),
        path_save=str(output_dir / f"{args.base_name}_polylingual_s{args.seed_lang}_t{args.target_lang}_n{args.ndocs}.parquet.gzip")
    )
