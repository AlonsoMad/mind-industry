import argparse
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from colorama import Fore, Style

from mind.pipeline.corpus import Chunk
from mind.pipeline.pipeline import MIND

def detect_model_from_filename(filename: str) -> str:
    """
    Infer the LLM model from the filename.
    """
    name = filename.lower()
    if "qwen" in name:
        return "qwen:32b"
    if "llama" in name:
        return "llama3.3:70b"
    if "gpt" in name:
        return "gpt-4o-2024-08-06"
    else:
        raise ValueError(f"Could not detect model from filename: {filename}")

def process_file(path_queries: str,
                 path_relevant: str,
                 raw: pd.DataFrame,
                 method_eval: str,
                 top_k: int,
                 topic: int) -> pd.DataFrame:
    """
    Process a single parquet file of questions and return a DataFrame of results.
    """
    df = pd.read_parquet(os.path. join(path_relevant, path_queries))
    df = df.drop_duplicates(subset=["question"], keep="first")

    llm_model = detect_model_from_filename(path_queries)
    mind = MIND(llm_model=llm_model, do_check_entailement=True)
    print(f"Initialized MIND with model {llm_model}")

    results = []

    for id_row, row in tqdm(df.iterrows(), total=len(df)):
        if id_row % 100 == 0:
            print(f"Processing row {id_row} with LLM {llm_model}")

        try:
            results_rel = row[method_eval]
            # Flatten nested list of lists: [[{doc_id, score}, ...], ...] -> [{...}, ...]
            flattened_list = [
                {"doc_id": entry["doc_id"], "score": entry["score"]}
                for subarray in results_rel
                for entry in subarray
            ]
            top_docs = [el["doc_id"] for el in flattened_list][:top_k]

            # Build source chunk from the current row
            chunk_a = Chunk(
                id=row.doc_id,
                text=row.passage,
                full_doc=row.full_doc,
                metadata=None,
            )

            # Generate answer in source language
            a_s, _ = mind._generate_answer(row.question, chunk_a)

            # Check entailment against the source passage
            _, _, entails = mind._check_entailement(a_s, chunk_a.text)

            if not entails:
                print(
                    f"{Fore.RED}Discarding question '{row.question}' since the answer does not entail the original passage."
                    f"{Style.RESET_ALL}\n ANSWER: {a_s}\nPASSAGE: {chunk_a.text}\n"
                )
                # still record one row per top_doc for consistency, but mark as N/A
                for top_doc in top_docs:
                    results.append(
                        {
                            "question_id": row.question_id,
                            "doc_id": top_doc,
                            "question": row.question,
                            "passage_s": chunk_a.text,
                            "answer_s": a_s,
                            "passage_t": "N/A",
                            "answer_t": "N/A",
                            "discrepancy": "N/A",
                            "reason": "N/A",
                        }
                    )
                continue

            # If entailed, evaluate against each top target doc
            for top_doc in top_docs:
                # Retrieve target passage/full_doc from the raw corpus
                target_rows = raw[raw.doc_id == top_doc]
                if target_rows.empty:
                    print(
                        f"{Fore.YELLOW}Warning: doc_id {top_doc} not found in source corpus. Skipping.{Style.RESET_ALL}"
                    )
                    continue

                passage_t = target_rows.text.iloc[0]
                full_doc_t = target_rows.full_doc.iloc[0]

                chunk_t = Chunk(
                    id=top_doc,
                    text=passage_t,
                    full_doc=full_doc_t,
                    metadata=None,
                )

                a_t, discrepancy_label, reason = mind._evaluate_pair(
                    question=row.question,
                    a_s=a_s,
                    source_chunk=chunk_a,
                    target_chunk=chunk_t,
                    topic=topic,
                    subquery=None,
                    save=False,
                )

                results.append(
                    {
                        "question_id": row.get("question_id", None),
                        "doc_id": top_doc,
                        "question": row.question,
                        "passage_s": chunk_a.text,
                        "answer_s": a_s,
                        "passage_t": chunk_t.text,
                        "answer_t": a_t,
                        "discrepancy": discrepancy_label,
                        "reason": reason,
                    }
                )

        except Exception as e:
            print(f"Error with question {row.get('question_id', 'UNKNOWN')}: {e}")
            continue

    df_results = pd.DataFrame(results)
    # Tag with model for aggregation later
    if not df_results.empty:
        df_results["model"] = llm_model

    return df_results


def main():
    parser = argparse.ArgumentParser(
        description="Run MIND QA/discrepancy evaluation for a given topic."
    )
    parser.add_argument(
        "--topic",
        type=int,
        required=True,
        help="Topic ID (integer), used to build input/output paths.",
    )
    parser.add_argument(
        "--method_eval",
        type=str,
        default="results_3_weighted",
        help="Column name containing retrieval results to evaluate.",
    )
    parser.add_argument(
        "--setting",
        type=str,
        default="thr__dynamic.parquet",
        help="Filename suffix used to filter the relevant files (e.g., 'thr__dynamic.parquet').",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top documents to evaluate per question.",
    )
    parser.add_argument(
        "--path_source",
        type=str,
        default="/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/passages/26_jan/df_1.parquet",
        help="Parquet file with the source corpus (must include columns: doc_id, text, full_doc).",
    )
    parser.add_argument(
        "--path_relevant",
        type=str,
        default="data/mind_runs/rosie/v1",
        help="Base directory where 'outs_good_model_tpc{topic}/relevant' lives.",
    )
    parser.add_argument(
        "--path_save",
        type=str,
        default="data/ablations/qa/v2",
        help="Base directory where outputs will be saved under 'outs_good_model_tpc{topic}/answers'.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/config.yaml",
        help="Path to a YAML config if needed by downstream utilities (kept for parity).",
    )

    args = parser.parse_args()

    topic = args.topic
    method_eval = args.method_eval
    setting = args.setting
    top_k = args.top_k
    path_relevant=args.path_relevant
    path_save=args.path_save

    print(f"Reading source data from {args.path_source}")
    raw = pd.read_parquet(args.path_source)
    print(f"Source data has {len(raw)} entries")
    print(f"HEAD: {raw.head(2)}")

    # Discover input files
    if not os.path.isdir(path_relevant):
        raise FileNotFoundError(f"Relevant path not found: {path_relevant}")

    paths_ = [
        p for p in os.listdir(path_relevant)
        if p.endswith(setting)
    ]
    print(f"Found {len(paths_)} files to process in {path_relevant}")
    if not paths_:
        print(f"{Fore.YELLOW}No files matching '*{setting}' were found. Exiting.{Style.RESET_ALL}")
        return
    
    # sort elements in path_queries so that if "qwen" is present, it is processed first
    if "qwen" in paths_:
        path_queries = [pq for pq in path_queries if "qwen" in pq] + [pq for pq in path_queries if "qwen" not in pq]

    # Ensure output dir exists
    Path(path_save).mkdir(parents=True, exist_ok=True)

    results_all = []

    for path_queries in paths_:
        print(f"Processing {path_queries}")
        df_results = process_file(
            path_queries=path_queries,
            path_relevant=path_relevant,
            raw=raw,
            method_eval=method_eval,
            top_k=top_k,
            topic=topic,
        )

        # Save per-file results
        out_file = os.path.join(path_save, path_queries)
        if not df_results.empty:
            df_results.to_parquet(out_file)
            # also save rolling intermediate by method_eval name (optional)
            df_results.to_parquet(os.path.join(path_save, f"{method_eval}.parquet"))
            results_all.append(df_results)
            print(f"Saved {len(df_results)} rows to {out_file}")
        else:
            print(f"{Fore.YELLOW}No results for {path_queries}{Style.RESET_ALL}")

    # Save concatenated results across models
    if results_all:
        df_all = pd.concat(results_all, ignore_index=True)
        all_out = os.path.join(path_save, f"all_models_eval_tpc{topic}.parquet")
        df_all.to_parquet(all_out)
        print(f"Saved aggregated results to {all_out}")
    else:
        print(f"{Fore.YELLOW}No aggregated results to save.{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
