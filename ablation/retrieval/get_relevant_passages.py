# test_four_methods_with_indexretriever.py

import argparse
from functools import partial
import os
import ast
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from mind.pipeline.retriever import IndexRetriever
from mind.pipeline.utils import get_doc_top_tpcs  

def build_or_load_retriever(method: str, model: SentenceTransformer, text_col:str, id_col:str) -> IndexRetriever:
    """
    Creates an IndexRetriever configured for a given method and weighting mode,
    then builds or loads indices using .build_or_load_index().
    """
    r = IndexRetriever(
        model=model,
        top_k=TOP_K,
        batch_size=BATCH_SIZE,
        min_clusters=MIN_CLUSTERS,
        nprobe=NPROBE,
        nprobe_fixed=False,
        do_norm=True,
    )
    r.build_or_load_index(
        source_path=str(PATH_SOURCE),
        thetas_path=str(PATH_THETAS_LANG),
        save_path_parent=str(INDICES_ROOT),
        method=method,
        col_to_index=text_col,
        col_id=id_col,
        lang=LANG,
        thr_assignment=EPSILON_INDEX
    )
    return r


def run_methods_for_query(
    retrievers,
    query_text: str,
    theta_query = None,
    thrs_opt = None,
    topic_based = False
):
    """
    Runs ENN, ANN, TB-ENN (weighted/unweighted), TB-ANN (weighted/unweighted); stores results and times.
    """
    
    out = {}
    if topic_based: 
        for tag, key_time, key_results in [
            ("TB-ENN-W", "time_3_weighted", "results_3_weighted"),
            ("TB-ENN", "time_3_unweighted", "results_3_unweighted"),
            ("TB-ANN-W", "time_4_weighted", "results_4_weighted"),
            ("TB-ANN", "time_4_unweighted", "results_4_unweighted"),
        ]:
            t0 = time.time()
            print(f"Running {tag}...")
            tag_ = tag.replace("-W", "").replace("-U", "")
            do_weighting = tag.endswith("-W")
            res, _ = retrievers[tag_].retrieve(
                query=query_text,
                theta_query=theta_query,
                thrs_opt=thrs_opt,
                do_weighting=do_weighting
                )
            t1 = time.time()
            out[key_results] = res
            out[key_time] = t1 - t0
    else:
            for tag, key_time, key_results in [
                ("ENN", "time_1", "results_1"),
                ("ANN", "time_2", "results_2"),
            ]:
                t0 = time.time()
                print(f"Running {tag}...")
                print(f"Quering with {tag}")
                res, _ = retrievers[tag].retrieve(
                    query=query_text,
                    theta_query=theta_query,
                    )
                t1 = time.time()
                out[key_results] = res
                out[key_time] = t1 - t0
    return out


def main():
    print(f"Loading SentenceTransformer model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    print(f"Creating retrievers...")
    fixed_params = {"model": model, "text_col": TEXT_COL, "id_col": ID_COL}
    factory = partial(build_or_load_retriever, **fixed_params)

    retrievers = {
        kind: factory(kind)
        for kind in ("ENN", "ANN", "TB-ENN", "TB-ANN")
    }
    print("Retrievers ready.")
    
    thetas_anchor = sparse.load_npz(PATH_THETAS_ANCHOR).toarray()

    # Load corpus df to map doc_id -> text (match your original mapping)
    raw = pd.read_parquet(PATH_SOURCE)
    raw_en = raw[raw["lang"] == "EN"].copy()
    if "lang" in raw.columns:
        raw = raw[raw["lang"] == LANG].copy()
    else:
        raw = raw[raw[ID_COL].astype(str).str.contains(LANG)].copy()
        
    # thetas for query
    docid_to_theta_anchor = dict(zip(raw_en[ID_COL].tolist(), thetas_anchor))

    # adapt to the original ID_COL and TEXT_COL
    doc_map = raw.set_index(ID_COL)[TEXT_COL].to_dict()

    # iterate over excel files with questions; there is one file per LLM
    xlsx_files = [f for f in os.listdir(PATH_QUERIES_DIR) if f.endswith(".xlsx")]
    for filename in xlsx_files:
        df_q = pd.read_excel(PATH_QUERIES_DIR / filename)

        # Ensure result/time columns exist (theta_10 matches your original: created but not populated)
        for c in [
            "results_1", "results_2",
            "results_3_weighted", "results_3_unweighted",
            "results_4_weighted", "results_4_unweighted",
            "time_1", "time_2", "time_3_weighted", "time_3_unweighted", "time_4_weighted", "time_4_unweighted", "theta_10"
        ]:
            if c not in df_q.columns:
                df_q[c] = None
        
        # do pass for not topic-based methods
        print("Calculating results for non-topic-based methods (ENN, ANN)...")
        for ridx, row in tqdm(df_q.iterrows(), total=len(df_q)):
            queries = row.get("queries")
            if isinstance(queries, str):
                queries = ast.literal_eval(queries)
            if not isinstance(queries, list):
                queries = [queries]
                
            r1_all, r2_all = [], []
            t1_all, t2_all = [], []

            for q in queries:
                out = run_methods_for_query(retrievers, q, topic_based=False)
                r1_all.append(out["results_1"])
                r2_all.append(out["results_2"])

                t1_all.append(out["time_1"])
                t2_all.append(out["time_2"])

                print(f"Exact NN: {out['time_1']:.2f}s, Approx NN: {out['time_2']:.2f}s")

            # write per-row
            df_q.at[ridx, "results_1"] = r1_all
            df_q.at[ridx, "results_2"] = r2_all

            df_q.at[ridx, "time_1"] = float(np.mean(t1_all)) if t1_all else None
            df_q.at[ridx, "time_2"] = float(np.mean(t2_all)) if t2_all else None
        
        # run for topic-based methods with different thresholding options
        print("Calculating results for topic-based methods (TB-ENN, TB-ANN)...")
        # Two passes: thresholds=None and thresholds=dynamic
        for thrs_opt, thr_tag in [(None, ""), ("var", "_dynamic")]:
            print(f"Calculating results with thresholds: {'dynamic' if thrs_opt is not None else 'None'}")

            for ridx, row in tqdm(df_q.iterrows(), total=len(df_q)):
                
                theta = docid_to_theta_anchor.get(row["doc_id"])
                theta_query = get_doc_top_tpcs(theta, topn=TOP_K) if theta is not None else []
                
                # Parse list of queries
                queries = row.get("queries")
                if isinstance(queries, str):
                    queries = ast.literal_eval(queries)
                if not isinstance(queries, list):
                    queries = [queries]

                r3w_all, r3u_all, r4w_all, r4u_all = [], [], [], []
                t3w_all, t3u_all, t4w_all, t4u_all = [], [], [], []

                for q in queries:
                    out = run_methods_for_query(retrievers, q, theta_query, thrs_opt, topic_based=True)
                    r3w_all.append(out["results_3_weighted"])
                    r3u_all.append(out["results_3_unweighted"])
                    r4w_all.append(out["results_4_weighted"])
                    r4u_all.append(out["results_4_unweighted"])

                    t3w_all.append(out["time_3_weighted"])
                    t3u_all.append(out["time_3_unweighted"])
                    t4w_all.append(out["time_4_weighted"])
                    t4u_all.append(out["time_4_unweighted"])

                    print(f"TB-ENN-W: {out['time_3_weighted']:.2f}s, TB-ENN-U: {out['time_3_unweighted']:.2f}s, "
                          f"TB-ANN-W: {out['time_4_weighted']:.2f}s, TB-ANN-U: {out['time_4_unweighted']:.2f}s")

                df_q.at[ridx, "results_3_weighted"] = r3w_all
                df_q.at[ridx, "results_3_unweighted"] = r3u_all
                df_q.at[ridx, "results_4_weighted"] = r4w_all
                df_q.at[ridx, "results_4_unweighted"] = r4u_all

                df_q.at[ridx, "time_3_weighted"] = float(np.mean(t3w_all)) if t3w_all else None
                df_q.at[ridx, "time_3_unweighted"] = float(np.mean(t3u_all)) if t3u_all else None
                df_q.at[ridx, "time_4_weighted"] = float(np.mean(t4w_all)) if t4w_all else None
                df_q.at[ridx, "time_4_unweighted"] = float(np.mean(t4u_all)) if t4u_all else None

            # Save per-row results parquet (same path pattern as your original)
            path_save_1 = OUT_DIR / filename.replace(
                ".xlsx", f"_results_model{NR_TPCS}tpc_thr_{thr_tag}.parquet"
            )
            df_q.to_parquet(path_save_1)


            df_exp = df_q.copy()
            columns_to_collect = [
                "results_1", "results_2",
                "results_3_weighted", "results_3_unweighted",
                "results_4_weighted", "results_4_unweighted",
            ]

            def parse_if_str(x):
                if isinstance(x, str):
                    try:
                        return ast.literal_eval(x)
                    except Exception:
                        return x
                return x

            def extract_doc_ids(obj):

                ids = set()
                obj = parse_if_str(obj)

                if obj is None or (isinstance(obj, float) and pd.isna(obj)):
                    return ids

                if isinstance(obj, dict):
                    if "doc_id" in obj and obj["doc_id"] is not None:
                        ids.add(obj["doc_id"])
                    if "docid" in obj and obj["docid"] is not None:
                        ids.add(obj["docid"])
                    return ids

                if isinstance(obj, (list, tuple, set, np.ndarray, pd.Series)):
                    for item in obj:
                        ids |= extract_doc_ids(item)
                    return ids

                return ids  # ignore other types

            def combine_results(row):
                ids = set()
                for col in columns_to_collect:
                    ids |= extract_doc_ids(row.get(col))
                return sorted(ids)

            df_exp["all_results"] = df_exp.apply(combine_results, axis=1)
            df_exp = df_exp[df_exp["all_results"].astype(bool)].copy()

            # Remove unused columns
            base_keep = ['pass_id', 'doc_id', 'passage', 'top_k', 'question', 'queries', 'all_results']
            keep_cols = [c for c in base_keep if c in df_exp.columns]
            df_eval = df_exp[keep_cols].copy()
            df_eval = df_eval.loc[:, ~df_eval.columns.duplicated()]

            # Union lists of duplicate logical rows
            group_keys = [c for c in ['pass_id', 'doc_id', 'passage', 'top_k', 'question', 'queries'] if c in df_eval.columns]
            if group_keys:
                from itertools import chain
                df_eval = (
                    df_eval
                    .groupby(group_keys, dropna=False, as_index=False)
                    .agg(all_results=('all_results', lambda lists: sorted(set(chain.from_iterable(lists)))))
                )

            # Explode all_results
            df_eval = df_eval.explode("all_results", ignore_index=True)
            df_eval = df_eval.dropna(subset=["all_results"])

            # Drop duplicate rows if any (same passage retrieved multiple times for same question)
            dedup_keys = [c for c in ['pass_id', 'doc_id', 'question', 'all_results'] if c in df_eval.columns]
            if dedup_keys:
                df_eval = df_eval.drop_duplicates(subset=dedup_keys, keep='first')

            # Map content
            df_eval["all_results_content"] = df_eval["all_results"].map(doc_map)

            path_save_2 = OUT_DIR / filename.replace(
                ".xlsx", f"_results_model{NR_TPCS}tpc_thr_{thr_tag}_combined_to_retrieve_relevant.parquet"
            )
            df_eval.to_parquet(path_save_2)
    print("Done.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Retrieve relevant passages using various methods.")
    parser.add_argument("--model_name", type=str, default="BAAI/bge-m3",
                        help="Name of the SentenceTransformer model.")
    parser.add_argument("--lang", type=str, default="ES",
                        help="Language of the corpus.")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of top results to retrieve.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for retrieval.")
    parser.add_argument("--nprobe", type=int, default=1,
                        help="Number of probes for ANN.")
    parser.add_argument("--min_clusters", type=int, default=8,
                        help="Minimum number of clusters.")
    parser.add_argument("--nr_tpcs", type=int, default=30,
                        help="Number of topics.")
    parser.add_argument("--text_col", type=str, default="text",
                        help="Column name for text in the corpus.")
    parser.add_argument("--id_col", type=str, default="doc_id",
                        help="Column name for document IDs in the corpus.")
    parser.add_argument("--epsilon_index", type=float,
                        default=0.0, help="Threshold for index assignment.")
    parser.add_argument("--path_source", type=str, required=True,
                        help="Path to the source corpus file.")
    parser.add_argument("--path_model_dir", type=str,
                        required=True, help="Path to the model directory.")
    parser.add_argument("--path_save_indices", type=str,
                        required=True, help="Root path for saving indices.")
    parser.add_argument("--path_queries_dir", type=str, required=True,
                        help="Path to the directory containing query files.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for results.")

    args = parser.parse_args()

    MODEL_NAME = args.model_name
    LANG = args.lang
    TOP_K = args.top_k
    BATCH_SIZE = args.batch_size
    NPROBE = args.nprobe
    MIN_CLUSTERS = args.min_clusters
    NR_TPCS = args.nr_tpcs
    TEXT_COL = args.text_col
    ID_COL = args.id_col
    EPSILON_INDEX = args.epsilon_index

    PATH_SOURCE = Path(args.path_source)
    PATH_MODEL_DIR = Path(args.path_model_dir)
    PATH_THETAS_LANG = PATH_MODEL_DIR / "mallet_output" / f"thetas_{LANG}.npz"
    PATH_THETAS_ANCHOR = PATH_MODEL_DIR / "mallet_output" / f"thetas_EN.npz"

    INDICES_ROOT = Path(args.path_save_indices)
    PATH_QUERIES_DIR = Path(args.path_queries_dir)

    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    main()
