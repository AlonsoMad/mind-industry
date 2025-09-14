from __future__ import annotations

import argparse
import ast
import math
from os import listdir
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

try:
    from statsmodels.stats.multitest import multipletests  # type: ignore
    has_statsmodels = True
except Exception:
    has_statsmodels = False

METHOD_MAPPING: Dict[str, str] = {
    "1": "ENN",
    "2": "ANN",
    "3_weighted": "TB-ENN-W",
    "3_weighted_dynamic": "TB-ENN-W-D",
    "3_unweighted": "TB-ENN",
    "3_unweighted_dynamic": "TB-ENN-D",
    "4_weighted": "TB-ANN-W",
    "4_weighted_dynamic": "TB-ANN-W-D",
    "4_unweighted": "TB-ANN",
    "4_unweighted_dynamic": "TB-ANN-D",
    "time_1": "ENN",
    "time_2": "ANN",
    "time_3_weighted": "TB-ENN-W",
    "time_3_weighted_dynamic": "TB-ENN-W-D",
    "time_3_unweighted": "TB-ENN",
    "time_3_unweighted_dynamic": "TB-ENN-D",
    "time_4_weighted": "TB-ANN-W",
    "time_4_weighted_dynamic": "TB-ANN-W-D",
    "time_4_unweighted": "TB-ANN",
    "time_4_unweighted_dynamic": "TB-ANN-D",
}

def _safe_eval_listlike(x) -> List:
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, str):
        try:
            x = ast.literal_eval(x)
        except Exception:
            return []
    return x if isinstance(x, list) else []


def extract_doc_ids(cell) -> List[str]:
    data = _safe_eval_listlike(cell)
    if not data:
        return []
    try:
        return [el["doc_id"] for el in data[0]]
    except Exception:
        try:
            flat = [
                {"doc_id": entry["doc_id"], "score": entry.get("score")}
                for sub in data
                for entry in (sub if isinstance(sub, list) else [])
                if isinstance(entry, dict) and "doc_id" in entry
            ]
            return [el["doc_id"] for el in flat]
        except Exception:
            return []

def precision_at_k(doc_ids: List[str], relevant_docs: Iterable[str], k: int) -> float:
    """
    Precision@k: proportion of top-k retrieved that are relevant.
    """
    if k <= 0:
        return 0.0
    rel = set(relevant_docs)
    retrieved = doc_ids[:k]
    tp = sum(1 for d in retrieved if d in rel)
    return tp / k                                

def recall_at_k(doc_ids: List[str], relevant_docs: Iterable[str], k: int) -> float:
    """
    Recall@k: proportion of relevant documents that are in the top-k retrieved.
    """
    rel = set(relevant_docs)
    if not rel:
        return 0.0
    return len(set(doc_ids[:k]) & rel) / len(rel)

def mrr_multi_at_k(doc_ids: List[str], relevant_docs: Iterable[str], k: int) -> float:
    """
    Multiple Mean Reciprocal Rank at k (MRR@k): how close the average rank of all relevant items is to the best possible under top-k.
    It penalizes if any relevant item is missing/low in the top-k.
    """
    rel = set(relevant_docs)
    n = len(rel)
    if n == 0 or k <= 0:
        return 0.0

    retrieved = doc_ids[:k]
    first_pos = {}
    for i, d in enumerate(retrieved):
        if d not in first_pos:
            first_pos[d] = i + 1  # 1-based rank

    ranks = [first_pos[d] for d in rel if d in first_pos]
    R = len(ranks)

    observed_avg = (sum(ranks) + (k + 1) * (n - R)) / n

    R_star = min(n, k)  # best possible under top-k
    best_possible_avg = (R_star * (R_star + 1) / 2 + (k + 1) * (n - R_star)) / n

    return best_possible_avg / observed_avg

def mrr_at_k(doc_ids, relevant_docs, k):
    """
    Mean Reciprocal Rank at k (MRR@k): how close the rank of the first relevant item is to the best possible under top-k.
    """
    if k <= 0: return 0.0
    rel = set(relevant_docs)
    if not rel: return 0.0
    for i, d in enumerate(doc_ids[:k], start=1):  # 1-based ranks
        if d in rel:
            return 1.0 / i
    return 0.0

def ndcg_at_k(doc_ids: List[str], relevant_docs: Iterable[str], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at k (NDCG@k) with binary relevance: how well the method puts relevant items near the very top.
    
    It rewards if some relevant items are at the very top, even others are missing.
    """
    if k <= 0:
        return 0.0

    rel = set(relevant_docs)
    retrieved = doc_ids[:k]

    # DCG with binary relevance: we only sum the one retrieved that are relevant, and in the denominator we place (i+1)+1 to account for the enumerate starting at 0
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(retrieved) if d in rel)

    # IDCG: place min(#relevant, k) ones at the top
    R_star = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(R_star))

    return (dcg / idcg) if idcg > 0 else 0.0

def is_higher_better(metric_name: str) -> bool:
    return "Time (s)" not in metric_name

def metric_pretty_name(raw_metric_key: str) -> str:
    if raw_metric_key.startswith("mrr_"):
        return f"MRR@{raw_metric_key.split('_', 1)[1]}"
    if raw_metric_key.startswith("precision_"):
        return f"Precision@{raw_metric_key.split('_', 1)[1]}"
    if raw_metric_key.startswith("recall_"):
        return f"Recall@{raw_metric_key.split('_', 1)[1]}"
    if raw_metric_key.startswith("ndcg_"):
        return f"NDCG@{raw_metric_key.split('_', 1)[1]}"
    if raw_metric_key.startswith("time"):
        return "Time (s)"
    return raw_metric_key


def extract_metric_and_method(col: str) -> Tuple[str, str]:
    if col.startswith("time_"):
        method_key = col
        method = METHOD_MAPPING.get(method_key, method_key)
        return "Time (s)", method
    if "_results_" in col:
        metric_key, method_key = col.split("_results_", 1)
        return metric_pretty_name(metric_key), METHOD_MAPPING.get(method_key, method_key)
    return metric_pretty_name(col), col


def prepare_annotations(relevant_check_path: str, prepare=True) -> pd.DataFrame:
    if prepare:
        ann_path = relevant_check_path.replace(".parquet", "_all_added.parquet")
    else:
        ann_path = relevant_check_path
    df = pd.read_parquet(ann_path)
    keep = (
        (df["relevance_llama3.3:70b"] == 1)
        & (df["relevance_qwen:32b"] == 1)
       #  & (df["relevance_llama3.3:70b-instruct-q5_K_M"] == 1)
       & (df["relevance_llama3:70b-instruct"] == 1)
        & (df["relevance_llama3.1:8b-instruct-q8_0"] == 1)
    )
    df = df[keep].copy()
    grouped = (
        df.groupby("question")["all_results"].apply(list).reset_index().rename(columns={"all_results": "relevant_docs"})
    )
    grouped["num_relevant_docs"] = grouped["relevant_docs"].apply(len)
    return grouped


def compute_per_row_metrics(df: pd.DataFrame, key: str, ks: Iterable[int]) -> pd.DataFrame:
    doc_col = f"doc_ids_{key}"
    df[doc_col] = df[key].apply(extract_doc_ids)
    for k in ks:
        df[f"mrr_{k}_{key}"] = df.apply(lambda r: mrr_multi_at_k(r[doc_col], r["relevant_docs"], k), axis=1)
        df[f"precision_{k}_{key}"] = df.apply(lambda r: precision_at_k(r[doc_col], r["relevant_docs"], k), axis=1)
        df[f"recall_{k}_{key}"] = df.apply(lambda r: recall_at_k(r[doc_col], r["relevant_docs"], k), axis=1)
        df[f"ndcg_{k}_{key}"] = df.apply(lambda r: ndcg_at_k(r[doc_col], r["relevant_docs"], k), axis=1)
    return df

def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, ci: float = 0.95, random_state=None):
    """
    Compute the bootstrap confidence interval spread for the mean of a given dataset.

    Parameters:
        values (np.ndarray): Array of numerical values for which the confidence interval is calculated.
        n_boot (int, optional): Number of bootstrap samples to generate. Default is 1000.
        ci (float, optional): Confidence level for the interval, expressed as a proportion (e.g., 0.95 for 95%). Default is 0.95.
        random_state (int or np.random.Generator, optional): Seed or random number generator for reproducibility. Default is None.

    Returns:
        float: Half the width of the confidence interval (spread) around the mean. Returns 0.0 if the input array has fewer than 2 elements.
    """
    rng = np.random.default_rng(random_state)
    vals = np.asarray(values, dtype=float)
    n = vals.size
    if n < 2:
        return 0.0
    idx = rng.integers(0, n, size=(n_boot, n))  # resample query indices
    means = vals[idx].mean(axis=1)
    alpha = (1 - ci) / 2.0
    lower, upper = np.percentile(means, [100*alpha, 100*(1-alpha)])
    return float((upper - lower) / 2.0)

def summarize(df: pd.DataFrame, metric_cols: List[str]) -> List[Tuple[str, int, str, str, float, float]]:
    """
    Group, FileNum, Method, Metric, Mean, CI (in the current version the file could actually be removed)
    """
    results = []
    group = str(df["Group"].iat[0]) if "Group" in df.columns else "ALL"
    file_num = int(df["FileNum"].iat[0]) if "FileNum" in df.columns else 1

    for col in metric_cols:
        metric_name, method_display = extract_metric_and_method(col)
        if col not in df.columns:
            continue
        values = df[col].dropna()
        if values.empty:
            continue

        vals = values.values.astype(float)
        mean_val = float(vals.mean())
        spread = bootstrap_ci(vals, n_boot=1000, ci=0.95, random_state=1234)

        results.append((group, file_num, method_display, metric_name, mean_val, spread))
    return results

def compute_significance_rm(
    df_results: pd.DataFrame,
    retrieval_metrics: List[str],
) -> Dict[str, Dict]:
    """
    Significance tests:
      1) Omnibus Friedman across methods on per-query scores.
      2) One-sided Wilcoxon signed-rank, Holm-adjusted within families of metrics.

      Hypotheses (A > B):
        - TB-ANN   > ANN
        - TB-ANN-W > ANN
        - TB-ANN-W-D > ANN
        - TB-ENN   > ENN
        - TB-ENN-W > ENN
        - TB-ENN-W-D > ENN
        - TB-ANN-W > TB-ANN
        - TB-ANN-W-D > TB-ANN-D
        - TB-ENN-W > TB-ENN
        - TB-ENN-W-D > TB-ENN-D
    """
    # collect columns
    keep_cols = [c for c in df_results.columns if any(m in c for m in retrieval_metrics + ["time_"])]
    metric_cols = [c for c in keep_cols if ("results_" in c or c.startswith("time_"))]

    df_long = df_results.melt(
        id_vars=["question", "num_relevant_docs"],
        value_vars=metric_cols,
        var_name="Metric",
        value_name="Value",
    ).dropna(subset=["Value"])

    def _to_method(col: str) -> str:
        if col.startswith("time_"):
            return METHOD_MAPPING.get(col, col)
        if "_results_" in col:
            return METHOD_MAPPING.get(col.split("_results_", 1)[-1], col)
        return col

    df_long["Method"] = df_long["Metric"].apply(_to_method)
    df_long["MetricName"] = df_long["Metric"].apply(lambda x: extract_metric_and_method(x)[0])

    friedman_p: Dict[str, float] = {}
    directional_by_metric: Dict[str, Dict[Tuple[str, str], Dict[str, float]]] = {}

    # matched rank-biserial effect size
    def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
        d = a - b
        nz = d != 0
        if nz.sum() == 0:
            return 0.0
        ranks = stats.rankdata(np.abs(d[nz]))
        sgn = np.sign(d[nz])
        return float((sgn * ranks).sum() / (len(ranks) * (len(ranks) + 1) / 2.0))

    for m in sorted(df_long["MetricName"].unique()):
        sub = df_long[df_long["MetricName"] == m].copy()
        mat = sub.pivot_table(index="question", columns="Method", values="Value", aggfunc="mean")

        # 1) Friedman omnibus
        try:
            mat_full = mat.dropna(how="any")
            if mat_full.shape[0] >= 2 and mat_full.shape[1] >= 2:
                _, p = stats.friedmanchisquare(*[mat_full.iloc[:, j].values for j in range(mat_full.shape[1])])
                friedman_p[m] = float(p)
        except Exception as e:
            print(f"Exception in Friedman for {m}: {e}")
            continue

        # 2) Wilcoxon
        directional_pairs = [
            ("TB-ANN",   "ANN"),
            ("TB-ANN-W", "ANN"),
            ("TB-ANN-W-D", "ANN"),
            ("TB-ENN",   "ENN"),
            ("TB-ENN-W", "ENN"),
            ("TB-ENN-W-D", "ENN"),
            ("TB-ANN-W", "TB-ANN"),
            ("TB-ENN-W", "TB-ENN"),
            ("TB-ANN-W-D", "TB-ANN-D"),
            ("TB-ENN-W-D", "TB-ENN-D"),
        ]
        higher_better = (m != "Time (s)")
        alt = "greater" if higher_better else "less"

        dir_ps: List[float] = []
        dir_names: List[Tuple[str, str]] = []
        dir_es: List[float] = []
        dir_n: List[int] = []

        print("DEBUG Wilcoxon columns for", m, ":", list(mat.columns))
        
        for a, b in directional_pairs:
            if a in mat.columns and b in mat.columns:
                pair = mat[[a, b]].dropna(how="any")
                if len(pair) >= 2:
                    try:
                        w = stats.wilcoxon(
                            pair[a].values, pair[b].values,
                            zero_method="pratt",
                            alternative=alt,
                            mode="approx"
                        )
                        dir_ps.append(float(w.pvalue))
                        dir_names.append((a, b))
                        dir_es.append(rank_biserial(pair[a].values, pair[b].values))
                        dir_n.append(len(pair[a].values))
                    except Exception:
                        pass

        families = {
            "ANN_dynamic":   [("TB-ANN",   "ANN"), ("TB-ANN-W",   "ANN"),("TB-ANN-D", "ANN"), ("TB-ANN-W-D", "ANN")],
            "ENN_dynamic":   [("TB-ENN",   "ENN"), ("TB-ENN-W",   "ENN"),("TB-ENN-D", "ENN"), ("TB-ENN-W-D", "ENN")],
            "TB_weight_dynamic": [("TB-ANN-W-D", "TB-ANN-D"), ("TB-ENN-W-D", "TB-ENN-D"), ("TB-ANN-W", "TB-ANN"), ("TB-ENN-W", "TB-ENN")],
        }

        dir_dict = {}
        for fam_name, fam_pairs in families.items():
            print("DEBUG family", fam_name, "pairs", fam_pairs)
            idxs = [i for i,(a,b) in enumerate(dir_names) if (a,b) in fam_pairs]
            if not idxs:
                continue
            fam_ps  = [dir_ps[i] for i in idxs]
            fam_es  = [dir_es[i] for i in idxs]
            fam_ns  = [dir_n[i] for i in idxs]
            fam_n   = len(fam_ps)

            if has_statsmodels:
                _, p_adj_fam, _, _ = multipletests(fam_ps, method="holm")
            else:
                order = np.argsort(fam_ps)
                p_adj_fam = [1.0]*fam_n
                running_max = 0.0
                for rank, j in enumerate(order):
                    adj = min(1.0, (fam_n - rank) * fam_ps[j])
                    running_max = max(running_max, adj)
                    p_adj_fam[j] = running_max

            for local_idx, i in enumerate(idxs):
                a, b = dir_names[i]
                dir_dict[(a, b)] = {"p_adj": float(p_adj_fam[local_idx]),
                                    "r_rb": float(fam_es[local_idx]),
                                    "n":    int(fam_ns[local_idx])}

        directional_by_metric[m] = dir_dict

    return {
        "Friedman p-value": friedman_p,
        "Directional Wilcoxon-Holm": directional_by_metric,
    }


def format_value_with_sig(mean: float,
                          spread: float,
                          metric: str,
                          method: str,
                          best: bool,
                          sig_block: Dict[str, Dict]) -> str:
    """
    Put daggers based on one-sided Wilcoxon-Holm adjusted p-values.

    Markers:
      † : TB-* > baseline (ANN or ENN) when p_adj < 0.05
      ‡ : weighted TB > unweighted TB (TB-*-W > TB-*) when p_adj < 0.05

    No markers for Time (s).
    """
    val = f"{mean:.3f} \\pm {spread:.3f}"
    marks: List[str] = []

    if metric != "Time (s)":
        dir_pack = (sig_block or {}).get("Directional Wilcoxon-Holm", {})
        dir_for_metric = dir_pack.get(metric, {}) if isinstance(dir_pack, dict) else {}

        def dir_p(a: str, b: str) -> Optional[float]:
            entry = dir_for_metric.get((a, b))
            if entry is None:
                return None
            if isinstance(entry, dict):
                return entry.get("p_adj")
            return float(entry)

        # daggers: TB vs baseline
        if method.startswith("TB-ANN"):
            p = dir_p(method, "ANN")
            if p is not None and p < 0.05:
                marks.append("\\dagger")
        if method.startswith("TB-ENN"):
            p = dir_p(method, "ENN")
            if p is not None and p < 0.05:
                marks.append("\\dagger")

        # double daggers: weighted vs unweighted TB
        if method.startswith("TB-") and "-W" in method:
            base = method.replace("-W", "")
            p = dir_p(method, base)
            if p is not None and p < 0.05:
                marks.append("\\ddagger")

    sup = "".join(marks)
    formatted = rf"{val}^{{{sup}}}" if sup else val
    return rf"$\boldsymbol{{{formatted}}}$" if best else rf"${formatted}$"


def to_latex_table(
    df_summary: pd.DataFrame,
    significance_by_gf: Dict[Tuple[str, int], Dict[str, Dict]]
) -> str:
    dfw = df_summary.copy()

    if "FileNum" not in dfw.columns:
        dfw["FileNum"] = 1
    if "Group" not in dfw.columns:
        dfw["Group"] = "ALL"

    # Mark best per (Group, Metric)
    best_mask = np.zeros(len(dfw), dtype=bool)
    for (_, metric), grp_df in dfw.groupby(["Group", "Metric"], dropna=False):
        target = grp_df["Mean"].max() if is_higher_better(metric) else grp_df["Mean"].min()
        best_mask[grp_df.index] = grp_df["Mean"].eq(target).values
    dfw["IsBest"] = best_mask

    # Add significance markers
    def _fmt_row(r):
        sig_block = significance_by_gf.get((r["Group"], int(r["FileNum"])), {}) if isinstance(significance_by_gf, dict) else {}
        return format_value_with_sig(
            float(r["Mean"]), float(r["CI"]), str(r["Metric"]), str(r["Method"]), bool(r["IsBest"]), sig_block
        )

    dfw["Mean ± CI"] = dfw.apply(_fmt_row, axis=1)

    # @TODO: remove multi-file
    multi_file = dfw.groupby("Group")["FileNum"].nunique().max() > 1

    # Pivot for latex
    if multi_file:
        pivot = (
            dfw.pivot(index=["Group", "FileNum", "Method"], columns="Metric", values="Mean ± CI")
               .sort_index()
        )
    else:
        pivot = (
            dfw.pivot(index=["Group", "Method"], columns="Metric", values="Mean ± CI")
               .sort_index()
        )

    metric_cols = list(pivot.columns)
    header_metrics = " & ".join([f"\\textbf{{{c}}}" for c in metric_cols])

    if multi_file:
        col_spec = f"c c {'c'*len(metric_cols)}"
        header_line = f"\\textbf{{File}} & \\textbf{{Method}} & {header_metrics} \\\\"
        leading_cols = 2
    else:
        col_spec = f"c {'c'*len(metric_cols)}"
        header_line = f"\\textbf{{Method}} & {header_metrics} \\\\"
        leading_cols = 1

    latex = (
        "\\begin{table*}[h]\n\\centering\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\arrayrulecolor{black}\n\\toprule\n"
        f"{header_line}\n"
        "\\midrule\n"
    )

    total_cols = len(metric_cols) + leading_cols
    current_group = None

    # Iterate by group
    for group, group_df in pivot.groupby(level=0, sort=False):
        # group header
        if current_group is not None:
            latex += "\\arrayrulecolor{black}\\specialrule{0.5pt}{0pt}{0pt}\\arrayrulecolor{black}\n"
        latex += f"\\multicolumn{{{total_cols}}}{{l}}{{\\textbf{{{group}}}}} \\\\\n"
        latex += "\\arrayrulecolor{black}\\specialrule{0.5pt}{0pt}{0pt}\\arrayrulecolor{black}\n"
        current_group = group

        if multi_file:
            # Files inside this group (level 1 = FileNum)
            files_in_group = list({idx[1] for idx in group_df.index})
            files_in_group = sorted(files_in_group)
            for i_file, fnum in enumerate(files_in_group):
                file_df = group_df.xs((group, fnum), level=(0, 1), drop_level=False)
                methods = list(file_df.index.get_level_values(2))
                n_methods = len(methods)
                for j, method in enumerate(methods):
                    row = file_df.xs((group, fnum, method)).reindex(metric_cols).fillna("").astype(str)
                    metrics_line = " & ".join(row.tolist())
                    if j == 0:
                        latex += f"\\multirow{{{n_methods}}}{{*}}{{{int(fnum)}}} & \\textbf{{{method}}} & {metrics_line} \\\\\n"
                    else:
                        latex += f" & \\textbf{{{method}}} & {metrics_line} \\\\\n"
                if i_file < len(files_in_group) - 1:
                    latex += "\\midrule\n"
        else:
            # Single-file case: rows are just methods
            group_only = group_df.xs(group, level=0, drop_level=False)
            methods = list(group_only.index.get_level_values(1))
            for method in methods:
                row = group_only.xs((group, method)).reindex(metric_cols).fillna("").astype(str)
                metrics_line = " & ".join(row.tolist())
                latex += f"\\textbf{{{method}}} & {metrics_line} \\\\\n"

    latex += "\\bottomrule\n\\end{tabular}}\n"
    latex += "\\caption{Performance metrics per method. "
    latex += "Values are means over queries with 95\\% bootstrap confidence intervals for retrieval metrics at $L \\in \\{3,5\\}$. "
    latex += "Results are based on relevant $c_p^{(c)}$ passages for questions from 100 $t_{16}$ $c_p^{(a)}$ passages. "
    latex += "Best values are bolded. Daggers $\\dagger$ indicate topic-based methods (static or dynamic) significantly outperform their baselines (ANN or ENN). Double daggers $\\ddagger$ indicate weighted topic-based methods (static or dynamic) significantly outperform their unweighted counterparts.} \n"
    latex += "\\label{tab:grouped_results}\n\\end{table*}\n"
    return latex


def main(group_label: str, experiment_paths, k_values=(3, 5)) -> None:
    print(f"\n***** {group_label} *****")
    print(f"[{group_label}] files used:")
    for p in files_found_relevant:
        print("  -", p)

    relevant_check_path = experiment_paths[0][0]
    result_paths = experiment_paths[0][1]
    annotations = prepare_annotations(relevant_check_path, prepare=True)
    print(
        f"Number of relevant passages for {group_label}: "
        f"{annotations['num_relevant_docs'].mean():.3f}±{annotations['num_relevant_docs'].std():.3f}"
    )

    # ---- file 1 (baselines + epsilon = 0 --> static)
    df1 = (
        pd.read_parquet(result_paths[0])
        .drop_duplicates(subset=["question"], keep="first")
        .rename(columns={"questions": "question"})
        .merge(annotations, on="question", how="inner")
    )
    df1["Group"] = group_label
    df1["FileNum"] = 1

    keys_f1 = [
        "results_1", "results_2",
        "results_3_weighted", "results_3_unweighted",  
        "results_4_weighted", "results_4_unweighted",
    ]
    for key in keys_f1:
        if key in df1.columns:
            df1 = compute_per_row_metrics(df1, key, k_values)

    metric_cols_f1 = []
    for key in keys_f1:
        if key in df1.columns:
            for k in k_values:
                metric_cols_f1 += [
                    f"mrr_{k}_{key}",
                    f"precision_{k}_{key}",
                    f"recall_{k}_{key}",
                    f"ndcg_{k}_{key}",
                ]
                
    time_cols_f1 = [c for c in df1.columns if c.startswith("time_")]
    metric_cols_f1 += time_cols_f1
    keep_f1 = ["question", "num_relevant_docs"] + metric_cols_f1
    df1 = df1.loc[:, keep_f1]

    # ---- file 2 (var epsilon --> dynamic)
    df2 = (
        pd.read_parquet(result_paths[1])
        .drop_duplicates(subset=["question"], keep="first")
        .rename(columns={"questions": "question"})
        .merge(annotations, on="question", how="inner")
    )
    df2["Group"] = group_label
    df2["FileNum"] = 2

    # rename to avoid collisions
    df2 = df2.rename(columns={
        "results_3_weighted": "results_3_weighted_dynamic",
        "results_3_unweighted": "results_3_unweighted_dynamic",
        "results_4_weighted": "results_4_weighted_dynamic",
        "results_4_unweighted": "results_4_unweighted_dynamic",
        "time_3_weighted": "time_3_weighted_dynamic",
        "time_3_unweighted": "time_3_unweighted_dynamic",
        "time_4_weighted": "time_4_weighted_dynamic",
        "time_4_unweighted": "time_4_unweighted_dynamic",
    })

    keys_f2 = [
        "results_3_weighted_dynamic", "results_3_unweighted_dynamic",
        "results_4_weighted_dynamic", "results_4_unweighted_dynamic",
    ]
    for key in keys_f2:
        if key in df2.columns:
            df2 = compute_per_row_metrics(df2, key, k_values)

    metric_cols_f2 = []
    for key in keys_f2:
        if key in df2.columns:
            for k in k_values:
                metric_cols_f2 += [
                    f"mrr_{k}_{key}",
                    f"precision_{k}_{key}",
                    f"recall_{k}_{key}",
                    f"ndcg_{k}_{key}",
                ]
    time_cols_f2 = [c for c in df2.columns if c.startswith("time_")]
    metric_cols_f2 += time_cols_f2

    keep_f2 = ["question", "num_relevant_docs"] + metric_cols_f2
    df2 = df2.loc[:, keep_f2]

    # Merge files (fixed epsilon / variable epsilon)
    df2 = df2.drop(columns=["time_1", "time_2"], errors="ignore")
    df = pd.merge(df1, df2, on=["question", "num_relevant_docs"], how="inner")

    df["Group"] = group_label
    df["FileNum"] = 1  # single merged file for reporting

    # assert: baselines from merged should equal file-1 baselines
    def _mean(df_, col): return df_[col].mean()
    for k in k_values:
        for base_key in ["results_1", "results_2"]:
            mrr_col = f"mrr_{k}_{base_key}"
            if mrr_col in df1 and mrr_col in df:
                m1 = _mean(df1, mrr_col)
                m  = _mean(df,  mrr_col)
                assert np.isclose(m, m1, rtol=1e-12, atol=1e-12), f"Baseline mismatch for {mrr_col}: {m} vs {m1}"

    metric_cols = list(dict.fromkeys(metric_cols_f1 + metric_cols_f2))

    required_time_cols = ["time_1", "time_2"]
    for c in required_time_cols:
        if c in df.columns and c not in metric_cols:
            metric_cols.append(c)
    metric_cols = [c for c in metric_cols if c in df.columns]

    # optional debug
    print("Merged time cols:", [c for c in df.columns if c.startswith("time_")])
    print("Metrics passed to summarize:", metric_cols)

    # get significance
    summary_rows = summarize(df, metric_cols)
    sig = compute_significance_rm(df, [f"{p}_{k}" for p in ("mrr","precision","recall","ndcg") for k in k_values])

    df_summary = pd.DataFrame(summary_rows, columns=["Group", "FileNum", "Method", "Metric", "Mean", "CI"])
    latex = to_latex_table(df_summary, {(group_label, 1): sig})
    print(latex)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval runs with explicit paths (corrected).")
    parser.add_argument("--model_eval", type=str, default="gpt-4o-2024-08-06") #"llama3.3:70b", "qwen:32b"
    parser.add_argument("--path_gold_relevant", type=str, default="", help="Path to the relevant_check parquet file.")
    parser.add_argument("--paths_found_relevant", type=str, help="List of paths to found relevant results parquet files.")
    parser.add_argument("--tpc", type=int, default=11, help="Topic ID (used only for printing labels).")
    args = parser.parse_args()

    exp_paths = []
    files_found_relevant = listdir(args.paths_found_relevant)

    files_found_relevant = [f for f in files_found_relevant if
                            (args.model_eval in f
                                and (f.endswith("_thr_.parquet") or f.endswith("_thr__dynamic.parquet"))
                                )]
    files_found_relevant = [f"{args.paths_found_relevant}/{f}" for f in files_found_relevant]
    files_found_relevant = sorted(files_found_relevant)
    print(f"[{args.model_eval}] files used:", *files_found_relevant, sep="\n  - ")

    exp_paths.append(
        (
            args.path_gold_relevant,
            files_found_relevant
        )
    )

    main(group_label=args.model_eval, experiment_paths=exp_paths, k_values=(3,5))