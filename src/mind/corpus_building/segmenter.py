from typing import Dict, Optional, Tuple, List
import subprocess
import re
from pathlib import Path
import pandas as pd

import time
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer
from datasets import Dataset

from mind.utils.utils import init_logger

class Segmenter():
    def __init__(
        self,
        config_path: Path = Path("config/config.yaml"),
        logger=None
    ):
        self._logger = logger if logger else init_logger(config_path, __name__)

    def segment(
        self,
        path_df: Path,
        path_save: Path,
        text_col: str = "text",
        min_length: int = 100,
        sep: str = "\n"
    ):
        """
        Segments each entry in the specified text column into paragraphs, filters out short/empty ones, and saves the resulting dataframe to the specified path.

        Parameters:
        -----------
        text_col: str 
            Name of the column to segment.
        min_length: int
            Minimum length for a paragraph to be kept.
        sep: str
            Separator for splitting paragraphs (default: newline).
        """

        self._logger.info(f"Loading dataframe from {path_df}")
        df = pd.read_parquet(path_df)
        self._logger.info(
            f"Loaded {len(df)} rows. Starting segmentation on column '{text_col}'...")

        # we preserve the original document metadata columns for each new paragraph
        orig_cols = list(df.columns)
        new_rows = []

        import time
        self._logger.info(
            f"Segmenting paragraphs using separator '{sep}' and minimum length {min_length}...")
        start_time = time.time()
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Segmenting paragraphs"):
            full_doc_text = str(row[text_col])
            paragraphs = [p for p in full_doc_text.split(
                sep) if p and len(p) > min_length]
            for idx, p in enumerate(paragraphs):
                entry = {col: row.get(col, None) for col in orig_cols}
                entry[text_col] = p  # replace with paragraph
                entry['full_doc'] = full_doc_text  # add full document text
                entry['id'] = None  # will set below
                entry['id_preproc'] = f"{row.get('id_preproc', '')}_{idx}"
                new_rows.append(entry)
        elapsed = time.time() - start_time
        self._logger.info(f"Segmentation took {elapsed:.2f} seconds.")

        seg_df = pd.DataFrame(new_rows)
        seg_df['id'] = range(len(seg_df))
        self._logger.info(
            f"Segmented into {len(seg_df)} paragraphs. Saving to {path_save}")
        seg_df.to_parquet(path_save, compression="gzip")
        self._logger.info(f"Saved segmented dataframe to {path_save}")
        return path_save
