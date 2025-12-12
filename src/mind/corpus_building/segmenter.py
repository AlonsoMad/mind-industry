import argparse
from pathlib import Path

import pandas as pd
from mind.utils.utils import init_logger
from tqdm import tqdm


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
        id_col: str = "id_preproc",
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
                entry['id_preproc'] = f"{row.get(id_col, '')}_{idx}"
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Segmenter to split documents into segments.")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input file (parquet or csv).")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save segmented output file.")
    parser.add_argument("--text_col", type=str, default="text",
                        help="Name of the text column to segment.")
    parser.add_argument("--min_length", type=int, default=100,
                        help="Minimum length for a paragraph to be kept.")
    parser.add_argument("--separator", type=str, default="\n",
                        help="Separator for splitting paragraphs.")
    args = parser.parse_args()

    segmenter = Segmenter()
    result_path = segmenter.segment(
        path_df=Path(args.input),
        path_save=Path(args.output),
        text_col=args.text_col,
        min_length=args.min_length,
        sep=args.separator
    )
    
    # Read the result to get row count
    result_df = pd.read_parquet(result_path)
    print(
        f"Segmentation complete. Saved to {args.output}. Rows: {len(result_df)}")
