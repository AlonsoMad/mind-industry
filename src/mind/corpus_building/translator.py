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

class Translator:
    def __init__(
        self,
        config_path: Path = Path("config/config.yaml"),
        logger=None
    ):
        self._logger = logger if logger else init_logger(config_path, __name__)
        self.models = {}
        self.tokenizers = {}
        self.supported = {
            ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
            ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
            ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
            ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
            ("en", "it"): "Helsinki-NLP/opus-mt-en-it",
            ("it", "en"): "Helsinki-NLP/opus-mt-it-en",
        }
        self.translated_df = None

    def _add_pair(self, src, tgt, repo):
        self.models[(src, tgt)] = pipeline("translation", model=repo)
        self.tokenizers[(src, tgt)] = AutoTokenizer.from_pretrained(repo)

    def _split(
        self,
        df: pd.DataFrame,
        src_lang: str,
        tgt_lang: str,
        text_col: str = "text",
        lang_col: str = "lang"
    ) -> pd.DataFrame:
        """
        Split each paragraph into sentences, filtering out too-long chunks
        based on the translation tokenizer for (src->tgt). Keeps all metadata columns.
        """
        tok = self.tokenizers[(src_lang, tgt_lang)]
        model_max = getattr(tok, "model_max_length", 512)
        max_tokens = int(model_max * 0.9)

        def token_len(text: str) -> int:
            return len(tok.encode(text, truncation=False))

        orig_cols = list(df.columns)
        rows = []
        dropped_ids = set()

        for _, row in df.iterrows():
            sentences = [s for s in str(row[text_col]).split(". ") if s]
            any_kept = False
            for j, s in enumerate(sentences):
                if token_len(s) < max_tokens:
                    entry = {col: row.get(col, None) for col in orig_cols}
                    entry[text_col] = s
                    entry[lang_col] = row[lang_col]
                    entry['index'] = None  # will set below
                    entry['id_preproc'] = f"{row.get('id_preproc', '')}_{j}"
                    rows.append(entry)
                    any_kept = True
            if not any_kept:
                dropped_ids.add(row.get("id_preproc", None))

        if dropped_ids:
            df = df[~df["id_preproc"].isin(dropped_ids)].reset_index(drop=True)

        out = pd.DataFrame(rows)
        out["index"] = range(len(out))
        return out

    def _translate_split(
        self,
        split_df: pd.DataFrame,
        src_lang: str,
        tgt_lang: str,
        text_col: str = "text"
    ) -> pd.Series:
        """
        Splits the DataFrame into smaller chunks for translation. The pandas Df is converted into a Huggingface Dataset for batch processing.

        Parameters
        ----------
        split_df: pd.DataFrame
            DataFrame containing the sentences to translate.
        src_lang: str
            Source language code.
        tgt_lang: str
            Target language code.
        text_col: str
            Name of the text column to translate.

        Returns
        -------
        pd.Series
            Series containing the translated text.
        """
        ds = Dataset.from_pandas(split_df)
        model = self.models[(src_lang, tgt_lang)]

        def translate_batch(batch):
            out = model(batch[text_col])
            batch["translated_text"] = [o["translation_text"] for o in out]
            return batch

        ds = ds.map(translate_batch, batched=True)
        return ds.to_pandas()["translated_text"]

    def _assemble(
        self,
        split_df: pd.DataFrame,
        translated_text: pd.Series,
        tgt_lang: str,
        text_col: str = "text",
        lang_col: str = "lang"
    ) -> pd.DataFrame:
        """
        Given a DataFrame of split sentences and their translations, reconstructs full translated paragraphs and preserves all original metadata.

        Steps:
        1. Replaces the sentence text in the DataFrame with the translated sentences.
        2. Groups sentences by their original paragraph (using 'id_preproc'), and joins them to form full translated paragraphs.
        3. For each paragraph, restores all metadata columns from the original data (except for the text and id_preproc, which are updated).
        4. Sets the language column to the target language and updates the id to indicate translation.

        Returns a DataFrame where each row is a translated paragraph, with all original metadata preserved and updated for the new language.
        """
        tmp = split_df.copy()
        tmp[text_col] = translated_text.values

        tmp["aux_id"] = tmp["id_preproc"].str.rsplit("_", n=1).str[0]
        grouped = (
            tmp.groupby("aux_id")[text_col]
            .agg(lambda x: ' '.join(x.astype(str).str.strip()))
            .reset_index()
            .rename(columns={text_col: "assembled_text"})
        )

        # Merge back all metadata columns except text_col and id_preproc
        meta_cols = [col for col in tmp.columns if col not in [
            text_col, "id_preproc", "aux_id"]]
        meta_df = tmp.drop_duplicates(subset=["aux_id"])[
            ["aux_id"] + meta_cols]
        merged = (
            meta_df.merge(grouped, on="aux_id", how="outer")
            .rename(columns={"aux_id": "id_preproc", "assembled_text": text_col})
            .assign(id_preproc=lambda x: "T_" + x["id_preproc"])
            .assign(**{lang_col: tgt_lang})
            .reset_index(drop=True)
        )
        return merged

    def translate(
        self,
        path_df: Path,
        src_lang: str,
        tgt_lang: str,
        text_col: str = "text",
        lang_col: str = "lang",
        save_path: str = None
    ):
        """
        Translates the given DataFrame from src_lang to tgt_lang and appends the translated documents. Saves the result to disk if save_path is provided.

        Parameters
        ----------
        path_df: Path
            Path to the input DataFrame (parquet format).
        text_col: str
            name of the text column to translate
        lang_col: str
            name of the language column
        save_path: str
            directory to save the translated DataFrame (as parquet)
        """
        if src_lang == tgt_lang:
            raise ValueError(
                f"Source and target languages must differ. Got: {src_lang}")
        if (src_lang, tgt_lang) not in self.supported:
            raise Exception(
                f"Unsupported language pair: {(src_lang, tgt_lang)}")

        self._logger.info(f"Loading dataframe from {path_df}")
        df = pd.read_parquet(path_df)
        self._logger.info(f"Loaded dataframe with {len(df)} rows.")
        if src_lang not in df[lang_col].unique():
            raise ValueError(
                f"Source language '{src_lang}' not found in column '{lang_col}'")

        self._logger.info(
            f"Preparing translation pipeline for {src_lang} → {tgt_lang}")
        if (src_lang, tgt_lang) not in self.models:
            repo = self.supported[(src_lang, tgt_lang)]
            self._add_pair(src_lang, tgt_lang, repo)

        self._logger.info(
            f"Splitting paragraphs into sentences for translation...")
        start_time = time.time()
        split_df = self._split(df, src_lang, tgt_lang,
                               text_col=text_col, lang_col=lang_col)
        self._logger.info(
            f"Split into {len(split_df)} sentences. Translating...")
        trans_text = self._translate_split(
            split_df, src_lang, tgt_lang, text_col=text_col)
        self._logger.info(f"Translation complete. Reassembling paragraphs...")
        merged = self._assemble(
            split_df, trans_text, tgt_lang, text_col=text_col, lang_col=lang_col)
        elapsed = time.time() - start_time
        self._logger.info(
            f"Translation and assembly took {elapsed:.2f} seconds.")

        # Append translated docs to original
        self.translated_df = pd.concat([df, merged], ignore_index=True)

        if save_path is not None:
            self.translated_df.to_parquet(save_path, compression="gzip")
            self._logger.info(f"Saved translated DataFrame to {save_path}")

        return self.translated_df
