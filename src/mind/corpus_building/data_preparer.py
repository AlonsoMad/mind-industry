"""
Builds a dataset in the format required by the PLTM wrapper, starting from:
    - One row per chunk/passage with original metadata
    - 'lemmas' = preprocessed original text (via NLPipe, once per language)
    - 'lemmas_tr' = lemmas of the cross-language translated counterpart

Assumptions:
    - Two input parquet files, one per language (anchor/target).
    - Chunk ids follow patterns like:
            Originals:     EN_<doc>_<chunk>   /  DE_<doc>_<chunk> / ES_...
            Translations: T_EN_<doc>_<chunk>  / T_DE_<doc>_<chunk> ...

Output columns include (plus any extra metadata preserved):
    - chunk_id   (from schema)
    - doc_id     (from schema or derived from chunk_id)
    - full_doc   (from summary-like field if available)
    - text       (original chunk text)
    - lang       (UPPER, e.g., EN/DE/ES)
    - lemmas     (from NLPipe run on this language)
    - lemmas_tr  (from the *other* language's NLPipe run over translations)
    - title, url, equivalence (if present)
"""

from typing import Dict, Optional, Tuple, List
import subprocess
import re
from pathlib import Path
import pandas as pd

from mind.utils.utils import init_logger


class DataPreparer:
    def __init__(
        self,
        preproc_script: Optional[str] = None,
        config_path: Optional[str] = None,
        stw_path: Optional[str] = None,
        python_exe: str = "python3",
        spacy_models: Optional[Dict[str, str]] = None,
        schema: Optional[Dict[str, str]] = None,
        config_logger_path: Path = Path("config/config.yaml"),
        logger=None
    ):
        self._logger = logger if logger else init_logger(
            config_logger_path, __name__)

        # configure NLPipe
        self.preproc_script = Path(preproc_script) if preproc_script else None
        self.config_path = Path(config_path) if config_path else None
        self.stw_path = Path(stw_path) if stw_path else None
        self.python_exe = python_exe
        self.spacy_models = {k.upper(): v for k, v in (
            spacy_models or {}).items()}

        # Schema mapping: user must provide all required fields; optional fields are preserved automatically
        required_fields = ['chunk_id', 'text', 'lang', 'full_doc']
        msg_fails = """
        You must provide a schema mapping all required fields ('chunk_id', 'text', 'lang', 'full_doc') to column names:
        - 'chunk_id': unique id for each chunk/passage
        - 'text': text content of the chunk
        - 'lang': language code (e.g., EN, ES)
        - 'full_doc': full document before chunking
        Any other columns in your input file will be preserved as extra metadata.\n
        Example: schema = {
            'chunk_id': 'id_preproc',
            'text': 'chunk_text',
            'lang': 'language',
            'full_doc': 'summary'
        }
        """
        if not schema or not all(f in schema for f in required_fields):
            raise ValueError(msg_fails)
        self.schema = schema

    @staticmethod
    def _upper_lang(x) -> str:
        return (x if isinstance(x, str) else str(x)).strip().upper()

    def _infer_lang_code(self, df: pd.DataFrame) -> str:
        """
        
        """
        lang_col = self.schema.get("lang")
        if not lang_col or lang_col not in df.columns:
            raise ValueError(
                "Schema must provide a valid 'lang' column name present in the DataFrame.")
        if df[lang_col].notna().any():
            return self._upper_lang(df[lang_col].dropna().iloc[0])
        # Fallback: guess from chunk_id patterns (only if chunk_id provided)
        cid = self.schema.get("chunk_id")
        if cid and cid in df.columns:
            for v in df[cid].dropna().astype(str):
                m = re.match(r"^(?:T_)?([A-Za-z]{2})_", v)
                if m:
                    return m.group(1).upper()
        return "XX"

    def _derive_doc_id(self, row: pd.Series, doc_col: Optional[str], chunk_col: str) -> str:
        if doc_col and pd.notna(row.get(doc_col)):
            return str(row[doc_col])
        # Derive from chunk id by dropping final _<number>
        cid = str(row.get(chunk_col, ""))
        m = re.match(r"^(.*)_(\d+)$", cid)
        return m.group(1) if m else cid

    # ---------- decontamination & ordering ----------

    def _starts_with(self, series: pd.Series, prefix: str) -> pd.Series:
        return series.fillna("").astype(str).str.startswith(prefix)

    def _decontaminate(self, df: pd.DataFrame, self_lang: str, other_lang: str) -> pd.DataFrame:
        chunk_col = self.schema.get("chunk_id")
        if not chunk_col or chunk_col not in df.columns:
            raise ValueError(
                "Schema must provide a valid 'chunk_id' column name present in the DataFrame.")
        raw_col = "raw_text" if "raw_text" in df.columns else None
        out = df.copy()
        ip = out[chunk_col].astype(str)
        mask_self_trans = self._starts_with(ip, f"T_{self_lang}_")
        mask_other_stray = self._starts_with(ip, f"{other_lang}_")
        out = out[~(mask_self_trans | mask_other_stray)].copy()
        if raw_col:
            out = out[~out[raw_col].str.contains("isbn", case=False, na=False)]
        return out.reset_index(drop=True)

    def _stable_order(self, df: pd.DataFrame) -> pd.DataFrame:
        chunk_col = self.schema.get("chunk_id")
        if not chunk_col or chunk_col not in df.columns:
            raise ValueError(
                "Schema must provide a valid 'chunk_id' column name present in the DataFrame.")
        ip = df[chunk_col].astype(str)
        is_translated = ip.str.startswith("T_")
        return pd.concat([df[is_translated], df[~is_translated]], ignore_index=True)

    # ---------- normalization ----------

    def _normalize(self, df: pd.DataFrame, force_lang: Optional[str] = None) -> pd.DataFrame:
        """
        Rename/mint the standard columns. Preserve all extra metadata columns not used for mapping.
        User must provide all required column names in schema.
        """
        df = df.copy()
        # Required fields
        c_chunk = self.schema.get("chunk_id")
        c_text = self.schema.get("text")
        c_lang = self.schema.get("lang")
        c_full = self.schema.get("full_doc")
        if not c_chunk or not c_text or not c_lang or not c_full:
            raise ValueError(
                "Schema must provide 'chunk_id', 'text', 'lang', and 'full_doc' column names.")
        for col in [c_chunk, c_text, c_lang, c_full]:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' specified in schema is missing from DataFrame.")

        # Optional fields (if present in schema and DataFrame)
        c_doc = self.schema.get("doc_id")
        c_title = self.schema.get("title")
        c_url = self.schema.get("url")
        c_equiv = self.schema.get("equivalence")

        # Build normalized frame
        out = pd.DataFrame({
            "chunk_id": df[c_chunk].astype(str),
            "text": df[c_text],
            "full_doc": df[c_full],
        })

        # doc_id
        if c_doc and c_doc in df.columns:
            out["doc_id"] = df[c_doc].astype(str)
        else:
            out["doc_id"] = df.apply(
                lambda r: self._derive_doc_id(r, c_doc, c_chunk), axis=1)

        # lang
        lang_val = force_lang or (
            df[c_lang].iloc[0] if df[c_lang].notna().any() else "XX")
        out["lang"] = self._upper_lang(lang_val)

        # extras from schema (if present)
        if c_title and c_title in df.columns:
            out["title"] = df[c_title]
        if c_url and c_url in df.columns:
            out["url"] = df[c_url]
        if c_equiv and c_equiv in df.columns:
            out["equivalence"] = df[c_equiv]

        # Keep any extra columns from input that are not mapped by schema
        mapped_cols = set([c_chunk, c_text, c_lang, c_full])
        mapped_cols.update([c for c in [c_doc, c_title, c_url, c_equiv] if c])
        extras = [c for c in df.columns if c not in mapped_cols]
        for c in extras:
            out[c] = df[c]

        return out

    def _spacy_model_for(self, lang_upper: str) -> str:
        if not self.spacy_models or lang_upper not in self.spacy_models:
            raise ValueError(f"No spaCy model configured for '{lang_upper}'. "
                             f"Provide spacy_models like {{'EN':'en_core_web_sm'}}.")
        return self.spacy_models[lang_upper]

    def _preprocess_df(self, df: pd.DataFrame, lang_upper: str, tag: str) -> pd.DataFrame:
        """
        Run NLPipe on a TEMP parquet that exposes exactly the columns it expects:
        id_preproc (← df['chunk_id']), raw_text (← df['text']), lang
        Then merge the returned 'lemmas' back onto the *normalized* df by chunk_id.
        """
        # 0) sanity: normalized columns must exist
        required = {"chunk_id", "text", "lang"}
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"_preprocess_df expects normalized df with columns {required}. "
                f"Missing: {missing}. Did _normalize() set 'chunk_id' from your schema?"
            )

        tmp_dir = (self.storing_path / "_tmp_preproc")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # 1) minimal parquet for NLPipe
        work = pd.DataFrame({
            "id_preproc": df["chunk_id"].astype(str),
            "text":  df["text"],
            "lang":  df["lang"].astype(str),
        })
        tmp_parq = tmp_dir / f"{tag}_{lang_upper}.parquet"
        work.to_parquet(tmp_parq, compression="gzip")

        # 2) run NLPipe (logs to console)
        if self.preproc_script and self.config_path and self.stw_path:
            cmd = [
                self.python_exe, str(self.preproc_script),
                "--source_path", str(tmp_parq),
                "--source_type", "parquet",
                "--source", "mind",
                "--destination_path", str(tmp_parq),
                "--lang", lang_upper.lower(),
                "--spacy_model", self._spacy_model_for(lang_upper),
                "--config_file", str(self.config_path),
                "--stw_path", str(self.stw_path),
            ]
            print("Running NLPipe:", " ".join(cmd))
            # no capture_output -> live console logs
            subprocess.run(cmd, check=True)
            print(f"✓ Preprocessed (lang={lang_upper})")
        else:
            print("Preprocessing skipped (not configured).")
            return df

        # 3) read NLPipe output and merge lemmas back by id_preproc ↔ chunk_id
        proc = pd.read_parquet(tmp_parq)
        if "id_preproc" not in proc.columns or "lemmas" not in proc.columns:
            raise RuntimeError(
                f"NLPipe output missing id_preproc/lemmas; got: {list(proc.columns)}")

        merged = df.merge(
            proc[["id_preproc", "lemmas"]],
            left_on="chunk_id",
            right_on="id_preproc",
            how="left",
            validate="one_to_one",
        )

        to_drop = [c for c in merged.columns if c.startswith("id_preproc")]
        merged = merged.drop(columns=to_drop)

        return merged

    @staticmethod
    def _pair_key_from_chunk_id(chunk_id: str, row_lang: str) -> Tuple[str, str]:
        """
        Returns a stable key representing the *original source* chunk, regardless
        of whether the row is original or translated.

        Examples:
          EN_12_3     + row_lang=EN  -> ("EN", "12_3")
          T_EN_12_3   + row_lang=DE  -> ("EN", "12_3")
          DE_77_0     + row_lang=DE  -> ("DE", "77_0")
          T_DE_77_0   + row_lang=EN  -> ("DE", "77_0")
        """
        s = str(chunk_id)
        m = re.match(r"^T_([A-Za-z]{2})_(.+)$", s)
        if m:
            return (m.group(1).upper(), m.group(2))
        m2 = re.match(r"^([A-Za-z]{2})_(.+)$", s)
        if m2:
            return (m2.group(1).upper(), m2.group(2))
        # fallback: treat whole id as the "rest"
        return (row_lang.upper(), s)

    # ---------- main API ----------

    def format_dataframes(self) -> None:
        self._logger.info("Starting format_dataframes process...")
        self.read_dataframes()

        # Infer language tags
        anchor_lang = self._infer_lang_code(self.anchor_df)
        target_lang = self._infer_lang_code(self.target_df)
        self._logger.info(
            f"Anchor language: {anchor_lang}, Target language: {target_lang}")

        # Clean & order
        self._logger.info(
            "Decontaminating and ordering anchor and target dataframes...")
        anc = self._decontaminate(self.anchor_df, anchor_lang, target_lang)
        tgt = self._decontaminate(self.target_df, target_lang, anchor_lang)
        anc = self._stable_order(anc)
        tgt = self._stable_order(tgt)

        # Normalize columns to a common schema
        self._logger.info("Normalizing columns to a common schema...")
        anc_norm = self._normalize(anc, force_lang=anchor_lang)
        tgt_norm = self._normalize(tgt, force_lang=target_lang)

        # Save per-language temporaries, run NLPipe once per language to fill 'lemmas'
        tmp_dir = (self.storing_path / "_tmp_preproc")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        anc_parq = tmp_dir / f"anchor_{anchor_lang}.parquet"
        tgt_parq = tmp_dir / f"target_{target_lang}.parquet"

        anc_norm.to_parquet(anc_parq, compression="gzip")
        tgt_norm.to_parquet(tgt_parq, compression="gzip")
        self._logger.info(
            f"Saved anchor and target normalized parquets to {tmp_dir}")

        # Run preprocessing per language
        self._logger.info(
            "Running NLPipe preprocessing for anchor and target...")
        anc_proc = self._preprocess_df(anc_norm, anchor_lang, tag="anchor")
        tgt_proc = self._preprocess_df(tgt_norm, target_lang, tag="target")

        # Build pairing keys representing the original source chunk
        def add_pair_key(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            keys: List[Tuple[str, str]] = [
                self._pair_key_from_chunk_id(cid, L)
                for cid, L in zip(df["chunk_id"].astype(str), df["lang"].astype(str))
            ]
            df["pair_src_lang"] = [k[0] for k in keys]
            df["pair_rest"] = [k[1] for k in keys]
            df["pair_key"] = df["pair_src_lang"] + ":" + df["pair_rest"]
            return df

        anc_proc = add_pair_key(anc_proc)
        tgt_proc = add_pair_key(tgt_proc)

        # For EN rows, lemmas_tr should come from DE's translations of EN originals: rows in target with id 'T_EN_...'
        # But thanks to pair_key, we can simply map by pair_key.
        # Build maps from each side's *translation rows* to their lemmas:
        def is_translation_of(lang_src: str, df: pd.DataFrame) -> pd.Series:
            # rows whose chunk_id starts with f"T_{lang_src}_"
            return df["chunk_id"].astype(str).str.startswith(f"T_{lang_src}_")

        anc_trans_map = anc_proc.loc[is_translation_of(target_lang, anc_proc), [
            "pair_key", "lemmas"]].dropna()
        tgt_trans_map = tgt_proc.loc[is_translation_of(anchor_lang, tgt_proc), [
            "pair_key", "lemmas"]].dropna()

        # translations of TARGET in ANCHOR
        map_from_anc = dict(
            zip(anc_trans_map["pair_key"], anc_trans_map["lemmas"]))
        # translations of ANCHOR in TARGET
        map_from_tgt = dict(
            zip(tgt_trans_map["pair_key"], tgt_trans_map["lemmas"]))

        # Fill lemmas_tr:
        # - For anchor originals (pair_key = ANCHOR:<rest>), find lemmas in target translations (map_from_tgt)
        # - For target originals, find lemmas in anchor translations (map_from_anc)
        anc_proc["lemmas_tr"] = anc_proc["pair_key"].map(map_from_tgt)
        tgt_proc["lemmas_tr"] = tgt_proc["pair_key"].map(map_from_anc)

        # keep only originals (no T_* rows) for the final output
        def is_original(df): return ~df["chunk_id"].astype(
            str).str.startswith("T_")
        anc_orig = anc_proc[is_original(anc_proc)].copy()
        tgt_orig = tgt_proc[is_original(tgt_proc)].copy()

        # drop pairing helper columns
        cols_to_drop = ["pair_src_lang", "pair_rest", "pair_key"]
        for c in cols_to_drop:
            if c in anc_orig.columns:
                anc_orig.drop(columns=c, inplace=True)
            if c in tgt_orig.columns:
                tgt_orig.drop(columns=c, inplace=True)

        # final stack
        self.final_df = pd.concat([anc_orig, tgt_orig], ignore_index=True)

        # drop all rows where lemmas is None
        self.final_df = self.final_df[~self.final_df.lemmas.isnull()]
        # replace None in lemmas_tr with empty string
        self.final_df["lemmas_tr"] = self.final_df["lemmas_tr"].fillna("")

        # Save unified parquet
        out_path = self.storing_path / "polylingual_df.parquet"
        self.final_df.to_parquet(out_path, compression="gzip")
        self._logger.info(f"Saved: {out_path.as_posix()}")

    def save_to_parquet(self) -> None:
        if self.final_df is None or self.final_df.empty:
            self._logger.error(
                "final_df is empty. Run format_dataframes() first.")
            raise RuntimeError(
                "final_df is empty. Run format_dataframes() first.")
        out_path = self.storing_path / "polylingual_df.parquet"
        self.final_df.to_parquet(out_path, compression="gzip")
        self._logger.info(f"Saved: {out_path.as_posix()}")
