import logging
import pathlib
from subprocess import check_output
import pandas as pd
import os

class RosieCorpus(object):
    def __init__(
        self,
        paths_data: dict,  # dict como {'EN': path1, 'ES': path2}
        path_preproc: str = "src/corpus_building/preprocessing/NLPipe/nlpipe.py",
        multilingual: bool = True,
        logger: logging.Logger = None
    ) -> None:

        # Logger setup
        self._logger = logger or logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)

        if not paths_data:
            raise ValueError("You must provide at least one path to a data file.")

        self.df_by_lang = {}
        for lang, path in paths_data.items():
            lang = lang.upper()
            if not pathlib.Path(path).exists():
                raise FileNotFoundError(f"File not found: {path}")
            self.df_by_lang[lang] = pd.read_parquet(path)
            self._logger.info(f"-- -- Loaded {len(self.df_by_lang[lang])} rows from {path} for language {lang}")

        self._langs = list(self.df_by_lang.keys())
        self._multilingual = multilingual # Wether the corpus is going to be used for training a PolylingualTM or LdaTM. If PolylingualTM, both the raw and translated corpus will be preprocessed
        self._path_preproc = pathlib.Path(path_preproc)

    def generate_tm_tr_corpus(
        self,
        path_save: str,
        level: str = "passage",
        sample: float = 1.0,
        spacy_models=None,
        column_preproc=None,
        column_id="passage_id"
    ):
        if spacy_models is None:
            spacy_models = {lang: "en_core_web_lg" if lang == "EN" else "es_core_news_lg" for lang in self._langs}
        elif isinstance(spacy_models, list):
            spacy_models = {lang: spacy_models[i] for i, lang in enumerate(self._langs)}

        path_save = pathlib.Path(path_save)
        path_save_preproc = path_save.parent / "preproc"
        path_save_final = path_save.parent / f"{path_save.stem}_{sample}.parquet"

        if path_save_final.exists():
            self._logger.info(f"-- Training corpus already exists at {path_save_final}.")
            return path_save_final

        path_save_preproc.mkdir(parents=True, exist_ok=True)

        dfs_sampled = {}
        for lang in self._langs:
            df = self.df_by_lang[lang].sample(frac=sample)
            #df = df[df["lang"] == ("eng_Latn" if lang == "EN" else "spa_Latn")]
            self._logger.info(f"-- {len(df)} samples for language {lang} after filtering.")
            dfs_sampled[lang] = df

        def get_doc_id(lang, index, row, column_id="passage_id", column_preproc=None):
            pas_to_return = "passage" if column_preproc is None else column_preproc
            if level == "passage":
                return f"{lang}_{index}_{row[column_id]}", pas_to_return
            elif level == "document":
                return f"{lang}_{index}", "contents"
            else:
                raise ValueError(f"Level {level} not recognized.")

        processed_dfs = []
        for lang in self._langs:
            df = dfs_sampled[lang]
            other_lang = [l for l in self._langs if l != lang][0] if self._multilingual else None

            doc_ids, texts, tr_texts = [], [], []
            for index, row in df.iterrows():
                doc_id, text_col = get_doc_id(lang, index, row, column_id, column_preproc)
                doc_ids.append(doc_id)
                texts.append(row[text_col])
                if self._multilingual:
                    tr_texts.append(row["tr_text"])

            new_df = pd.DataFrame({
                "id_preproc": range(len(doc_ids)),
                "doc_id": doc_ids,
                "text": texts,
                "lang": [lang] * len(doc_ids)
            })

            if self._multilingual:
                new_df["tr_text"] = tr_texts

            path_text = path_save_preproc / f"{lang}_{level}_text.parquet"
            path_tr_text = path_save_preproc / f"{lang}_{level}_tr_text.parquet"

            new_df.to_parquet(path_text)
            self._logger.info(f"-- Saved intermediate text to {path_text}")

            if self._multilingual:
                new_df.to_parquet(path_tr_text)
                self._logger.info(f"-- Saved intermediate tr_text to {path_tr_text}")

            # Preprocess original language
            self.preproc_tr_corpus(
                source_path=path_text, lang=lang, spacy_model=spacy_models[lang], column_preproc="text"
            )

            if self._multilingual:
                lang_tr = other_lang
                spacy_model_tr = spacy_models[lang_tr]
                self.preproc_tr_corpus(
                    source_path=path_tr_text, lang=lang_tr, spacy_model=spacy_model_tr, column_preproc="tr_text"
                )

                df_preproc_text = pd.read_parquet(path_text)
                df_preproc_tr_text = pd.read_parquet(path_tr_text).rename(columns={
                    "lemmas": "lemmas_tr",
                    "raw_text": "text_tr"
                })[["id_preproc", "lemmas_tr", "text_tr"]]

                df_preproc_combined = df_preproc_text.merge(df_preproc_tr_text, on="id_preproc", how="left")
                df_preproc_combined.to_parquet(path_save_preproc / f"{lang}_{level}_{sample}.parquet")
            else:
                df_preproc_combined = pd.read_parquet(path_text)
                df_preproc_combined.to_parquet(path_save_preproc / f"{lang}_{level}_{sample}.parquet")

            df_merged = df_preproc_combined.merge(new_df, on="id_preproc", how="left")
            processed_dfs.append(df_merged)

        self._logger.info(f"-- Merging all processed corpora...")
        if self._multilingual:
            final_df = pd.concat(processed_dfs)[
                ["id_preproc", "lemmas", "lemmas_tr", "doc_id", "text", "text_tr", "lang"]
            ]
        else:
            final_df = pd.concat(processed_dfs)[
                ["id_preproc", "lemmas", "doc_id", "text", "lang"]
            ]

        final_df.to_parquet(path_save_final)
        self._logger.info(f"-- Final training corpus saved at {path_save_final} with {len(final_df)} entries.")
        return path_save_final

    def preproc_tr_corpus(
        self,
        source_path,
        lang,
        spacy_model,
        column_preproc,
    ):

        if column_preproc == "text":
            source_type = "rosie"
        elif column_preproc == "tr_text":
            source_type = "rosie_tr"
        
        path_python_str = "python3 "  # f"{path_python.as_posix()} "
        path_nlpipe = os.getcwd() / self._path_preproc

        source_save_path = source_path.resolve().as_posix()
        cmd = path_python_str + path_nlpipe.as_posix() + \
            ' --source_path %s ' \
            '--source_type %s '\
            '--source %s '\
            '--destination_path %s '\
            '--lang %s ' \
            '--spacy_model %s ' \
            '--path_config %s ' \
            '--stw_path %s'
        cmd = cmd % \
            (source_save_path, "parquet", source_type,
             source_save_path, lang.lower(), spacy_model,
             "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/src/corpus_building/config.json",
             "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/src/corpus_building/preprocessing/stw_lists")

        self._logger.info(cmd)

        try:
            self._logger.info(
                f'-- -- Preprocessing corpus {source_save_path}. Command is {cmd}')
            check_output(args=cmd, shell=True)
        except:
            self._logger.error('-- -- Preprocessing corpus. Revise command')
            return

        return
