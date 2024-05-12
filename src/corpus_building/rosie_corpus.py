import logging
import pathlib
from subprocess import check_output
import pandas as pd
import os
from transformers import MarianMTModel, MarianTokenizer
from transformers import pipeline as HF_pipeline
import torch


class RosieCorpus(object):
    def __init__(
        self,
        path_data_en: str,
        path_data_es: str,
        path_preproc: str = "src/corpus_building/preprocessing/NLPipe/nlpipe.py",
        batch_size: int = 256,
        translate: bool = False,
        logger: logging.Logger = None
    ) -> None:

        if logger is not None:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)
            self._logger.setLevel(logging.INFO)
            # Add a console handler to output logs to the console
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)

        if not pathlib.Path(path_data_en).exists():
            raise FileNotFoundError(f"File not found: {path_data_en}")
        if not pathlib.Path(path_data_es).exists():
            raise FileNotFoundError(f"File not found: {path_data_es}")

        self._logger.info(
            f"-- -- Reading data from {path_data_en} and {path_data_es}")
        # pd.read_json(path_data_en, lines=True)
        self.df_en = pd.read_parquet(path_data_en)
        self._logger.info(
            f"-- -- {len(self.df_en)} elements read from {path_data_en}")
        # pd.read_json(path_data_es, lines=True)
        self.df_es = pd.read_parquet(path_data_es)
        self._logger.info(
            f"-- -- {len(self.df_es)} elements read from {path_data_es}")

        self._path_preproc = pathlib.Path(path_preproc)

        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self._translate = translate
        if self._translate:
            # Load the translation model and tokenizer
            self._tr_tokenizers = {}
            self._tr_models = {}
            self._tr_pipelines = {}
            for lang in ["es", "en"]:
                if lang == "es":
                    target = "en"
                elif lang == "en":
                    target = "es"
                model_name = \
                    f'Helsinki-NLP/opus-mt-{lang}-{target}'
                try:
                    tr_pipeline = HF_pipeline(
                        'translation', model=model_name, batch_size=batch_size, device=self._device)
                    tokenizer = MarianTokenizer.from_pretrained(model_name)
                except:
                    tr_pipeline = HF_pipeline(
                        'translation', model=model_name, batch_size=batch_size, device=self._device)
                    tokenizer = MarianTokenizer.from_pretrained(
                        model_name, force_download=True)
                self._tr_tokenizers[lang] = tokenizer
                self._tr_pipelines[lang] = tr_pipeline
        return

    def generate_tm_tr_corpus(
        self,
        path_save: str,
        level: str = "passage",
        sample: float = 1.0,
        spacy_models=["en_core_web_lg", "es_core_news_lg"]
    ):
        """
        Generate a training corpus for the Polylingual Topic Model LDA Topic Model (2 corpus version) from the Rosie corpus (json files in English and Spanish at either document or passage level).

        Parameters
        ----------
        path_save : str
            Path to save the training corpus.
        level : str
            Level of the corpus to use, either "document" or "passage".
        sample : float
            Fraction of the corpus to use. Default is 1.0.
        """

        #######################################################################
        # READ DATA AND CREATE FOLDER STRUCTURE
        #######################################################################
        # Check if a corpus for this sample size and level has already been generated. If yes, use it, if not, generate it.
        path_save = pathlib.Path(path_save)
        path_save_preproc = path_save.parent / "preproc"
        path_save = path_save.parent / pathlib.Path(pathlib.Path(
            path_save).stem + f"_{sample}.parquet")

        if path_save.is_file():
            self._logger.info(
                f"-- Training corpus for sample {sample} and level {level} already exists. Loading from {path_save}...")
            return path_save
        else:
            self._logger.info(
                f"-- Training corpus for sample {sample} and level {level} does not exist. Generating...")

            # Create intermediate files for preprocessing if they don't exist
            path_save_preproc.mkdir(exist_ok=True)
            self._logger.info(
                f"-- Final preprocessed corpus for sample {sample} will be saved at {path_save}")
            self._logger.info(
                f"-- Intermediate files for preprocessing will be saved at {path_save_preproc}")
            path_save_en = path_save_preproc / f"en_{level}_{sample}.parquet"
            path_save_es = path_save_preproc / f"es_{level}_{sample}.parquet"

            if self._translate:
                # Create intermediate files for saving translated corpus
                # We will save, for each language, the translated corpus in a parquet files. When this method is called again, we will check if the documents in the sample have already been translated and saved in these files in order to avoid translating them again.
                path_save_tr = path_save.parent / "translated"
                path_save_tr.mkdir(exist_ok=True)
                path_save_en_tr = path_save_tr / f"en_{level}.parquet"
                path_save_es_tr = path_save_tr / f"es_{level}.parquet"

                # Column to translate
                if level == "passage":
                    col_tr = level
                elif level == "document":
                    col_tr = "contents"

        # Get the sample of the corpus
        self._logger.info(
            f"-- Generating training corpus at {level} level with a sample of {sample}...")
        df_en_sample = self.df_en.sample(frac=sample)
        df_es_sample = self.df_es.sample(frac=sample)
        self._logger.info(
            f"-- {len(df_en_sample)} elements in English and {len(df_es_sample)} elements in Spanish.")

        # Keep documents from the english corpus that are indeed in english and from the spanish corpus that are indeed in spanish
        df_en_sample = df_en_sample[df_en_sample["lang"] == "eng_Latn"]
        df_es_sample = df_es_sample[df_es_sample["lang"] == "spa_Latn"]
        self._logger.info(
            f"-- {len(df_en_sample)} elements in English and {len(df_es_sample)} elements in Spanish after filtering by language."
        )

        #######################################################################
        # TRANSLATION (OPTIONAL)
        #######################################################################
        if self._translate:
            # Carry out translation from the beginning if the translated files don't exist
            if not path_save_en_tr.exists() or not path_save_es_tr.exists():

                # Translate the documents
                self._logger.info(
                    f"-- -- Translating documents to English and Spanish...")
                df_en_sample = self.translate_corpus(
                    df_en_sample, source_language="en", target_language="es", col_tr=col_tr)
                df_es_sample = self.translate_corpus(
                    df_es_sample, source_language="es", target_language="en", col_tr=col_tr)

                # Save the translated documents
                self._logger.info(
                    f"-- -- Saving translated documents at {path_save_en_tr} and {path_save_es_tr}...")
                df_en_sample.to_parquet(path_save_en_tr)
                df_es_sample.to_parquet(path_save_es_tr)

            else:
                # Load the translated documents if they already exist
                self._logger.info(
                    f"-- -- Translated documents already exist. Loading from {path_save_en_tr} and {path_save_es_tr}...")

                df_en_sample_tr = pd.read_parquet(path_save_en_tr)
                ids_translated_en = df_en_sample_tr["document_id"].values.tolist(
                )
                not_translated_en = df_en_sample[~df_en_sample["document_id"].isin(
                    ids_translated_en)]
                self._logger.info(
                    f"-- -- English translated corpus loaded. There are {len(not_translated_en)} / {len(df_en_sample)} not translated elements."
                )
                df_es_sample_tr = pd.read_parquet(path_save_es_tr)
                ids_translated_es = df_es_sample_tr["document_id"].values.tolist(
                )
                not_translated_es = df_es_sample[~df_es_sample["document_id"].isin(
                    ids_translated_es)]
                self._logger.info(
                    f"-- -- Spanish translated corpus loaded. There are {len(not_translated_es)} / {len(df_es_sample)} not translated elements."
                )

                # Merge translated documents with original documents
                self._logger.info(
                    f"-- -- Combining translated documents with original documents...")
                df_en_sample = pd.concat(
                    [df_en_sample, df_en_sample_tr], ignore_index=True)
                df_en_sample = df_en_sample.drop_duplicates(
                    subset='document_id', keep='last')
                df_es_sample = pd.concat(
                    [df_es_sample, df_es_sample_tr], ignore_index=True)
                df_es_sample = df_es_sample.drop_duplicates(
                    subset='document_id', keep='last')

                # Translate the documents that were not translated
                self._logger.info(
                    f"-- -- Translating documents that were not translated...")
                df_en_sample_tr = self.translate_corpus(
                    df_en_sample[df_en_sample["raw_text_tr"].isnull()], source_language="en", target_language="es", col_tr=col_tr)
                df_es_sample_tr = self.translate_corpus(
                    df_es_sample[df_es_sample["raw_text_tr"].isnull()], source_language="es", target_language="en", col_tr=col_tr)

                # Update the translated documents
                self._logger.info(
                    f"-- -- Updating translated documents...")
                df_en_sample.update(df_en_sample_tr)
                df_es_sample.update(df_es_sample_tr)

                # Save the updated translated documents
                self._logger.info(
                    f"-- -- Saving updated translated documents at {path_save_en_tr} and {path_save_es_tr}...")
                df_en_sample.to_parquet(path_save_en_tr)
                df_es_sample.to_parquet(path_save_es_tr)

        tuples_get = [
            (df_en_sample, "EN", path_save_en),
            (df_es_sample, "ES", path_save_es)]

        def get_doc_id(lang, index, row, level):
            if level == "passage":
                return f"{lang}_{index}_{row.passage_id}", "passage"
            elif level == "document":
                return f"{lang}_{index}", "contents"
            else:
                raise ValueError(f"Level {level} not recognized.")

        #######################################################################
        # PREPROCESSING
        #######################################################################
        if not path_save_en.exists() or not path_save_es.exists() or not path_save.exists():

            self._logger.info(
                f"-- -- Saving intermediate files for preprocessing at {path_save_preproc}...")

            dicts_to_df = []
            for id, (df, lang, path_save_) in enumerate(tuples_get):
                self._logger.info(
                    f"-- -- Generating training corpus at {level} level...")
                doc_ids, texts, tr_texts = [], [], []
                for index, row in df.iterrows():
                    doc_id, text_col = get_doc_id(lang, index, row, level)
                    doc_ids.append(doc_id)
                    texts.append(row[text_col])
                    if self._translate:
                        tr_texts.append(row["raw_text_tr"])

                if self._translate:
                    new_df = pd.DataFrame({
                        "id_preproc": range(len(doc_ids)),
                        "doc_id": doc_ids,
                        "text": texts,
                        "lang": [lang] * len(doc_ids),
                        "tr_text": tr_texts
                    })
                else:
                    new_df = pd.DataFrame({
                        "id_preproc": range(len(doc_ids)),
                        "doc_id": doc_ids,
                        "text": texts,
                        "lang": [lang] * len(doc_ids)
                    })

                if path_save_en.exists() and path_save_es.exists():

                    self._logger.info(
                        f"-- -- Intermediate files for preprocessing already exist. Loading from {path_save_}...")
                else:
                    # Save intermediate files for preprocessing
                    path_save_text = \
                        path_save_.parent / f"{lang}_{level}_text.parquet"
                    new_df.to_parquet(path_save_text)
                    self._logger.info(
                        f"-- -- Saving intermediate files for preprocessing at {path_save_text}...")

                    if self._translate:
                        path_save_tr_text = \
                            path_save_.parent / \
                            f"{lang}_{level}_tr_text.parquet"
                        new_df.to_parquet(path_save_tr_text)
                        self._logger.info(
                            f"-- -- Saving intermediate files for preprocessing of translated text at {path_save_tr_text}...")

                    # Carry out preprocessing for "text"
                    self._logger.info(
                        f"-- -- Preprocessing {lang} corpus in original language...")
                    self.preproc_tr_corpus(
                        source_path=path_save_text, lang=lang, spacy_model=spacy_models[id],
                        column_preproc="text")

                    if self._translate:
                        # Carry out preprocessing for "tr_text"
                        self._logger.info(
                            f"-- -- Preprocessing {lang} corpus in translated language...")
                        if lang == "EN":
                            lang_tr = "ES"
                        elif lang == "ES":
                            lang_tr = "EN"
                        if id == 0:
                            spacy_model_tr = spacy_models[1]
                        elif id == 1:
                            spacy_model_tr = spacy_models[0]
                        self.preproc_tr_corpus(
                            source_path=path_save_tr_text, lang=lang_tr, spacy_model=spacy_model_tr,
                            column_preproc="tr_text")

                        # Merge preprocessed data and save it
                        self._logger.info(
                            f"-- -- Merging preprocessed {lang} corpus with preprocessed translated data and saving at {path_save_.as_posix()}...")
                        df_preproc_text = pd.read_parquet(path_save_text)
                        df_preproc_tr_text = pd.read_parquet(path_save_tr_text).rename(columns={
                            "lemmas": "lemmas_tr",
                            "raw_text": "text_tr"})[["id_preproc", "lemmas_tr", "text_tr"]]
                        df_preproc_with_tr = df_preproc_text.merge(
                            df_preproc_tr_text, how="left", on="id_preproc")
                        df_preproc_with_tr.to_parquet(path_save_)

                    else:
                        df_preproc_text = pd.read_parquet(path_save_text)
                        df_preproc_text.to_parquet(path_save_)

                # Get preprocessed dataframe and append to list
                self._logger.info(
                    f"-- -- Loading preprocessed {lang} corpus...")
                df_preproc = pd.read_parquet(path_save_)
                self._logger.info(
                    f"-- -- Merging {lang} corpus with preprocessed data...")
                df_preproc_with_all_info = df_preproc.merge(
                    new_df, how="left", on="id_preproc")
                self._logger.info(
                    f"-- -- {len(df_preproc_with_all_info)} elements in {lang} corpus.")
                dicts_to_df.append(df_preproc_with_all_info)

            self._logger.info(f"-- -- Merging both corpora...")
            if self._translate:
                final_df = pd.concat(dicts_to_df)[
                    ['id_preproc', 'lemmas', 'lemmas_tr', 'doc_id', 'text', 'text_tr', 'lang']]
            else:
                final_df = pd.concat(dicts_to_df)[
                    ['id_preproc', 'lemmas', 'doc_id', 'text', 'lang']]
            self._logger.info(
                f"-- -- Showing some samples of the final dataframe...")
            self._logger.info(f"{final_df.head()}")
            self._logger.info(
                f"-- -- Training corpus generated. Nr elements is {len(dicts_to_df[0])} in {tuples_get[0][1]} and {len(dicts_to_df[1])} in {tuples_get[1][1]}. TOTAL: {len(final_df)}. Saving at {path_save}...")

            try:
                final_df.to_parquet(path_save)
            except:
                self._logger.error(
                    f"-- -- Training corpus could not be saved at {path_save}.")
                return
        else:
            self._logger.info(
                f"-- Intermediate files for preprocessing already exist. Loading from {path_save}...")

        return path_save

    def translate_corpus(
        self,
        df: pd.DataFrame,
        source_language: str = "en",
        col_tr: str = "raw_text"
    ) -> pd.DataFrame:
        """
        Translate a corpus from a source language to a target language.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with the corpus to translate.
        source_language : str
            Source language of the corpus. Default is "en".

        Returns
        -------
        pd.DataFrame
            DataFrame with the translated corpus. The columns are the same as the input DataFrame, plus a new column "raw_text_tr" with the translated text.
        """

        # Get tokenizer and model for the language
        tokenizer = self._tr_tokenizers[source_language]
        # model = self._tr_models[source_language]
        tr_pipeline = self._tr_pipelines[source_language]

        # Function to translate a sentence using the provided tokenizer and model
        def translate_sentence(sentence):
            translated_text = tr_pipeline(sentence)[0]["translation_text"]
            return translated_text

        # Function to translate a text, handling chunking if necessary
        def translate(text):
            if len(text) > tokenizer.model_max_length:
                texts_splits = [text[i:i + tokenizer.model_max_length]
                                for i in range(0, len(text), tokenizer.model_max_length)]
            else:
                texts_splits = [text]

            translations = [translate_sentence(
                sentence) for sentence in texts_splits]
            return ' '.join(translations)

        # Apply translation function to each text in the DataFrame
        df_copy = df.copy()
        df_copy["raw_text_tr"] = df_copy[col_tr].apply(translate)

        return df_copy

    def preproc_tr_corpus(
        self,
        source_path,
        lang,
        spacy_model,
        column_preproc
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

        # Luego hay que volver a juntarlo para tener los "empties"

        return
