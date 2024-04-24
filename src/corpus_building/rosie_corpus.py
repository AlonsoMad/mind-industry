import logging
import pathlib
from subprocess import check_output
import pandas as pd
import os
from transformers import MarianMTModel, MarianTokenizer


class RosieCorpus(object):
    def __init__(
        self,
        path_data_en: str,
        path_data_es: str,
        path_preproc: str = "src/corpus_building/preprocessing/NLPipe/nlpipe.py",
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
        self.df_en = pd.read_json(path_data_en, lines=True)
        self._logger.info(
            f"-- -- {len(self.df_en)} elements read from {path_data_en}")
        self.df_es = pd.read_json(path_data_es, lines=True)
        self._logger.info(
            f"-- -- {len(self.df_es)} elements read from {path_data_es}")

        self._path_preproc = pathlib.Path(path_preproc)

        return

    def generate_tm_tr_corpus(
        self,
        path_save: str,
        level: str = "passage",
        sample: float = 1.0,
        spacy_models=["en_core_web_sm", "es_core_news_sm"]
    ):
        """
        Generate a training corpus for the Polylingual Topic Model from the Rosie corpus (json files in English and Spanish at either document or passage level).

        Parameters
        ----------
        path_save : str
            Path to save the training corpus.
        level : str
            Level of the corpus to use, either "document" or "passage".
        sample : float
            Fraction of the corpus to use. Default is 1.0.
        """

        self._logger.info(
            f"-- Generating training corpus at {level} level with a sample of {sample}...")
        df_en_sample = self.df_en.sample(frac=sample)
        df_es_sample = self.df_es.sample(frac=sample)

        self._logger.info(
            f"-- {len(df_en_sample)} elements in English and {len(df_es_sample)} elements in Spanish.")

        # Create intermediate files for preprocessing if they don't exist
        path_save = pathlib.Path(path_save)
        path_save_preproc = path_save.parent / "preproc"
        path_save_preproc.mkdir(exist_ok=True)
        path_save = path_save.parent / pathlib.Path(pathlib.Path(
            path_save).stem + f"_{sample}.parquet")
        self._logger.info(
            f"-- Final preprocessed corpus for sample {sample} will be saved at {path_save}")
        self._logger.info(
            f"-- Intermediate files for preprocessing will be saved at {path_save_preproc}")
        path_save_en = path_save_preproc / f"en_{level}_{sample}.parquet"
        path_save_es = path_save_preproc / f"es_{level}_{sample}.parquet"

        # Create intermediate files for saving translated corpus
        # We will save, for each language, the translated corpus in a parquet files. When this method is called again, we will check if the documents in the sample have already been translated and saved in these files in order to avoid translating them again.
        path_save_tr = path_save.parent / "translated"
        path_save_tr.mkdir(exist_ok=True)
        path_save_en_tr = path_save_tr / f"en_{level}.parquet"
        path_save_es_tr = path_save_tr / f"es_{level}.parquet"

        # Carry out translation
        if not path_save_en_tr.exists() or not path_save_es_tr.exists():

            # Translate the documents
            self._logger.info(
                f"-- -- Translating documents to English and Spanish...")
            df_en_sample_tr = self.translate_corpus(
                df_en_sample, source_language="en", target_language="es")
            df_es_sample_tr = self.translate_corpus(
                df_es_sample, source_language="es", target_language="en")

            # Save the translated documents
            self._logger.info(
                f"-- -- Saving translated documents at {path_save_en_tr} and {path_save_es_tr}...")
            df_en_sample_tr.to_parquet(path_save_en_tr)
            df_es_sample_tr.to_parquet(path_save_es_tr)

            # Merge translated documents with original documents
            self._logger.info(
                f"-- -- Merging translated documents with original documents...")
            df_en_sample = df_en_sample.merge(
                df_en_sample_tr, how="left", on="doc_id")
            df_es_sample = df_es_sample.merge(
                df_es_sample_tr, how="left", on="doc_id")

        else:
            self._logger.info(
                f"-- -- Translated documents already exist. Loading from {path_save_en_tr} and {path_save_es_tr}...")
            df_en_sample_tr = pd.read_parquet(path_save_en_tr)
            self._logger.info(
                f"-- -- English translated corpus loaded. There are {len(df_en_sample_tr[df_en_sample['raw_text_tr'].notnull()])} /{len(df_en_sample)} translated elements."
            )
            df_es_sample_tr = pd.read_parquet(path_save_es_tr)
            self._logger.info(
                f"-- -- Spanish translated corpus loaded. There are {len(df_es_sample_tr[df_es_sample['raw_text_tr'].notnull()])} /{len(df_es_sample)} translated elements."
            )

            # Merge translated documents with original documents
            self._logger.info(
                f"-- -- Merging translated documents with original documents...")
            df_en_sample = df_en_sample.merge(
                df_en_sample_tr, how="left", on="doc_id")
            df_es_sample = df_es_sample.merge(
                df_es_sample_tr, how="left", on="doc_id")

            # Translate the documents that were not translated
            self._logger.info(
                f"-- -- Translating documents that were not translated...")
            df_en_sample_tr = self.translate_corpus(
                df_en_sample[df_en_sample["raw_text_tr"].isnull()], source_language="en", target_language="es")
            df_es_sample_tr = self.translate_corpus(
                df_es_sample[df_es_sample["raw_text_tr"].isnull()], source_language="es", target_language="en")

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

        if not path_save_en.exists() or not path_save_es.exists() or not path_save.exists():

            self._logger.info(
                f"-- -- Saving intermediate files for preprocessing at {path_save_preproc}...")

            dicts_to_df = []
            for id, (df, lang, path_save_) in enumerate(tuples_get):
                self._logger.info(
                    f"-- -- Generating training corpus at {level} level...")
                doc_ids, texts = [], []
                for index, row in df.iterrows():
                    doc_id, text_col = get_doc_id(lang, index, row, level)
                    doc_ids.append(doc_id)
                    texts.append(row[text_col])

                new_df = pd.DataFrame({
                    "id_preproc": range(len(doc_ids)),
                    "doc_id": doc_ids,
                    "text": texts,
                    "lang": [lang] * len(doc_ids)
                    # TODO: Add translation
                })

                if path_save_en.exists() and path_save_es.exists():

                    self._logger.info(
                        f"-- -- Intermediate files for preprocessing already exist. Loading from {path_save_}...")
                else:
                    # Save intermediate files for preprocessing
                    self._logger.info(
                        f"-- -- Saving intermediate files for preprocessing at {path_save_}...")
                    new_df.to_parquet(path_save_)

                    # Carry out preprocessing
                    self._logger.info(f"-- -- Preprocessing {lang} corpus...")
                    self.preproc_tr_corpus(
                        source_path=path_save_, lang=lang, spacy_model=spacy_models[id])

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
        target_language: str = "es"
    ) -> pd.DataFrame:
        """
        Translate a corpus from a source language to a target language.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with the corpus to translate.
        source_language : str
            Source language of the corpus. Default is "en".
        target_language : str
            Target language of the corpus. Default is "es".

        Returns
        -------
        pd.DataFrame
            DataFrame with the translated corpus. The columns are the same as the input DataFrame, plus a new column "raw_text_tr" with the translated text.
        """

        # Function to translate a sentence using the provided tokenizer and model
        def translate_sentence(tokenizer, model, sentence):
            inputs = tokenizer.encode(sentence, return_tensors="pt")
            outputs = model.generate(inputs, early_stopping=True)
            translated_text = tokenizer.decode(
                outputs[0], skip_special_tokens=True)
            return translated_text

        # Load the translation model and tokenizer
        model_name = f'Helsinki-NLP/opus-mt-{source_language}-{target_language}'
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

        # Function to translate a text, handling chunking if necessary
        def translate(text):
            if len(text) > tokenizer.model_max_length:
                texts_splits = [text[i:i + tokenizer.model_max_length]
                                for i in range(0, len(text), tokenizer.model_max_length)]
            else:
                texts_splits = [text]

            translations = [translate_sentence(
                tokenizer, model, sentence) for sentence in texts_splits]
            return ' '.join(translations)

        # Apply translation function to each text in the DataFrame
        df["raw_text_tr"] = df["raw_text"].apply(translate)
        return df

    def preproc_tr_corpus(
        self,
        source_path,
        lang,
        spacy_model
    ):

        # path_python = (self._path_preproc.parent.parent / ".venv_nlpipe/bin/python3")
        # path_python = pathlib.Path("/Users/lbartolome/Documents/GitHub/NLPipe/.venv/bin/python3")
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
            (source_save_path, "parquet", "rosie",
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
