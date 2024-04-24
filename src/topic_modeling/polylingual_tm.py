"""Python wrapper around the Mallet Polylingual Topic Model implementation.
# https://mallet.cs.umass.edu/topics-polylingual.php

The input to the Mallet Polylingual Topic Model is a corpus of documents in multiple languages. This corpus of documents is given as a dataframe with the following columns:

- `doc_id`: A unique identifier for each document.
- `lang`: The language of the document. 
- `text`: The text of the document.

The output is a folder with the following strucutre:
- model_folder
|--- train_data
|    |--- corpus_lang1.txt
|    |--- corpus_lang2.txt
|--- mallet_input
|    |--- corpus_lang1.mallet
|    |--- corpus_lang2.mallet
|--- mallet_output
|    |--- doc-topics.txt
|    |--- inferencer.mallet.0
|    |--- inferencer.mallet.1
|    |--- output-state.gz
|    |--- topickeys.txt
"""


import logging
import pathlib
from subprocess import check_output
import pandas as pd
import os
import shutil
import numpy as np


class PolylingualTM(object):

    def __init__(
        self,
        lang1: str,
        lang2: str,
        model_folder: str,
        num_topics: int,
        alpha: float = 1.0,
        token_regexp: str = r"\p{L}+",
        mallet_path: str = "src/topic_modeling/Mallet-202108/bin/mallet",
        logger: logging.Logger = None
    ) -> None:
        """Initialize the PolylingualTM object.

        Parameters
        ----------
        lang1 : str
            The first language in the corpus.
        lang2 : str
            The second language in the corpus.
        model_folder : str
            The folder where all information related to the model will be stored.
        num_topics : int
            The number of topics to extract.
        alpha : float
            The alpha hyperparameter for the model.
        token_regexp : str
            The regular expression to use for tokenization.
        mallet_path : str
            The path to the Mallet executable.

        """

        self._lang1 = lang1
        self._lang2 = lang2
        self._num_topics = num_topics
        self._alpha = alpha
        self._token_regexp = token_regexp
        self._model_folder = model_folder
        self._mallet_path = pathlib.Path(mallet_path)

        if logger:
            self._logger = logger
        else:
            logging.basicConfig(level='INFO')
            self._logger = logging.getLogger('PolylingualTM')
            # Add a console handler to output logs to the console
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)  # Set handler level to INFO or lower
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)

        # Create folder for the model
        # If a model with the same name already exists, save a copy; then create a new folder
        if self._model_folder.exists():
            self._logger.info(
                f"-- -- Given model folder {self._model_folder} already exists. Saving a copy ..."
            )
            old_model_folder = self._model_folder.parent / \
                (self._model_folder.name + "_old")
            if not old_model_folder.is_dir():
                os.makedirs(old_model_folder)
                shutil.move(self._model_folder, old_model_folder)

        self._model_folder.mkdir(exist_ok=True)

        return

    def _create_mallet_input_corpus(
        self,
        df_path: pathlib.Path,
    ) -> int:
        """Given a corpus of documents, create the input files for the Mallet Polylingual Topic Model, i.e., creates 'corpus_lang1.txt' and 'corpus_lang2.txt' files.

        Parameters
        ----------
        df_path : pd.DataFrame
            Path to a dataframe with the following columns: `doc_id`, `lang`, and `raw_text`.

        Returns
        -------
        status : int
            Status of the operation:
            - 2 if the operation was successful.
            - 1 if the input format is incorrect.
            - 0 if the operation failed.
        """

        # Create 'train_data' folder
        self._train_data_folder = self._model_folder / "train_data"
        self._train_data_folder.mkdir(exist_ok=True)

        # Read the dataframe and create the input files
        df = pd.read_parquet(df_path)
        for lang in [self._lang1, self._lang2]:
            df_lang = df.copy()
            #import pdb; pdb.set_trace()
            
            df_lang["lemmas"] = np.where(df_lang["lang"] != lang, df_lang["lemmas_tr"], df_lang["lemmas"])

            if df_lang.empty:
                self._logger.error(
                    f"-- -- No documents found for language {lang}.")
                return 1
            corpus_txt_path = self._train_data_folder / f"corpus_{lang}.txt"
            self._logger.info(
                f"-- -- Creating Mallet {corpus_txt_path.as_posix()}...")
            with corpus_txt_path.open("w", encoding="utf8") as fout:
                for i, t in zip(df_lang.doc_id, df_lang.lemmas):
                    fout.write(f"{i} {lang.upper()} {t}\n")
            self._logger.info(
                f"-- -- Mallet {corpus_txt_path.as_posix()} created.")

        if (self._train_data_folder / "corpus_lang1.txt").exists() and (self._train_data_folder / "corpus_lang2.txt").exists():
            return 2
        else:
            return 0

    def _prepare_mallet_input(
        self,
    ) -> int:
        """Assuming there is a 'corpus_es.txt' and 'corpus_en.txt' files in the model folder, prepare the input files for the Mallet Polylingual Topic Model, i.e., generates the files 'corpus_lang1.mallet' and 'corpus_lang2.mallet'.

        Returns
        -------
        status : int
            Status of the operation:
            - 2 if the operation was successful.
            - 1 if the input files are missing.
            - 0 if the operation failed.
        """

        # Create folder for saving Mallet input files
        self._mallet_input_folder = self._model_folder / "mallet_input"
        self._mallet_input_folder.mkdir(exist_ok=True)

        # Transform training data into the format expected by Mallet
        self._logger.info(f"-- -- Importing data to Mallet...")
        for lang in [self._lang1, self._lang2]:
            corpus_txt_path = self._train_data_folder / f"corpus_{lang}.txt"
            corpus_mallet = self._mallet_input_folder / f"corpus_{lang}.mallet"

            cmd = self._mallet_path.as_posix() + \
                ' import-file --preserve-case --keep-sequence ' + \
                '--remove-stopwords --token-regex "' + self._token_regexp + \
                '" --print-output --input %s --output %s'
            cmd = cmd % (corpus_txt_path, corpus_mallet)

            try:
                self._logger.info(f'-- -- Running command {cmd}')
                check_output(args=cmd, shell=True)
            except:
                self._logger.error(
                    '-- -- Mallet failed to import data. Revise command')
                return 0

            self._logger.info(f"-- -- Data imported to Mallet.")

        if (self._mallet_input_folder / "corpus_lang1.mallet").exists() and (self._mallet_input_folder / "corpus_lang2.mallet").exists():
            return 2
        else:
            return 1

    def train(
        self,
        df_path: pathlib.Path,
    ) -> int:
        """Assuming there is a 'corpus_es.mallet' and 'corpus_en.mallet' files in the model folder, train the Mallet Polylingual Topic Model.

        Returns
        -------
        status : int
            Status of the operation:
            - 2 if the operation was successful.
            - 1 if the input files are missing.
            - 0 if the operation failed.
        """

        # Create the input files for Mallet
        self._create_mallet_input_corpus(df_path)

        # Prepare the input files for Mallet
        self._prepare_mallet_input()

        # Create folder for saving Mallet output
        self._mallet_out_folder = self._model_folder / "mallet_output"
        self._mallet_out_folder.mkdir(exist_ok=True)

        # Actual training of the model
        mallet_input_files = [self._mallet_input_folder /
                              f"corpus_{lang}.mallet" for lang in [self._lang1, self._lang2]]
        language_inputs = ' '.join(
            [file.resolve().as_posix() for file in mallet_input_files])
        cmd = self._mallet_path.as_posix() + \
            ' run cc.mallet.topics.PolylingualTopicModel  ' + \
            '--language-inputs %s --num-topics %s --alpha %s ' + \
            '--output-state %s '\
            '--output-doc-topics %s '\
            '--output-topic-keys %s '\
            '--inferencer-filename %s '
        cmd = cmd % \
            (language_inputs, self._num_topics, self._alpha,
             (self._mallet_out_folder / "output-state.gz").resolve().as_posix(),
             (self._mallet_out_folder / "doc-topics.txt").resolve().as_posix(),
             (self._mallet_out_folder / "topickeys.txt").resolve().as_posix(),
             (self._mallet_out_folder / "inferencer.mallet").resolve().as_posix(),
             )

        try:
            self._logger.info(
                f'-- -- Training Mallet PolylingualTopicModel. Command is {cmd}')
            check_output(args=cmd, shell=True)
        except:
            self._logger.error('-- -- Model training failed. Revise command')
            return
