import logging
import pathlib
from typing import Optional, List, Tuple

import pandas as pd
import gzip
from scipy import sparse

from src.topic_modeling.polylingual_tm import PolylingualTM
from src.utils.utils import init_logger


class HierarchicalTM(object):
    """
    Class for the creation of hierarchical topic models. It generates the corpus of a second-level submodel based on the chosen hierarchical algorithm and the specified first-level topic model.
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        path_logs: pathlib.Path = pathlib.Path(
            __file__).parent.parent.parent.parent / "data/logs"
    ) -> None:
        """
        Initialize the HierarchicalTM class.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger object to log activity.
        path_logs : pathlib.Path, optional
            Path for saving logs.
        """
        self._logger = logger if logger else init_logger(__name__, path_logs)

    def create_submodel_tr_corpus(
        self,
        father_model_path: pathlib.Path,
        langs: List[str],
        exp_tpc: int,
        tr_topics: str,
        htm_version: str,
        thr: float = None
    ) -> None:
        """
        Create submodel training corpus.

        Parameters
        ----------
        father_model_path: pathlib.Path
            Path to the TModel object associated with the father model.
        langs: List[str]
            List of languages to process.
        exp_tpc: int
            Topic number from the father model to be expanded.
        tr_topics: str
            Training topics.
        htm_version: str
            Version of hierarchical topic modeling ('htm-ws' or 'htm_ds').
        thr: float, optional
            Threshold value for filtering documents.
        """
        
        if isinstance(father_model_path, str):
            father_model_path = pathlib.Path(father_model_path)

        # Create necessary directories
        submodel_path = self.create_directories(father_model_path, htm_version, exp_tpc, tr_topics)

        if htm_version == "htm_ws":
            self._logger.info('-- -- Creating training corpus according to HTM-WS.')
            self.create_htm_ws_corpus(father_model_path, submodel_path, langs, exp_tpc)
        elif htm_version == "htm_ds":
            self._logger.info('-- -- Creating training corpus according to HTM-DS.')
            self.create_htm_ds_corpus(father_model_path, submodel_path, langs, exp_tpc, thr)
            
        return submodel_path

    def create_directories(
        self,
        father_model_path: pathlib.Path,
        htm_version: str,
        exp_tpc: int,
        tr_topics: str
    ) -> pathlib.Path:
        """
        Create necessary directories for submodel.

        Parameters
        ----------
        father_model_path: pathlib.Path
            Path to the father model.
        htm_version: str
            Version of hierarchical topic modeling.
        exp_tpc: int
            Topic number from the father model to be expanded.
        tr_topics: str
            Training topics.

        Returns
        -------
        pathlib.Path
            Path to the created submodel directory.
        """
        (father_model_path / "submodels").mkdir(exist_ok=True)
        submodel_path = father_model_path / "submodels" / f"{htm_version}_from_tpc_{exp_tpc}_train_with_{tr_topics}"
        submodel_path.mkdir(exist_ok=True)
        sub_train_data_path = submodel_path / "train_data"
        sub_train_data_path.mkdir(exist_ok=True)
        return submodel_path

    def create_htm_ws_corpus(
        self,
        father_model_path: pathlib.Path,
        submodel_path: pathlib.Path,
        langs: List[str],
        exp_tpc: int
    ) -> None:
        """
        Create corpus for HTM-WS version.

        Parameters
        ----------
        father_model_path: pathlib.Path
            Path to the father model.
        submodel_path: pathlib.Path
            Path to the submodel directory.
        langs: List[str]
            List of languages to process.
        exp_tpc: int
            Topic number from the father model to be expanded.
        """
        # Load topic state data
        topic_state_model = father_model_path / "mallet_output" / "output-state.gz"
        with gzip.open(topic_state_model, 'rt') as fin:
            topic_state_df = pd.read_csv(
                fin, delim_whitespace=True,
                names=['docid', 'lang', 'wd_docid', 'wd_vocabid', 'wd', 'tpc'],
                header=None, skiprows=1
            )
        
        # Filter by the specified topic
        topic_state_df_tpc = topic_state_df[topic_state_df['tpc'] == exp_tpc]
        topic_to_corpus = topic_state_df_tpc.groupby(['docid', 'lang'])['wd'].apply(list).reset_index(name='new')
        
        # Get all unique document IDs
        all_docids = topic_to_corpus['docid'].unique()

        # Identify missing docid values for each language
        missing_entries = []
        for lang_id,_ in enumerate(langs):  # Assuming there are only two languages, 0 and 1
            missing_docids = [docid for docid in all_docids if not ((topic_to_corpus['docid'] == docid) & (topic_to_corpus['lang'] == lang_id)).any()]
            for docid in missing_docids:
                missing_entries.append({'docid': docid, 'lang': lang_id, 'new': []})

        # Append missing entries to the DataFrame
        if missing_entries:
            missing_df = pd.DataFrame(missing_entries)
            topic_to_corpus = pd.concat([topic_to_corpus, missing_df], ignore_index=True).sort_values(by=['docid', 'lang']).reset_index(drop=True)

        
        # Create corpus files for each language
        for lang_id, lang in enumerate(langs):
            sub_corpus_file = submodel_path / f"train_data/corpus_{lang}.txt"
            if sub_corpus_file.is_file():
                sub_corpus_file.unlink()
            topic_to_corpus_lang = topic_to_corpus[topic_to_corpus['lang'] == lang_id]
            topic_to_corpus_lang['new'] = topic_to_corpus_lang['new'].apply(lambda x: [i for i in x if pd.notna(i)])

            with open(sub_corpus_file, 'w', encoding='utf-8') as fout:
                for row in topic_to_corpus_lang.itertuples():
                    print(row)
                    fout.write(f"{row.docid} {lang} {' '.join(row.new)}\n")
                    
        return


    def create_htm_ds_corpus(
        self,
        father_model_path: pathlib.Path,
        submodel_path: pathlib.Path,
        langs: List[str],
        exp_tpc: int,
        thr: float
    ) -> None:
        """
        Create corpus for HTM-DS version.

        Parameters
        ----------
        father_model_path: pathlib.Path
            Path to the father model.
        submodel_path: pathlib.Path
            Path to the submodel directory.
        langs: List[str]
            List of languages to process.
        exp_tpc: int
            Topic number from the father model to be expanded.
        thr: float
            Threshold value for filtering documents.
        """
        # Get training corpus
        langs_corpus = {}
        dfs_save = {}
        for lang in langs:
            path_corpus = father_model_path / f"train_data/corpus_{lang}.txt"
            with path_corpus.open("r", encoding="utf-8") as f:
                lines = [line for line in f.readlines()]

            corpus_lang, ids = self.process_lines(lines, lang)
            df_lang = pd.DataFrame({"lemmas": corpus_lang, "doc_id": ids})
            df_lang["id_top"] = range(len(df_lang))
            
            langs_corpus[lang] = df_lang
            
            dfs_save[lang] = pd.DataFrame({
                'doc_id': pd.Series(dtype='str'),
                'id_top': pd.Series(dtype='int'),
                'lemmas': pd.Series(dtype='str')
            })
        
        for lang in langs:
            thetas = sparse.load_npz(father_model_path / f"mallet_output/thetas_{lang}.npz").toarray()
            
            # Keep documents from lang_id that have a proportion larger than thr
            doc_ids_to_keep = [idx for idx in range(thetas.shape[0]) if thetas[idx, exp_tpc] > thr]
            
            for lang in langs:
                df_lang = langs_corpus[lang].copy()
                dfs_save[lang] = pd.concat(
                    [dfs_save[lang],
                    df_lang[df_lang["id_top"].isin(doc_ids_to_keep)]],
                    ignore_index=True
                )
        for lang in langs:
            dfs_save[lang]['2mallet'] = dfs_save[lang]['id_top'].astype(str) + f" {lang} " + dfs_save[lang]['lemmas']
            sub_corpus_file = submodel_path / f"train_data/corpus_{lang}.txt"
            if sub_corpus_file.is_file():
                sub_corpus_file.unlink()
            dfs_save[lang][['2mallet']].to_csv(sub_corpus_file, index=False, header=False)

    def process_lines(
        self,
        lines: List[str],
        lang: str
    ) -> Tuple[List[str], List[str]]:
        """
        Process lines to extract corpus and IDs.

        Parameters
        ----------
        lines: List[str]
            List of lines from the corpus file.
        lang: str
            Language being processed.

        Returns
        -------
        Tuple[List[str], List[str]]
            Processed corpus and document IDs.
        """
        corpus_lang = [" ".join(line.rsplit(f' {lang} ')[1].strip().split()) for line in lines]
        ids = [line.split(" 0 ")[0] for line in lines]
        return corpus_lang, ids
