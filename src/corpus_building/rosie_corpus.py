import logging
import pathlib
from subprocess import check_output
import pandas as pd
import os

class RosieCorpus(object):
    def __init__(
        self,
        path_data_en: str,
        path_data_es: str,
        path_preproc: str = "src/corpus_building/preprocessing/NLPipe/nlpipe.py",
        logger: logging.Logger = None
    ) -> None:
        pass

        if logger is not None:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)
            self._logger.setLevel(logging.INFO)

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
        spacy_models = ["en_core_web_sm", "es_core_news_sm"]):
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
        
        df_en_sample = self.df_en.sample(frac=sample)
        df_es_sample = self.df_es.sample(frac=sample)
        
        # Create intermediate files for preprocessing if they don't exist
        path_save_preproc = pathlib.Path(path_save).parent / "preproc"
        path_save_preproc.mkdir(exist_ok=True)
        
        path_save_en = path_save_preproc / f"en_{sample}.parquet"
        path_save_es = path_save_preproc / f"es_{sample}.parquet"
        
        tuples_get = [
            (df_en_sample, "EN", path_save_en),
            (df_es_sample, "ES", path_save_es)]
        
        def get_doc_id(lang, index, row , level):
            if level == "passage":
                return f"{lang}_{index}_{row.passage_id}", "passage"
            elif level == "document":
                return f"{lang}_{index}", "contents"
            else:
                raise ValueError(f"Level {level} not recognized.")
        
        if not path_save_en.exists() or not path_save_es.exists():
            self._logger.info(
                f"-- -- Saving intermediate files for preprocessing at {path_save_preproc}...")
            
            dicts_to_df = []
            for id, (df, lang, path_save) in enumerate(tuples_get):
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
                    })
                dicts_to_df.append(new_df)
                
                # Save intermediate files
                new_df.to_parquet(path_save)

                # Carry out preprocessing
                self.preproc_tr_corpus(source_path=path_save, lang=lang, spacy_model=spacy_models[id])
                
                df_preproc = pd.read_parquet(path_save)
            
                import pdb; pdb.set_trace()
                    
            final_df = pd.concat(dicts_to_df)
            self._logger.info(
                f"-- -- Training corpus generated. Nr elements is {len(dicts_to_df[0])} in {tuples_get[0][1]} and {len(dicts_to_df[1])} in {tuples_get[1][1]}. Saving at {path_save}...")
            final_df.to_parquet(path_save)    
            
        return
    
    def preproc_tr_corpus(
        self,
        source_path,
        lang,
        spacy_model):
        
        #path_python = (self._path_preproc.parent.parent / ".venv_nlpipe/bin/python3")
        #path_python = pathlib.Path("/Users/lbartolome/Documents/GitHub/NLPipe/.venv/bin/python3")
        path_python_str = "python3 " #f"{path_python.as_posix()} "
        path_nlpipe = os.getcwd() / self._path_preproc
        
        source_save_path = source_path.resolve().as_posix()
        cmd = path_python_str + path_nlpipe.as_posix() + \
            ' --source_path %s ' \
            '--source_type %s '\
            '--source %s '\
            '--destination_path %s '\
            '--lang %s ' \
            '--spacy_model %s '
        cmd = cmd % \
            (source_save_path, "parquet", "rosie",
             source_save_path, lang.lower(), spacy_model)

        try:
            self._logger.info(
                f'-- -- Preprocessing corpus {source_save_path}. Command is {cmd}')
            check_output(args=cmd, shell=True)
        except:
            self._logger.error('-- -- Preprocessing corpus. Revise command')
            return
        
        # Luego hay que volver a juntarlo para tener los "empties"
        
        return
