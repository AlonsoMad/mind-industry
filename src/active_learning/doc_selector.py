import gzip
from pathlib import Path
from typing import List
import pandas as pd


class DocSelector(object):
    def __init__(self) -> None:
        pass
        
        
    def _load_model_state(
        self,
        path_model: Path,
        polylingual: bool = False,
        langs = List[str](["EN","ES"])
    ):
        
        if polylingual:
            pass
        else:
            for lang in langs:
                with gzip.open((path_model / f"mallet_output/lang" / "topic-state.gz")) as fin:
                    topic_state_df = pd.read_csv(
                        fin, delim_whitespace=True,
                        names=['docid', 'NA3','wd_idx_doc', 'wd_vocab','word', 'tpc'],header=None, skiprows=3)

    
    
    