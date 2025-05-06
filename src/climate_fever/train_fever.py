import logging
import pathlib

import numpy as np
from src.corpus_building.rosie_corpus import RosieCorpus
from src.topic_modeling.polylingual_tm import PolylingualTM
from src.topic_modeling.lda_tm import LDATM

def main():
    
    # path to json chunks
    path_df_chunks = "data/climate_fever/corpus_train_chunked.parquet"
    path_save = "data/climate_fever/preprocessed"
    
    # Generate training data
    print("-- -- Generating training data")
    sample=1
    rosie_corpus = RosieCorpus(
        paths_data={"EN": path_df_chunks}, 
        multilingual=False
    )
    path_save = rosie_corpus.generate_tm_tr_corpus(path_save, level="passage", sample=sample, column_preproc="chunk_text", column_id="chunk_id")
    
    
    print("-- -- Training PolyLingual Topic Model")
    for k in np.arange(5, 51, 5):
        model = LDATM(
            langs=["EN"],
            model_folder= pathlib.Path(f"data/climate_fever/models/{str(sample)}_{k}"),
            num_topics=k
        )
        model.train(path_save)
    
if __name__ == "__main__":
    main()
