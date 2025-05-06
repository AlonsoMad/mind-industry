import logging
import pathlib
from src.corpus_building.rosie_corpus import RosieCorpus
from src.topic_modeling.polylingual_tm import PolylingualTM
from src.topic_modeling.lda_tm import LDATM

def main():
    
    # path to json chunks
    path_df_chunks = "data/climate_fever/outs/1000/wikipedia_chunks_1234_1000_partial_1470.parquet"
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
    for k in [10,]:
        model = LDATM(
            langs=["EN"],
            model_folder= pathlib.Path(f"data/climate_fever/models/{str(sample)}_{k}"),
            num_topics=k
        )
        model.train(path_save)
    
if __name__ == "__main__":
    main()
