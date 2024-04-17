import logging
import pathlib
from src.corpus_building.rosie_corpus import RosieCorpus
from src.topic_modeling.polylingual_tm import PolylingualTM


def main():
    
    logger = logging.Logger(__name__)
    path_corpus_es = "/Users/lbartolome/Documents/GitHub/LinQAForge/data/source/corpus_rosie/corpus_strict_v2.0_es_compiled_documents.jsonl"
    path_corpus_en = "/Users/lbartolome/Documents/GitHub/LinQAForge/data/source/corpus_rosie/corpus_strict_v3.0_en_compiled_documents.jsonl"
    path_save_tr = "/Users/lbartolome/Documents/GitHub/LinQAForge/data/source/corpus_rosie/df.parquet"
    
    # Generate training data
    rosie_corpus = RosieCorpus(path_corpus_en, path_corpus_es,logger=logger)
    rosie_corpus.generate_tm_tr_corpus(path_save_tr, level="document", sample=0.000001)
    
    # Train PolyLingual Topic Model
    model = PolylingualTM(
        lang1="EN",
        lang2="ES",
        model_folder= pathlib.Path("/Users/lbartolome/Documents/GitHub/LinQAForge/data/models/rosie_test_1"),
        num_topics=8
    )
    model.train(pathlib.Path(path_save_tr))
    
if __name__ == "__main__":
    main()