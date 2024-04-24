import logging
import pathlib
from src.corpus_building.rosie_corpus import RosieCorpus
from src.topic_modeling.polylingual_tm import PolylingualTM


def main():
    
    print("ENTRA AQUI")
    path_corpus_es = "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/corpus_strict_v2.0_es_compiled_documents.jsonl"
    path_corpus_en = "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/corpus_strict_v3.0_en_compiled_documents.jsonl"
    path_save_tr = "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/df.parquet"
    
    # Generate training data
    print("-- -- Generating training data")
    rosie_corpus = RosieCorpus(path_corpus_en, path_corpus_es)
    path_save = rosie_corpus.generate_tm_tr_corpus(path_save_tr, level="document", sample=0.001)
    
    print("-- -- Training PolyLingual Topic Model")
    # Train PolyLingual Topic Model
    model = PolylingualTM(
        lang1="EN",
        lang2="ES",
        model_folder= pathlib.Path("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/models/rosie_test_all"),
        num_topics=20
    )
    model.train(path_save)
    
if __name__ == "__main__":
    main()
