import logging
import pathlib
from src.corpus_building.rosie_corpus import RosieCorpus
from src.topic_modeling.polylingual_tm import PolylingualTM
from src.topic_modeling.lda_tm import LDATM

def main():
    
    print("ENTRA AQUI")
    path_corpus_es = "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/corpus_strict_v2.0_es_compiled_documents_lang.parquet"
    path_corpus_en = "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/corpus_strict_v3.0_en_compiled_documents_lang.parquet"
    path_save_tr = "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/large/df.parquet"
    
    # Generate training data
    print("-- -- Generating training data")
    sample=1
    rosie_corpus = RosieCorpus(path_corpus_en, path_corpus_es)
    path_save = rosie_corpus.generate_tm_tr_corpus(path_save_tr, level="document", sample=sample)
    
    print("-- -- Training PolyLingual Topic Model")
    # Train PolyLingual Topic Model
    model_type = "lda"
    for k in [100,200,300,400,500]:
        model = LDATM(#PolylingualTM
            lang1="EN",
            lang2="ES",
            model_folder= pathlib.Path(f"/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/models/LDA/rosie_lg_{model_type}_{str(sample)}_{k}"),
            num_topics=k
        )
        model.train(path_save)
    
if __name__ == "__main__":
    main()
