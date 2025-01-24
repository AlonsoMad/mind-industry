import logging
import pathlib
from src.corpus_building.rosie_corpus import RosieCorpus
from src.topic_modeling.polylingual_tm import PolylingualTM
from src.topic_modeling.lda_tm import LDATM

def main():
    
    print("ENTRA AQUI")
    path_corpus_es = "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/corpus_pass_es_tr.parquet"
    path_corpus_en = "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/corpus_pass_en_tr.parquet"
    #path_save_tr = "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/passages/translated/df.parquet"
    path_save_tr = "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/passages/29_dec/all/df.parquet"
    
    # Generate training data
    print("-- -- Generating training data")
    sample=1
    rosie_corpus = RosieCorpus(path_corpus_en, path_corpus_es)
    path_save = rosie_corpus.generate_tm_tr_corpus(path_save_tr, level="passage", sample=sample)
    
    print("-- -- Training PolyLingual Topic Model")
    # Train PolyLingual Topic Model
    for k in [15,20,30,50,100,200,300,400,500]: #,100,200,300,400,500
        #model = PolylingualTM(
        model = LDATM(
            lang1="EN",
            lang2="ES",
            #model_folder= pathlib.Path(f"/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/models/29_dec/all/poly_rosie_{str(sample)}_{k}"),
            model_folder = pathlib.Path(f"/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/models/29_dec/LDA/lda_rosie_{str(sample)}_{k}"),
            num_topics=k
        )
        model.train(path_save)
    
if __name__ == "__main__":
    main()
