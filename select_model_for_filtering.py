import logging
import pathlib
from src.corpus_building.rosie_corpus import RosieCorpus
from src.topic_modeling.polylingual_tm import PolylingualTM
from src.topic_modeling.lda_tm import LDATM
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

def get_statistics(diffs, mallet_out_folder, n_tpcs, langs=['EN', 'ES']):
    # the difference between the maximum and minimum similarity was used as a filter to exclude models that included either too similar components (producing maximum similarity close to 1) or clear outliers (minimum similarity close to 0).
    
    # for each language, load the beta matrix and calculate the cosine similarity
    diffs_this_model = []
    for lang in langs:
        betas = np.load(mallet_out_folder / f"{lang}/betas.npy")
        cosine = cosine_similarity(betas)
        np.fill_diagonal(cosine, np.nan)
        cosine = cosine[~np.isnan(cosine)].reshape(cosine.shape[0], cosine.shape[1] - 1)
        max_cosine = cosine.max()
        min_cosine = cosine.min()

        print(lang, n_tpcs, 'cosine_mean', cosine.mean())
        print(lang, n_tpcs, 'cosine similarity_max', max_cosine)
        print(lang, n_tpcs, 'cosine similarity_min', min_cosine)
        print(lang, n_tpcs, 'cosine difference', max_cosine-min_cosine)
        diffs_this_model.append(max_cosine - min_cosine)
        
    diffs.append(
        [mallet_out_folder.parent,
        n_tpcs,
        diffs_this_model[0],
        diffs_this_model[1]]
    )
    
    return
        
def main():
    
    print("ENTRA AQUI")
    path_corpus_es = "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/corpus_pass_es_tr.parquet"
    path_corpus_en = "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/corpus_pass_en_tr.parquet"
    path_save_tr = "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/passages/translated/df.parquet"
    
    # Generate training data
    print("-- -- Generating training data")
    sample=1
    rosie_corpus = RosieCorpus(path_corpus_en, path_corpus_es)
    path_save = rosie_corpus.generate_tm_tr_corpus(path_save_tr, level="passage", sample=sample)
    
    # for saving the models sims
    diffs = []
    
    print("-- -- Training LDATM Topic Model")
    # Train LDATM 
    for k in [5,10,20,30,40,50,60,70,80,90,100,200,300,400,500]: 
        model = LDATM(
            lang1="EN",
            lang2="ES",
            model_folder= pathlib.Path(f"/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/models/filtering/rosie_{str(sample)}_{k}"),
            num_topics=k
        )
        print(f"-- -- Training with {k} topics...")
        mallet_out_folder = model.train(path_save)
        
        get_statistics(diffs, mallet_out_folder, k)
    
    # Convert stats to a pandas dataframe
    df = pd.DataFrame(diffs, columns=['model_folder', 'n_tpcs', 'diff_en', 'diff_es'])
    
    # We keep the k such that the difference of maximum and minimum cosine similarities for both languages is as closest to 0.5 as possible
    df['diff'] = (df['diff_en'] + df['diff_es']) / 2
    df = df.sort_values('diff')
    print(df)
    print(f"-- -- Best model: {df.iloc[0]['model_folder']}, k={df.iloc[0]['n_tpcs']}")

    
if __name__ == "__main__":
    main()
