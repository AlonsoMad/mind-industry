import fasttext
from huggingface_hub import hf_hub_download
import pandas as pd

fasttext.FastText.eprint = lambda x: None
model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
model = fasttext.load_model(model_path)

def langDetectHF (row):
	try:
		lang = model.predict (row.passage.replace("\n", ""))[0][0].replace('__label__','')
	except Exception as E:
		#import ipdb ; ipdb.set_trace()
		lang =str(E)
	return lang

def main():
    print("Reading files...")
    
    df_es = pd.read_json("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/corpus_strict_v2.0_es_compiled_passages.jsonl", lines=True)
    print("Spanish read.")
    
    df_en = pd.read_json("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/corpus_strict_v3.1_en_compiled_passages_valid_links.jsonl", lines=True)
    print("English read.")
    
    print(df_es.head())
    print("Detecting language for Spanish...")
    df_es["lang"] = df_es.apply(langDetectHF,axis=1)
    print("Spanish detected.")
    print(df_es.head())
    
    print(df_en.head())
    print("Detecting language for English...")
    df_en["lang"] = df_en.apply(langDetectHF,axis=1)
    print("English detected.")
    print(df_en.head())
    
    df_es.to_parquet("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/corpus_strict_v2.0_es_compiled_passages_lang.parquet")
    df_en.to_parquet("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/corpus_strict_v3.0_en_compiled_passages_lang.parquet")
        
if __name__ == "__main__":
    main()