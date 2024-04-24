import fasttext
from huggingface_hub import hf_hub_download
import pandas as pd

fasttext.FastText.eprint = lambda x: None

def detect_lang(text: str) -> str:
    model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
    model = fasttext.load_model(model_path)
    return model.predict(text)[0][0]


def main():
    df_es = pd.read_json("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/corpus_strict_v2.0_es_compiled_documents.jsonl", lines=True)
    df_en = pd.read_json("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/corpus_strict_v3.0_en_compiled_documents.jsonl", lines=True)
    
    df_es["lang"] = df_es["contents"].apply(detect_lang)
    df_en["lang"] = df_en["contents"].apply(detect_lang)
    
    df_es.to_json("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/corpus_strict_v2.0_es_compiled_documents_lang.jsonl", lines=True)
    df_en.to_json("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/corpus_strict_v3.0_en_compiled_documents_lang.jsonl", lines=True)
    
if __name__ == "__main__":
    main()