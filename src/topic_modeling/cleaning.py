import pandas as pd
import pathlib
import scipy.sparse as sparse
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def calculate_cohr(words):
    url = "http://palmetto.demos.dice-research.org/service/npmi?words="
    data = {"words": words}
    
    response = requests.post(url, data=data)
    
    if response.status_code == 200:
        return float(response.text)
    else:
        print("Error:", response.status_code, response.text)
        return None

def process_model_data(models, lang, base_path, output_file):
    data = []
    for model_name in models:
        print(f"Processing MODEL: {model_name} for language: {lang}")
        num_topics = int(model_name.split("_")[-1])
        
        path_thetas = f"{base_path}/{model_name}/mallet_output/{lang}/thetas.npz"
        thetas = sparse.load_npz(path_thetas)
        disp_perc = 100 * thetas.count_nonzero() / (thetas.shape[0] * thetas.shape[1])
        
        path_keys = f"{base_path}/{model_name}/mallet_output/{lang}/topickeys.txt"
        with open(path_keys, 'r') as file:
            lines = file.readlines()
        topic_keys = [" ".join(line.split('\t', 2)[2].strip().split()[:10]) for line in lines]
        avg_cohr = np.mean([calculate_cohr(el) for el in topic_keys])
        
        data.append({
            "Model": model_name,
            "Num_Topics": num_topics,
            "Dispersion": disp_perc,
            "Average_Coherence": avg_cohr
        })
        print(f"Average coherence: {avg_cohr}")
    
    # Convert the data to a DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

# Define paths and models
base_path = "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/models/LDA/passage"
path_models = pathlib.Path(base_path)
models = [dir.name for dir in path_models.iterdir() if dir.is_dir()]

# Process for English and Spanish
process_model_data(models, "EN", base_path, "cohrs_disp_en.csv")
process_model_data(models, "ES", base_path, "cohrs_disp_es.csv")
