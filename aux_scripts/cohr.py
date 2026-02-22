from collections import defaultdict
import gzip
import pandas as pd
import pathlib
import scipy.sparse as sparse
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import numpy as np


def calculate_cohr(words):
    url = "http://palmetto.demos.dice-research.org/service/npmi?words="
    data = {"words": words}

    response = requests.post(url, data=data)

    if response.status_code == 200:
        return float(response.text)
    else:
        print("Error:", response.status_code, response.text)
        return None


def calculate_cohr_parallel(topic_keys):
    """Calculate coherence scores in parallel for a list of topic keys."""
    with ThreadPoolExecutor() as executor:
        # Map the `calculate_cohr` function to each topic key
        coherence_scores = list(executor.map(calculate_cohr, topic_keys))
    return coherence_scores


def process_model_data(models, base_path, output_file):
    data = []
    for model_name in models:
        print(f"Processing MODEL: {model_name}")

        try:
            num_topics = int(model_name.split("_")[-1])

            disp_perc_list = defaultdict(float)
            for lang in ["EN", "DE"]:
                path_thetas = f"{base_path}/{model_name}/mallet_output/thetas_{lang}.npz"
                thetas = sparse.load_npz(path_thetas)
                disp_perc = 100 * thetas.count_nonzero() / \
                    (thetas.shape[0] * thetas.shape[1])
                disp_perc_list[lang] = disp_perc
                print(f"Dispersion percentage for {lang}: {disp_perc}")

            topic_state_model = f"{base_path}/{model_name}/mallet_output/output-state.gz"
            with gzip.open(topic_state_model) as fin:
                topic_state_df = pd.read_csv(
                    fin, delim_whitespace=True,
                    names=['docid', 'lang', 'wd_docid',
                           'wd_vocabid', 'wd', 'tpc'],
                    header=None, skiprows=1)

            # we use lang = EN for coherence calculation
            lang = "EN"
            path_keys = f"{base_path}/{model_name}/mallet_output/keys_{lang}.txt"
            with open(path_keys, 'r') as file:
                lines = file.readlines()
            topic_keys = [" ".join(line.strip().split()[:10])
                          for line in lines]
            cohr_per_tpc = calculate_cohr_parallel(topic_keys)
            print(f"Coherence per topic: {cohr_per_tpc}")
            avg_cohr = np.mean([c for c in cohr_per_tpc if c is not None])
            print(f"Average coherence: {avg_cohr}")

            data.append({
                "Model": model_name,
                "Num_Topics": num_topics,
                "Dispersion EN": disp_perc_list["EN"],
                "Dispersion ES": disp_perc_list["ES"],
                "Coherence": cohr_per_tpc,
                "Average_Coherence": avg_cohr,
            })
            print(f"Average coherence: {avg_cohr}")
        except Exception as e:
            print(f"Error processing model {model_name}: {e}")

    # Convert the data to a DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


# Define paths and models
base_path = "../data/models/wiki/ende"
path_models = pathlib.Path(base_path)
models = [dir.name for dir in path_models.iterdir() if dir.is_dir()]

# Process for English and Spanish
process_model_data(models, base_path, "stats.csv")
