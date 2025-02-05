from collections import defaultdict
import pandas as pd
import pathlib
import scipy.sparse as sparse
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from itertools import combinations, product
from scipy import spatial
import sys

def kl2(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions
    (should only be used for the Jensen-Shannon divergence)

    *********USING BASE 2 FOR THE LOGARITHM*******
 
    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
        Discrete probability distributions.
    """
    p = np.squeeze(np.asarray(p))
    q = np.squeeze(np.asarray(q))
    #print np.all([p != 0,q!= 0],axis=0)
    #Notice standard practice would be that the p * log(p/q) = 0 for p = 0,
    #but p * log(p/q) = inf for q = 0. We could use smoothing, but since this
    #function will only be called to calculate the JS divergence, we can also
    #use p * log(p/q) = 0 for p = q = 0 (if q is 0, then p is also 0)
    return np.sum(np.where(np.all([p != 0,q!= 0],axis=0), p * np.log2(p / q), 0))

def js_similarity(theta1,theta2):
    (n1, col1) = theta1.shape
    (n2, col2) = theta2.shape
    if col1 != col2:
        sys.exit("Matrices have different dimensions")

    js_sim = np.empty( (n1,n2) )
    for idx,pq in zip(product(range(n1),range(n2)),product(theta1,theta2)):
        av = (pq[0] + pq[1])/2
        js_sim[idx[0],idx[1]] = 1 - 0.5 * (kl2(pq[0],av) + kl2(pq[1],av))
        
    return js_sim

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
            for lang in ["EN", "ES"]:
                path_thetas = f"{base_path}/{model_name}/mallet_output/thetas_{lang}.npz"
                thetas = sparse.load_npz(path_thetas)
                disp_perc = 100 * thetas.count_nonzero() / (thetas.shape[0] * thetas.shape[1])
                disp_perc_list[lang] = disp_perc
                print(f"Dispersion percentage for {lang}: {disp_perc}")
                
            # calculate difference between maximum and minimum pairwise Jensen-Shannon similarities of the betas
            path_betas = f"{base_path}/{model_name}/mallet_output/betas.npy"
            betas = np.load(path_betas)
            js_sim = js_similarity(betas,betas)
            diff_max_min = np.max(js_sim) - np.min(js_sim)
            print(f"Max-Min JS similarity: {diff_max_min}")
 
            num_topics = betas.shape[0]
            mean_js_sim = (np.sum(js_sim) - np.trace(js_sim)) / (num_topics * (num_topics - 1))  # Exclude self-similarity
            print(f"Mean Pairwise JS Similarity: {mean_js_sim}")
                            
            # we use lang = EN for coherence calculation
            lang = "EN"
            path_keys = f"{base_path}/{model_name}/mallet_output/keys_{lang}.txt"
            with open(path_keys, 'r') as file:
                lines = file.readlines()
            topic_keys = [" ".join(line.strip().split()[:10]) for line in lines]
            cohr_per_tpc = calculate_cohr_parallel(topic_keys)
            print(f"Coherence per topic: {cohr_per_tpc}")
            avg_cohr = np.mean([c for c in cohr_per_tpc if c is not None])  
            print(f"Average coherence: {avg_cohr}")
            
            jaccard_similarities = []
            topic_keys = [set(key.split()) for key in topic_keys]
            for t1, t2 in combinations(topic_keys, 2):  # Iterate over all topic pairs
                intersection = len(t1.intersection(t2))  # Words in common
                union = len(t1.union(t2))  # Total unique words
                jaccard_sim = intersection / union if union != 0 else 0
                jaccard_similarities.append(jaccard_sim)

            avg_overlap = np.mean(jaccard_similarities) if jaccard_similarities else 0
            print(f"Average Topic Overlap (Jaccard Similarity): {avg_overlap}")
                        
            data.append({
                "Model": model_name,
                "Num_Topics": num_topics,
                "Dispersion EN": disp_perc_list["EN"],
                "Dispersion ES": disp_perc_list["ES"],
                "Diff_Max_Min": diff_max_min,
                "Mean_JS": mean_js_sim,
                "Coherence": cohr_per_tpc,
                "Average_Coherence": avg_cohr,
                "Average_Topic_Overlap": avg_overlap
            })
            print(f"Average coherence: {avg_cohr}")
        except Exception as e:
            print(f"Error processing model {model_name}: {e}")
    
    # Convert the data to a DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

# Define paths and models
base_path = "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/models/28_jan"
path_models = pathlib.Path(base_path)
models = [dir.name for dir in path_models.iterdir() if dir.is_dir()]

# Process for English and Spanish
process_model_data(models, base_path, "cohrs_disp_betas_diff.csv")
