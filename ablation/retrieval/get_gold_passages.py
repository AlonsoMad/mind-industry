import pandas as pd
from tqdm import tqdm # type: ignore
from mind.prompter.prompter import Prompter # type: ignore
import pathlib
template_path = "src/mind/pipeline/prompts/test_relevance.txt"
with open(template_path, 'r') as file:
    template = file.read()
    
models = ["llama3.3:70b", "qwen:32b"] 
#topics = [11, 15]
topics = [11]

for topic in tqdm(topics):
    for model in tqdm(models):
        path_annotations = f"/export/usuarios_ml4ds/lbartolome/Repos/umd/mind/data/ablations/retrieval/v2/BAAI/bge-m3/topic_{topic}/questions_topic_{topic}_{model}_100_seed_1234_results_model30tpc_thr__combined_to_retrieve_relevant.parquet"

        # Checkpoint path
        path_save = pathlib.Path(path_annotations).parent / f"relevant_{model}.parquet"
        llm_models = ["qwen:32b", "llama3.3:70b", "llama3.3:70b-instruct-q5_K_M", "llama3.1:8b-instruct-q8_0"]

        # Try to load checkpoint if exists
        if path_save.exists():
            annotations = pd.read_parquet(path_save)
            print(f"Loaded checkpoint from {path_save}")
        else:
            annotations = pd.read_parquet(path_annotations)

        # Check if all relevance columns exist and are fully filled
        all_cols_exist = all(f"relevance_{llm}" in annotations.columns for llm in llm_models)
        all_cols_filled = all(annotations[f"relevance_{llm}"].notna().all() for llm in llm_models if f"relevance_{llm}" in annotations.columns)
        if all_cols_exist and all_cols_filled:
            print(f"All relevance columns present and filled for {model}, skipping computation.")
            continue

        for llm_model in tqdm(llm_models):
            if f"relevance_{llm_model}" in annotations.columns and annotations[f"relevance_{llm_model}"].notna().all():
                print(f"Skipping {llm_model}, already computed.")
                continue
            prompter = Prompter(
                model_type=llm_model,
            )
            print(f"Processing for LLM {llm_model}")
            if f"relevance_{llm_model}" not in annotations.columns:
                annotations[f"relevance_{llm_model}"] = len(annotations) * [None]
            for id_row, row in annotations.iterrows():
                if pd.notna(annotations.loc[id_row, f"relevance_{llm_model}"]):
                    continue
                question = template.format(passage=row.all_results_content, question=row.question)
                response, _ = prompter.prompt(
                    #system_prompt_template_path=system_template_path,
                    question=question
                )
                relevance = 1 if "yes" in response.lower() else 0
                annotations.loc[id_row, f"relevance_{llm_model}"] = relevance
                if id_row % 10 == 0:
                    print(f"Processed {id_row+1} / {len(annotations)} using LLM {llm_model}", flush=True)
            # Save checkpoint after each LLM
            try:
                annotations.to_parquet(path_save, index=False)
                print(f"Checkpoint saved after {llm_model} at {path_save}")
            except Exception as e:
                print(f"Error saving checkpoint for {path_save}. Error: {e}")
                import pdb; pdb.set_trace()