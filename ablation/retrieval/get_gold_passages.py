import pandas as pd
from tqdm import tqdm # type: ignore
from mind.prompter.prompter import Prompter # type: ignore
import pathlib
template_path = "src/mind/pipeline/prompts/test_relevance.txt"
with open(template_path, 'r') as file:
    template = file.read()
    
models = ["gpt-4o-2024-08-06", "llama3.3:70b", "qwen:32b"]

topics = [11, 15]

for topic in tqdm(topics):
    for model in tqdm(models):
        path_annotations = f"/export/usuarios_ml4ds/lbartolome/Repos/umd/mind/data/ablations/retrieval/v2/BAAI/bge-m3/topic_{topic}/questions_topic_{topic}_{model}_100_seed_1234_results_model30tpc_thr__combined_to_retrieve_relevant.parquet"

        annotations = pd.read_parquet(path_annotations)
        for llm_model in tqdm(["qwen:32b", "llama3.3:70b", "llama3.3:70b-instruct-q5_K_M", "llama3.1:8b-instruct-q8_0"]):
            prompter = Prompter(
                model_type=llm_model,
            )
            print(f"Processing for LLM {llm_model}")
            annotations[f"relevance_{llm_model}"] = len(annotations) * [None]
            for id_row, row in annotations.iterrows():
                question = template.format(passage=row.all_results_content, question=row.question)
                response, _ = prompter.prompt(
                    #system_prompt_template_path=system_template_path,
                    question=question
                )
                #print(question)
                #print(row.all_results_content)
                #print(f"Response: {response}")
                relevance = 1 if "yes" in response.lower() else 0
                #if relevance == 1:
                #    import pdb; pdb.set_trace()
                annotations.loc[id_row, f"relevance_{llm_model}"] = relevance
                if id_row % 10 == 0: 
                    print(f"Processed {len(annotations) - id_row} / {len(annotations)} using LLM {llm_model}", flush=True)

        try:
            path_save = pathlib.Path(path_annotations).parent / f"relevant_{model}.parquet"
            annotations.to_parquet(path_save, index=False)
        except Exception as e:
            print(f"Error saving annotations for {path_annotations}. Error: {e}")
            import pdb; pdb.set_trace()