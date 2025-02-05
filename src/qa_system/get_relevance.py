import pandas as pd
from prompter import Prompter

template_path = "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/src/qa_system/templates/test_relevance.txt"
with open(template_path, 'r') as file:
    template = file.read()


path_annotations = "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/src/qa_system/df_q_29jan_topic_15v2_es_model30tpc_combined_to_retrieve_relevant.xlsx"
annotations = pd.read_excel(path_annotations)


for llm_model in ["llama3:70b-instruct", "qwen:32b", "llama3.3:70b", "llama3.1:8b-instruct-q8_0"]:
    
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
        relevance = 1 if "yes" in response.lower() else 0
        annotations.loc[id_row, f"relevance_{llm_model}"] = relevance
    
        if id_row % 100 == 0: 
            print(f"Processed {len(annotations) - id_row} / {len(annotations)}")
            
annotations.to_excel(path_annotations.replace(".xlsx", "_relevance.xlsx"), index=False)