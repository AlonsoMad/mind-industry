import numpy as np
import pandas as pd # type: ignore
from tqdm import tqdm # type: ignore
import tiktoken 
from ..prompter.prompter import Prompter

df = pd.read_json("data/climate_fever/climate-fever-dataset-r1.jsonl", lines=True)

INSTRUCTIONS_PATH = "src/climate_fever/transform_climate_fever.txt"

prompter = Prompter(
    model_type='gpt-4o-mini-2024-07-18',
    temperature=0)
#"qwen2.5:7b-instruct"

tokenizer = tiktoken.encoding_for_model("gpt-4o")
total_input_tokens = 0
total_output_tokens = 0

new_dataset = []
# set number of times to infiniten
nr_items = np.inf
#len(df)#1000
items_count = 0
for id_row, row in tqdm(df.iterrows(), total=df.shape[0]):
    for evidence in row.evidences:
        
        # -------------------------------------------------------------------#
        # GENERATE TRIPLETS
        # -------------------------------------------------------------------#
        with open(INSTRUCTIONS_PATH, 'r') as file: template = file.read()
        template_formatted = template.format(
            claim=row.claim,
            evidence=evidence["evidence"],
            label=evidence["evidence_label"],
        )
        
        #print(template_formatted)
        
        input_tokens = len(tokenizer.encode(template_formatted))
        total_input_tokens += input_tokens
        
        response, _ = prompter.prompt(question=template_formatted)
        
        output_tokens = len(tokenizer.encode(response))
        total_output_tokens += output_tokens
        
        print(f"Input tokens: {input_tokens}, Output tokens: {output_tokens}")
        
        lines = response.splitlines()
        question, answer = None, None
        
        for line in lines:
            if line.startswith("QUESTION:"):
                question = line.split("QUESTION:")[1].strip()
            elif line.startswith("ANSWER:"):
                answer = line.split("ANSWER:")[1].strip()
        if question == None or answer == None:
            import pdb; pdb.set_trace()
        print("Question:", question)
        print("Answer:", answer)
        if "cannot answer the question" in answer.lower():
            answer = "I cannot answer given the context."
                
        new_dataset.append({
            "claim_id": row.claim_id,
            "claim": row.claim,
            "evidence": evidence["evidence"],
            "label": evidence["evidence_label"],
            "question": question,
            "answer": answer
        })
        items_count += 1
        if items_count >= nr_items:
            break
    if items_count >= nr_items:
        break
    
    # save every 10 items
    if items_count % 10 == 0:
        new_df = pd.DataFrame(new_dataset)
        new_df['claim_group_index'] = new_df.groupby('claim_id').cumcount()
        new_df['claim_evidence_id'] = new_df['claim_id'].astype(str) + '-' + new_df['claim_group_index'].astype(str)
        new_df.to_json(f"data/climate_fever/questions_transformed_{nr_items}.json", orient="records", lines=True)

avg_input_tokens = total_input_tokens / items_count
avg_output_tokens = total_output_tokens / items_count
print(f"Average input tokens: {avg_input_tokens}")
print(f"Average output tokens: {avg_output_tokens}")
import pdb; pdb.set_trace()
# Save the new dataset to a JSON file
new_df = pd.DataFrame(new_dataset)
new_df['claim_group_index'] = new_df.groupby('claim_id').cumcount()
new_df['claim_evidence_id'] = new_df['claim_id'].astype(str) + '-' + new_df['claim_group_index'].astype(str)
new_df.to_json(f"data/climate_fever/questions_transformed_{nr_items}.json", orient="records", lines=True)