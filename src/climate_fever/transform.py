import re
import pandas as pd
from tqdm import tqdm

from ..prompter.prompter import Prompter

df = pd.read_json("data/climate_fever/climate-fever-dataset-r1.jsonl", lines=True)

df_claim_processed = pd.read_json("data/climate_fever/outs/1000/claim_to_page_links_1234_1000_partial_1470.json", lines=True)
claims_processed = df_claim_processed.claim_id.unique()

# # Filter the dataframe to only include rows where the claim is in claims_processed
df = df[df['claim_id'].isin(claims_processed)]

INSTRUCTIONS_PATH = "src/climate_fever/transform_climate_fever.txt" #src/climate_fever/

prompter = Prompter(
    model_type="qwen2.5:7b-instruct", ollama_host="http://kumo01.tsc.uc3m.es:11434")


new_dataset = []
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
        
        response, _ = prompter.prompt(question=template_formatted)
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
        
        new_dataset.append({
            "claim_id": row.claim_id,
            "claim": row.claim,
            "evidence": evidence["evidence"],
            "label": evidence["evidence_label"],
            "question": question,
            "answer": answer
        })
        
# Save the new dataset to a JSON file
new_df = pd.DataFrame(new_dataset)
new_df.to_json("data/climate_fever/transformed/climate_fever_transformed_1234_1000_partial_1470.json", orient="records", lines=True)