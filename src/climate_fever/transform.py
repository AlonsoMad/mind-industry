import pandas as pd # type: ignore
from tqdm import tqdm # type: ignore

from ..prompter.prompter import Prompter

df = pd.read_json("data/climate_fever/climate-fever-dataset-r1.jsonl", lines=True)

INSTRUCTIONS_PATH = "src/climate_fever/transform_climate_fever.txt"

prompter = Prompter(
    model_type="qwen2.5:7b-instruct")


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
new_df.to_json("data/climate_fever/questions_transformed.json", orient="records", lines=True)