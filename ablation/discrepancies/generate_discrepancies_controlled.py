import pandas as pd
from mind.pipeline.pipeline import MIND
from tqdm import tqdm

PATH_SOURCE = "data/datasets/dplaceq/FEVER-DPLACE-Q_v3.xlsx"
PATH_SAVE = "data/ablations/discrepancies/results/v2/FEVER-DPLACE-Q_v3_discp.csv"

df = pd.read_excel(PATH_SOURCE)

for llm_model in ["qwen:32b", "llama3.3:70b", "gpt-4o-2024-08-06"]:

    mind = MIND(llm_model=llm_model, do_check_entailement=False)

    print("#" * 50)
    print(f"Processing for LLM {llm_model}")
    print("#" * 50)

    for col_add in [f"discp_{llm_model}", f"reason_{llm_model}"]:
        df[col_add] = len(df) * [None]

    for id_row, row in tqdm(df.iterrows(), total=df.shape[0]):

        question = row.question
        a_s = row.answer1
        a_t = row.answer2

        discrepancy_label, reason = mind._check_contradiction(
            question, a_s, a_t)

        discrepancy_label = discrepancy_label.replace(
            "NO_ DISCREPANCY", "NO_DISCREPANCY")
        discrepancy_label = discrepancy_label.replace(
            "CULTURAL_ DISCREPANCY", "CULTURAL_DISCREPANCY")

        print("Discrepancy:", discrepancy_label)
        df.loc[id_row, f"discp_{llm_model}"] = discrepancy_label
        df.loc[id_row, f"reason_{llm_model}"] = reason

df.to_csv(PATH_SAVE, index=False)
