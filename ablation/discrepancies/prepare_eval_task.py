import pandas as pd

# Paths per model and topic
paths_ = []
for topic in [11,15]:
    for model in ["qwen:32b", "gpt-4o-2024-08-06", "llama3.3:70b"]:
        paths_.append(
            f"data/ablations/qa/v2/outs_good_model_tpc{topic}/answers/questions_topic_{topic}_{model}_100_seed_1234_results_model30tpc_thr__dynamic.parquet"
        )
path_save = "data/ablations/discrepancies/eval_tasks"

all_disc = []

# Process original datasets
for path in paths_:
    df = pd.read_parquet(path)

    # we want good questions here: they should have at least three words, end with "?", and do not start with "and" like they are a follow up    
    df = df[~df.question.str.contains("Here are the YES/NO questions generated based on the PASSAGE:")]
    
    no_yes_no = [
        "If a woman who is lactating chooses to consume alcohol moderately, what is the recommended maximum number of standard drinks per day?",
        "If a mother with pneumonic plague needs to avoid direct breastfeeding, what should she do to maintain milk supply and support the breastfeeding relationship?",
        "If a mother with pneumonic plague needs to avoid direct breastfeeding, what should she do to maintain milk supply and support the breastfeeding relationship?"
    ]
    
    df = df[~df.question.isin(no_yes_no)]
    
    df = df[df['question'].str.split().str.len() >= 3]
    df = df[df['question'].str.strip().str.endswith("?")]
    df = df[~df['question'].str.strip().str.lower().str.startswith("and")]
    # remove instances where a_s did not entail the original passage and hence a_t was not generated
    df = df[(df['answer_t'] != "N/A") & (df['discrepancy'] != "N/A")]

    if "gpt" in path:
        df["model"] = "gpt-4o"
    elif "llama" in path: 
        df["model"] = "llama3.3:70b"
    else:
        df["model"] = "qwen:32b"


    # Normalize NOT_ENOUGH_INFO
    # when a_t contains "cannot answer" should be NOT_ENOUGH_INFO
    df.loc[df.answer_t.str.contains("cannot answer", case=False, na=False), "discrepancy"] = "NOT_ENOUGH_INFO"

    # Clean up "NO_DISCREPANCY" variants
    df['discrepancy'] = (
        df['discrepancy']
        # Fix spacing: "NO_ DISCREPANCY" → "NO_DISCREPANCY"
        .str.replace(r'NO_\s+DISCREPANCY', 'NO_DISCREPANCY', regex=True)
        # Collapse any "TYPE: NO_*" style values to "NO_DISCREPANCY"
        .str.replace(r'^TYPE:\s*NO_.*$', 'NO_DISCREPANCY', regex=True)
        # Fix spacing: "CULTURAL_ DISCREPANCY" → "CULTURAL_DISCREPANCY"
        .str.replace(r'CULTURAL_\s+DISCREPANCY', 'CULTURAL_DISCREPANCY', regex=True)
    )
    

    # Add source column
    df["source"] = path.split("_tpc")[-1].split("/")[0]

    # Keep only relevant rows
    df_filtered = df[df["discrepancy"].isin(["CONTRADICTION", "CULTURAL_DISCREPANCY"])]

    # Sample 10 rows with NO_DISCREPANCY
    df_copy = df.copy()
    # drop duplicates by question
    df_copy = df_copy.drop_duplicates(subset="question", keep="first").reset_index(drop=True)
    no_discrepancy_sample = df_copy[df_copy["discrepancy"] == "NO_DISCREPANCY"].sample(n=10, random_state=42)
    print(len(no_discrepancy_sample))
    
    # Sample 10 rows with NOT_ENOUGH_INFO if available
    nei_sample = df_copy[df_copy["discrepancy"] == "NOT_ENOUGH_INFO"].sample(n=10, random_state=42)
    print(len(nei_sample))

    # Concatenate filtered results
    df_final = pd.concat([df_filtered, no_discrepancy_sample, nei_sample], ignore_index=True)

    # Keep only necessary columns
    df_final = df_final[["question", "answer_t", "answer_s", "discrepancy", "source", "model", "reason"]]

    # Append to all_disc
    all_disc.append(df_final)

# Load FEVER-DPLACE-Q dataset
df_fever = pd.read_csv("data/ablations/discrepancies/results/v2/FEVER-DPLACE-Q_v3_discp.csv")

# Sample 25 rows per unique label
#fever_samples = df_fever.groupby("label", group_keys=False).apply(lambda x: x.sample(n=min(30, len(x)), random_state=42))

# Rename columns
fever_samples = df_fever.copy()
fever_samples = fever_samples.rename(columns={"answer1": "answer_t", "answer2": "answer_s", "label": "discrepancy"})
fever_samples["model"] = "synthetic"
fever_samples["reason"] = "synthetic"
fever_samples["source"] = "FEVER-DPLACE-Q"
fever_samples["discrepancy"] = fever_samples["discrepancy"].str.replace("SUPPORTS", "NO_DISCREPANCY")
fever_samples["discrepancy"] = fever_samples["discrepancy"].str.replace("REFUTES", "CONTRADICTION")

# Keep only necessary columns
fever_samples = fever_samples[["question", "answer_t", "answer_s", "discrepancy", "source", "model", "reason"]]

# Append FEVER-DPLACE-Q to all_disc
all_disc.append(fever_samples)

# Create final dataframe
final_df = pd.concat(all_disc, ignore_index=True)
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
final_df["id_discr"] = range(len(final_df))

final_df = final_df[['id_discr','source','question', 'answer_t', 'answer_s', 'discrepancy', 'model', 'reason']]
final_df["label"] = [""] * len(final_df)
#final_df = final_df.drop_duplicates(subset="question", keep="first").reset_index(drop=True)

final_df.to_excel(f"{path_save}/discrepancies_v5.xlsx")
final_df.drop(columns=["source", "model", "discrepancy", "reason"]).to_excel(f"{path_save}/discrepancies_v5_users.xlsx", index=False)