import pandas as pd

# Paths per model
topic = 15
models_eval = ["qwen:32b", "gpt-4o-2024-08-06", "llama3.3:70b"]
path_results_mind = {model: f"data/ablations/qa/v2/outs_good_model_tpc{topic}/answers/questions_topic_{topic}_{model}_100_seed_1234_results_model30tpc_thr__dynamic.parquet" for model in models_eval}
path_save = "data/ablations/qa/eval_tasks"
eval_per_model = 50
prepare="answer"# "answer" or "both"

if prepare == "question" or prepare == "both":
    ############################################################################
    print("Preparing QUESTION EVAL task...")
    cols_keep = ['question', 'passage_s']
    dimensions_eval = ["Verifiability", "Passage Independence", "Clarity", "Terminology", "Self-Containment", "Naturalness"]

    all_results = []
    for model_name, path in path_results_mind.items():
        df = pd.read_parquet(path)
        
        # Drop duplicate questions if the column exists
        f = df.drop_duplicates(subset=['question'], keep='first')
        
        df = df[cols_keep]
        
        # Sample up to eval_per_model rows; use replacement if there aren't enough
        n_rows = len(df)
        if n_rows == 0:
            continue
        replace_flag = n_rows < eval_per_model
        df = df.sample(n=eval_per_model, replace=replace_flag, random_state=42)
        
        # Add model name
        df['model'] = model_name
        
        all_results.append(df)

    df_all = pd.concat(all_results, ignore_index=True)
    # create one column per dimension, initialized to empty string
    df_all["row_id"] = range(len(df_all))
    for col in dimensions_eval:
        df_all[col] = len(df_all) * [""]

    # order the columns so it appears:
    # id_row
    # passage_s
    # question
    # dimensions (Verifiability, Passage Independence, Clarity, Terminology, Self-Containment, Naturalness)
    # model
    ordered_cols = (
        ["row_id", "passage_s", "question"]
        + dimensions_eval
        + ["model"]
    )
    df_all = df_all[ordered_cols]
    df_all.to_excel(path_save + "/questions_eval.xlsx", index=False)
    print(f"Saved {len(df_all)} rows to {path_save}/questions_eval.xlsx")

if prepare == "answer" or prepare == "both":
###############################################################################
    print("Preparing ANSWER EVAL task...")
    cols_keep = ['question', 'passage_s', 'passage_t', 'answer_s', 'answer_t']
    # Evaluation dimensions
    dims_eval = [
        "Faithfulness",
        "Passage Dependence",
        "Passage Reference Avoidance",
        "Structured Response",
        "Language Consistency",
    ]

    all_results = []
    for model_name, path in path_results_mind.items():
        df = pd.read_parquet(path)
        
        # we want good questions here: they should have at least three words, end with "?", and do not start with "and" like they are a follow up
        df = df[df['question'].str.split().str.len() >= 3]
        df = df[df['question'].str.strip().str.endswith("?")]
        df = df[~df['question'].str.strip().str.lower().str.startswith("and")]
                 
        # remove instances where a_s did not entail the original passage and hence a_t was not generated
        df = df[df.answer_t != "N/A"]
        
        # Drop duplicate questions if the column exists
        df = df.drop_duplicates(subset=['question'], keep='first')

        df = df[cols_keep]
        
        # we do not want all a_t to be "I cannot answer the question given the context". We try to put a limit of 30%. in total we need to have eval_per_model rows
        limit = 0.3
        max_cannot_answer = int(eval_per_model * limit)
        n_cannot_answer = (df['answer_t'] == "I cannot answer the question given the context.").sum()
        if n_cannot_answer > max_cannot_answer:
            df_cannot_answer = df[df['answer_t'].str.contains("I cannot answer")]
            df_cannot_answer = df_cannot_answer.sample(n=max_cannot_answer, random_state=42)

            df_can_answer = df[~df['answer_t'].str.contains("I cannot answer")]
            df_can_answer = df_can_answer.sample(n=eval_per_model - max_cannot_answer, random_state=42)
            
            df = pd.concat([df_cannot_answer, df_can_answer], ignore_index=True)
            
        # if we do not have enough rows, we take more of "I cannot answer the question given the context." that have not been sampled yet
        if len(df) < eval_per_model:
            n_needed = eval_per_model - len(df)
            df_remaining = df[df['answer_t'].str.contains("I cannot answer")]
            df_remaining = df_remaining.drop(df.index)  # remove already sampled rows
            if len(df_remaining) > 0:
                replace_flag = len(df_remaining) < n_needed
                df_additional = df_remaining.sample(n=n_needed, replace=replace_flag, random_state=42)
                df = pd.concat([df, df_additional], ignore_index=True)
        
        # shuffle the final dataframe
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Add model name
        df['model'] = model_name

        all_results.append(df)

    df_all = pd.concat(all_results, ignore_index=True)

    # Add a row id
    df_all["row_id"] = range(len(df_all))

    for col in dims_eval:
        for corpus in ["s", "t"]:
            df_all[f"{col}_{corpus}"] = len(df_all) * [""]

    # order the columns so it appears:
    # id_row
    # question
    # passage_s
    # answer_s
    # dimensions for source (Faithfulness_s, Passage Dependence_s, Passage Reference Avoidance_s, Structured Response_s, Language Consistency_s)
    # passage_t
    # answer_t
    # dimensions for target (Faithfulness_t, Passage Dependence_t, Passage Reference Avoidance_t, Structured Response_t, Language Consistency_t)
    # model
    ordered_cols = (
        ["row_id", "question", "passage_s", "answer_s"]
        + [f"{col}_s" for col in dims_eval]
        + ["passage_t", "answer_t"]
        + [f"{col}_t" for col in dims_eval]
        + ["model"]
    )
    df_all = df_all[ordered_cols]

    # Create an identifier column for splitting
    split_idx = len(df_all) // 2
    df_part1 = df_all.iloc[:split_idx]
    df_part2 = df_all.iloc[split_idx:]

    # Save the files
    df_all.to_excel(path_save + "/answers_eval.xlsx", index=False)
    df_part2.to_excel(path_save + "/answers_eval_part1.xlsx", index=False)
    df_part2.to_excel(path_save + "/answers_eval_part2.xlsx", index=False)

    print(f"Split completed: {len(df_part1)} rows in part1, {len(df_part2)} rows in part2.")