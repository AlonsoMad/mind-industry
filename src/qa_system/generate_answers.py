import pathlib
import pandas as pd
import ast
from prompter import Prompter
import re

def extend_to_full_sentence(
    text: str,
    num_words: int
) -> str:
    """Truncate text to a certain number of words and extend to the end of the sentence so it's not cut off.
    """
    text_in_words = text.split()
    truncated_text = " ".join(text_in_words[:num_words])
    
    # Check if there's a period after the truncated text
    remaining_text = " ".join(text_in_words[num_words:])
    period_index = remaining_text.find(".")
    
    # If period, extend the truncated text to the end of the sentence
    if period_index != -1:
        extended_text = f"{truncated_text} {remaining_text[:period_index + 1]}"
    else:
        extended_text = truncated_text
    
    # Clean up screwed up punctuations        
    extended_text = re.sub(r'\s([?.!,"])', r'\1', extended_text)
    
    return extended_text

####
top_k = 3
PATH_SOURCE = pathlib.Path("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/passages/26_jan/df_1.parquet")
METHOD_EVAL = "results_4_unweighted"

######################
# PATHS TO TEMPLATES #
######################
# 3. ANSWER GENERATION
_3_INSTRUCTIONS_PATH = "templates/question_answering.txt"
_4_INSTRUCTIONS_PATH = "templates/discrepancy_detection.txt"

llm_model = "qwen:32b"  #"qwen:32b" #"llama3.3:70b" #"llama3:70b-instruct" # llama3.1:8b-instruct-q8_0
#####

df= pd.read_excel("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/src/qa_system/df_q_29jan_topic_15v2_es_model30tpc_thr__dynamic.xlsx")
df = df[["question_id", "doc_id", "passage", "question", "full_doc", METHOD_EVAL]]

raw = pd.read_parquet(PATH_SOURCE)

prompter = Prompter(
    model_type=llm_model,
)

results = []
for id_row, row in df.iterrows():
    results_4_unweighted = ast.literal_eval(row["results_4_unweighted"])[0]
    top_docs = [el["doc_id"] for el in results_4_unweighted[:top_k]]
    
    for top_doc in top_docs:
        
        # ---------------------------------------------------------------------#
        # 3. ANSWER GENERATION
        # ---------------------------------------------------------------------#
        with open(_3_INSTRUCTIONS_PATH, 'r') as file: template = file.read()
        
        ######################################
        # GENERATE ANSWER IN SOURCE LANGUAGE 
        ######################################
        passage_s = row.passage
        full_doc_s = row.full_doc
        
        formatted_template = template.format(question=row.question, passage=passage_s,full_document=(extend_to_full_sentence(full_doc_s, 100)+ " [...]"))
        
        answer_s, _ = prompter.prompt(question=formatted_template)
        
        ######################################
        # GENERATE ANSWER IN TARGET LANGUAGE #
        ######################################
        passage_t = raw[raw.doc_id == top_doc].text.iloc[0]
        full_doc_t = raw[raw.doc_id == top_doc].full_doc.iloc[0]
        
        formatted_template = template.format(question=row.question, passage=passage_t,full_document=(extend_to_full_sentence(full_doc_t, 100)+ " [...]"))
        answer_t, _ = prompter.prompt(question=formatted_template)
        
        #import pdb; pdb.set_trace()
        
        if answer_t != "I cannot answer given the context." and answer_t != "I cannot answer as the passage contains personal opinions.":
            # ---------------------------------------------------------------------#
            # 4. DISCREPANCY DETECTION
            # ---------------------------------------------------------------------#
            with open(_4_INSTRUCTIONS_PATH, 'r') as file: template = file.read()
            
            question = template.format(question=row.question, answer_1=answer_s, answer_2=answer_t)
            
            discrepancy, _ = prompter.prompt(question=question)
            
        else:
            if answer_t == "I cannot answer as the passage contains personal opinions.":
                discrepancy = "I cannot answer as the passage contains personal opinions."
            else:
                discrepancy = "I cannot answer given the context."
            
        #import pdb; pdb.set_trace()
            
        results.append({
            "question_id": row.question_id,
            "doc_id": top_doc,
            "question": row.question,
            "passage_s": passage_s,
            "answer_s": answer_s,
            "passage_t": passage_t,
            "answer_t": answer_t,
            "discrepancy": discrepancy
        })
        
df_results = pd.DataFrame(results)
        
import pdb; pdb.set_trace()