import pathlib
import re
import ast
import numpy as np
import pandas as pd
from scipy import sparse # type: ignore
import torch
from prompter import Prompter
from sklearn.svm import OneClassSVM
from sentence_transformers import SentenceTransformer
import spacy

def get_doc_top_tpcs(doc_distr, topn=10):
    sorted_tpc_indices = np.argsort(doc_distr)[::-1]
    top = sorted_tpc_indices[:topn].tolist()
    return [(k, doc_distr[k]) for k in top if doc_distr[k] > 0]

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

######################
# PATHS TO TEMPLATES #
######################
# 0. SUMMARIZATION
#_0_INSTRUCTIONS_PATH = "templates/0_instruction_prompt.txt"
#_0_SYSTEM_PATH = "templates/0_system_prompt.txt"
# 1. QUESTION GENERATION
_1_INSTRUCTIONS_PATH = "templates/1_instruction_prompt.txt"
_1_SYSTEM_PATH = "templates/1_system_prompt.txt"
# 2. SEARCH QUERY GENERATION
_2_INSTRUCTIONS_PATH = "templates/2_instruction_prompt.txt"
_2_SYSTEM_PATH = "templates/2_system_prompt.txt"
# 3. ANSWER GENERATION
_3_INSTRUCTIONS_PATH = "templates/3_instruction_prompt.txt"
_3_SYSTEM_PATH = "templates/3_system_prompt.txt"
# CONTRADICTION CHECK
_4_INSTRUCTIONS_PATH = "templates/4_instruction_prompt.txt"
_4_SYSTEM_PATH = "templates/4_system_prompt.txt"

####################
# LLM MODEL TO USE #
####################
llm_model = "qwen:32b" #"llama3.3:70b" #"llama3:70b-instruct" # llama3.1:8b-instruct-q8_0
prompter = Prompter(
    model_type=llm_model,
)

################
# INDICES PATH #
################
INDICES_PATH = "indices_multi"
#client = chromadb.PersistentClient(path=INDICES_PATH)
LLM_MODEL_EMBEDDINGS = 'mxbai-embed-large:latest'
BATCH_SIZE = 512
EMBEDDING_URL = "http://kumo01.tsc.uc3m.es:11434/api/embeddings"
N_RESULTS = 3

################
# ENGLISH DATA #
################
PATH_SOURCE = pathlib.Path("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/passages/26_jan/df_1.parquet")
PATH_MODEL = pathlib.Path("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/models/26_jan/poly_rosie_1_15")
print("-- Testing query from English corpus...")

# Filter English documents
thetas_en = sparse.load_npz(PATH_MODEL / "mallet_output" / "thetas_EN.npz").toarray()
raw = pd.read_parquet(PATH_SOURCE)
raw_en = raw[raw.doc_id.str.contains("EN")].copy()
raw_en["thetas"] = list(thetas_en)
raw_en["top_k"] = raw_en["thetas"].apply(lambda x: get_doc_top_tpcs(x, topn=10))
raw_en["main_topic"] = raw_en["thetas"].apply(lambda x: np.argmax(x))

# Initialize embedding function
#embedding_function = OllamaEmbeddingFunction(
#    model_name=LLM_MODEL_EMBEDDINGS,
#    url=EMBEDDING_URL,
#)

MODEL = SentenceTransformer("multi-qa-mpnet-base-cos-v1")

def embedding_function(text):
    return MODEL.encode(text)

################
# TRF MODEL    #
################
trf_model = SentenceTransformer('all-MiniLM-L6-v2')

################
# SPACY MODEL  #
################
nlp = spacy.load("en_core_web_sm")


#######################
# CONFIGURE One-SVM   #
#######################
path_tr_data="/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/src/qa_system/tr_data/questions_rosie/FullTrialQa7152024.csv"
df_positive = pd.read_csv(path_tr_data)
df_filtered = df_positive.dropna(
    subset=['question', 'answerPassageText'])
positive_questions = list(
    set(df_filtered[['question', 'answerPassageText']].question.values.tolist()))
ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
ocsvm.fit(trf_model.encode(positive_questions))

for topic_id in [3]: #range(num_topics)
    
    print("#"*50)
    print(f"--- Processing TOPIC {topic_id} ---")
    print("#"*50)
    
    #################################
    # Load collection for the topic #
    #################################
    #collection = client.get_collection(name=f"docs_{topic_id}_es")
    #collection = client.get_collection(name=f"docs_all_es")
    
    
    ############################
    # 1. QUESTION GENERATION   #
    ############################
    processed_passages = 0
    info_topic = []
    for pass_id, pass_row in raw_en.iterrows():
        
        if processed_passages < 200:
        
            print(f"--- Processing PASSAGE {pass_id} ---")
            print(f"-- -- -- Passage: {pass_row.text}")

            with open(_1_INSTRUCTIONS_PATH, 'r') as file:
                template = file.read()
            
            question = template.format(passage=pass_row.text, full_document=(extend_to_full_sentence(pass_row.full_doc, 100)+ " [...]"))
            
            questions, _ = prompter.prompt(
                system_prompt_template_path=_1_SYSTEM_PATH,
                question=question
            )

            reason = questions.split("[[ ## QUESTIONS ## ]]")[0]
            try:
                questions = [q.strip() for q in questions.split("[[ ## QUESTIONS ## ]]")[1].split("[[ ## completed ## ]]")[0].replace("\\n", "\n").split("\n") if q.strip()]
            except Exception as e:
                import pdb; pdb.set_trace()
            
            if questions == []:
                print("-- -- No questions generated.")
                questions_keep = []
            
            else:
                # predict if the questions are good or bad
                questions_embeddings = trf_model.encode(questions)
                predictions = ocsvm.predict(questions_embeddings)
                questions_keep = [q for q, p in zip(questions, predictions) if p == 1]
                
            info_topic.append({
                "pass_id": pass_id,
                "passage": pass_row.text,
                "full_doc": pass_row.full_doc,
                "questions": questions_keep,
                "all_questions": questions,
                "reason": reason
            })
            
            processed_passages += 1
            
            import pdb; pdb.set_trace()

    # convert to dataframe
    df_info_topic = pd.DataFrame(info_topic)
    
    # Create subdataFrame keeping the rows whose questions are not empty and transform it into a new DataFrame where each question is a new row, keeping the original pass_id and passage. We create a new column for the question_id
    df_info_topic = df_info_topic[df_info_topic.questions.apply(lambda x: len(x) > 0)]
    df_info_topic = df_info_topic.explode('questions').reset_index(drop=True)
    df_info_topic['question_id'] = df_info_topic.index
    
    # Extract entities from the questions
    df_info_topic['entities'] = df_info_topic.questions.apply(lambda x: [ent.text for ent in nlp(x).ents])
    
    df_info_topic["queries"] = None
    for question_id, question_row in df_info_topic.iterrows():
        
        print(f"--- Processing QUESTION {question_id} ---")
        print(f"Question: {question_row.questions}") 
        
        with open(_2_INSTRUCTIONS_PATH, 'r') as file:
            template = file.read()
        
        question = template.format(
            question=question_row.questions,
            entities=question_row.entities,
            context=question_row.passage
        )
        
        queries, _ = prompter.prompt(
            system_prompt_template_path=_2_SYSTEM_PATH,
            question=question
        )
        
        # extract the query
        queries_clean = [qu.strip() for qu in queries.split("[[ ## SEARCH_QUERY ## ]]")[1].split("[[ ## completed ## ]]")[0].strip().split(";") if qu.strip()]
        
        df_info_topic.loc[question_id, 'queries'] = str(queries_clean)
    df_info_topic.to_xlsx(f"questions_topic_{tpc}.xlsx")
    import pdb; pdb.set_trace()
    
    # Retrieve related passages from the collection
    df_info_topic["answers"] = None
    df_info_topic["ids_passages_used"] = None
    df_info_topic["text_passages_used"] = None
    for question_id, question_row in df_info_topic.iterrows():
        
        print(f"--- Processing QUESTION {question_id} ---")
        print(f"Question: {question_row.questions}") 
        
        queries = ast.literal_eval(question_row.queries)
        queries = [q for q in queries if q]
        
        answers = []
        ids_passages_used = []
        text_passages_used = []
        for query in queries:
            embedding = embedding_function(query)
            import pdb; pdb.set_trace()
            #[0]
            results = collection.query(query_embeddings=[embedding_function(query).tolist()],  n_results=N_RESULTS)
            
            passages = results["documents"][0]
            ids = results["ids"][0]
            full_docs = [meta["full_doc"] for meta in results["metadatas"][0]]
            
            with open(_3_INSTRUCTIONS_PATH, 'r') as file: template = file.read()
            
            for passage, full_doc in zip(passages, full_docs): 
                question = template.format(
                    question=question_row.questions,
                    question_context=query,
                    context_passage=passage,
                    context_full_document=(extend_to_full_sentence(full_doc, 100)+ " [...]")
                )
                answer, _ = prompter.prompt(
                    system_prompt_template_path=_3_SYSTEM_PATH,
                    question=question
                )
                
                answer_extracted = answer.split("[[ ## ANSWER ## ]]")[1].split("[[ ## completed ## ]]")[0].strip()
                
                if "i cannot answer" not in answer_extracted.strip(".").lower():
                    answers.append(answer_extracted)
                    ids_passages_used.append(ids)
                    text_passages_used.append(passage)
        
        df_info_topic.loc[question_id, 'answers'] = str(answers)
        df_info_topic.loc[question_id, 'ids_passages_used'] = str(ids_passages_used)
        df_info_topic.loc[question_id, 'text_passages_used'] = str(text_passages_used)
                
    # Check for contradictions
    df_info_topic["contradiction"] = None
    df_info_topic["contradiction_reason"] = None
    df_info_topic["contraction_type"] = None
    for question_id, question_row in df_info_topic.iterrows():
        
        print(f"--- Processing QUESTION {question_id} ---")
        print(f"Question: {question_row.questions}") 
        
        if question_row.answers == "[]":
            contradictions = []
            contradiction_reason = "No answers found."
            continue
        
        with open(_4_INSTRUCTIONS_PATH, 'r') as file:
            template = file.read()
        
        question = template.format(
            passage=question_row.passage,
            context=question_row.queries,
            question=question_row.questions,
            answer=question_row.answers
        )
        
        contradictions, _ = prompter.prompt(
            #system_prompt_template_path=_4_SYSTEM_PATH,
            question=question
        )
        
        contradictions_reason = contradictions.split("[[ ## reasoning ## ]]")[1].split("[[ ## CONTRADICTION ## ]]")[0].strip()
        if "CONTRADICTION_TYPE" in contradictions:
            contradictions_clean = contradictions.split("[[ ## CONTRADICTION ## ]]")[1].split("[[ ## CONTRADICTION_TYPE ## ]]")[0].strip()
            contraction_type = contradictions.split("[[ ## CONTRADICTION_TYPE ## ]]")[1].split("[[ ## completed ## ]]")[0].strip()
        else:
            contradictions_clean = contradictions.split("[[ ## CONTRADICTION ## ]]")[1].split("[[ ## completed ## ]]")[0].strip()
            contraction_type = ""
                    
        df_info_topic.loc[question_id, 'contradiction'] = contradictions_clean
        df_info_topic.loc[question_id, 'contradiction_reason'] = contradictions_reason
        df_info_topic.loc[question_id, 'contraction_type'] = contraction_type
    
    df_info_topic. to_csv(f"df_info_topic_qwen_{topic_id}.csv")

import pdb; pdb.set_trace()        
        
        
        
        
        
        


       
        


    
    
    
    