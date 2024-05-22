import pdb
import pathlib
import pandas as pd
import numpy as np
from scipy import sparse
from qa_metrics.prompt_llm import CloseLLM
import os
from dotenv import load_dotenv
import time

def get_most_representative_per_tpc(mat, topn=10):
    # Find the most representative document for each topic based on a matrix mat
    top_docs_per_topic = []

    for doc_distr in mat.T:
        sorted_docs_indices = np.argsort(doc_distr)[::-1]
        top = sorted_docs_indices[:topn].tolist()
        top_docs_per_topic.append(top)
    return top_docs_per_topic

model_path = pathlib.Path(
    "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/models/LDA/rosie_lg_lda_1_100")

def load_prompt_template(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        file_contents = file.read()
    return file_contents

path_env = pathlib.Path(os.getcwd()).parent.parent / '.env'
load_dotenv(path_env)
api_key = os.getenv("OPENAI_API_KEY")
gpt_model = CloseLLM()
gpt_model.set_openai_api_key(api_key)

prompt_template = load_prompt_template("./promt.txt")

path_corpus = model_path / "train_data" / "corpus_EN.txt"
with path_corpus.open("r", encoding="utf-8") as f:
    lines = [line for line in f.readlines()]
corpus = [line.rsplit(" 0 ")[1].strip().split() for line in lines]

ids = [line.split(" 0 ")[0] for line in lines]
df = pd.DataFrame({"lemmas": [" ".join(doc) for doc in corpus]})
df["doc_id"] = ids
df["len"] = df['lemmas'].apply(lambda x: len(x.split()))
print(df.head())

# Get raw text
raw = pd.read_parquet(
    "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/large/df_1.parquet")
df = df.merge(raw, how="inner", on="doc_id")[
    ["doc_id", "id_preproc", "lemmas_x", "text", "len"]]
df.head()

# Read thetas
thetas = sparse.load_npz(model_path.joinpath(
    'mallet_output/EN/thetas.npz')).toarray()

# Get topic keys
with open(model_path.joinpath('mallet_output/EN/topickeys.txt'), 'r') as file:
    lines = file.readlines()

# Strip newline characters and any leading/trailing whitespace from each line
topic_keys = [line.split("\t")[-1].split() for line in lines]

responses = []
for topic in range(len(topic_keys)):
    print(f"Topic {topic}: {topic_keys[topic]}")
    most_repr = get_most_representative_per_tpc(thetas, topn=3)[topic]
    import pdb; pdb.set_trace()
    most_repr_docs = [df[df.doc_id == f"EN_{id}"].text.values.tolist()[0][:500] for id in most_repr]
    
    time.sleep(1)
    this_tpc_promt = prompt_template.format(
        topic_keys[topic],
        *most_repr_docs
    )
    llm_response = gpt_model.prompt_gpt(
        prompt=this_tpc_promt, model_engine='gpt-3.5-turbo-instruct', temperature=0.1, max_tokens=500
    )  
    label, add, rationale = llm_response.split(" - ")
    
    responses.append(
        [topic_keys[topic], "\n".join(most_repr_docs), label, add, rationale]
    )

responses_df_en = pd.DataFrame(responses, columns=["topic", "most_repr_docs", "label", "add", "rationale"])

####
path_corpus = model_path / "train_data" / "corpus_ES.txt"
with path_corpus.open("r", encoding="utf-8") as f:
    lines = [line for line in f.readlines()]
corpus = [line.rsplit(" 0 ")[1].strip().split()
          for line in lines if line.rsplit(" 0 ")[1].strip().split() != []]

ids = [line.split(" 0 ")[0] for line in lines]
df = pd.DataFrame({"lemmas": [" ".join(doc) for doc in corpus]})
df["doc_id"] = ids
df["len"] = df['lemmas'].apply(lambda x: len(x.split()))
print(df.head())

# Get raw text
raw = pd.read_parquet(
    "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/large/df_1.parquet")
df = df.merge(raw, how="inner", on="doc_id")[
    ["doc_id", "id_preproc", "lemmas_x", "text", "len"]]
df.head()

# Read thetas
thetas = sparse.load_npz(model_path.joinpath(
    'mallet_output/ES/thetas.npz')).toarray()

# Get topic keys
with open(model_path.joinpath('mallet_output/ES/topickeys.txt'), 'r') as file:
    lines = file.readlines()

# Strip newline characters and any leading/trailing whitespace from each line
topic_keys = [line.split("\t")[-1].split() for line in lines]

responses = []
for topic in range(len(topic_keys)):
    print(f"Topic {topic}: {topic_keys[topic]}")
    most_repr = get_most_representative_per_tpc(thetas, topn=3)[topic]
    most_repr_docs = [df[df.doc_id == f"ES_{id}"].text.values.tolist()[0][:500] for id in most_repr]
    time.sleep(1)
    this_tpc_promt = prompt_template.format(
        topic_keys[topic],
        *most_repr_docs
    )
    llm_response = gpt_model.prompt_gpt(
        prompt=this_tpc_promt, model_engine='gpt-3.5-turbo-instruct', temperature=0.1, max_tokens=500
    )  
    label, add, rationale = llm_response.split(" - ")
    
    responses.append(
        [topic_keys[topic], "\n".join(most_repr_docs), label, add, rationale]
    )

responses_df_es = pd.DataFrame(responses, columns=["topic", "most_repr_docs", "label", "add", "rationale"])

responses_df_en.to_parquet("responses_df_en_2.parquet")
responses_df_es.to_parquet("responses_df_es_2.parquet")