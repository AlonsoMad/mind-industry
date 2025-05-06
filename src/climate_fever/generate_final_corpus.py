from typing import List
import pandas as pd # type: ignore
from tqdm import tqdm # type: ignore
from spacy_download import load_spacy # type: ignore
df_evidence = pd.read_json("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/climate_fever/evidence_candidates.json")
df_fever_original = pd.read_json("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/climate_fever/climate-fever-dataset-r1.jsonl", lines=True)
#df_transformed = pd.read_json("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/ata/climate_fever/questions_transformed.json", lines=True)

nlp = load_spacy("en_core_web_md")

def chunk_by_sentences(text: str, max_words: int = 250, overlap: int = 1, append_title: str = None) -> List[str]:
    """
    Divides the text into chunks based on sentences, i.e., it does not cut sentences when chunking.
    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    chunks = []
    current_chunk = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent.split())
        if current_len + sent_len > max_words:
            chunk_text = " ".join(current_chunk)
            if append_title:
                chunk_text = f"{append_title} {chunk_text}"
            chunks.append(chunk_text)
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []
            current_len = sum(len(s.split()) for s in current_chunk)

        current_chunk.append(sent)
        current_len += sent_len

    if current_chunk:
        chunk_text = " ".join(current_chunk)
        if append_title:
            chunk_text = f"{append_title} {chunk_text}"
        chunks.append(chunk_text)

    return chunks

def generate_chunk_dataframe(df: pd.DataFrame, max_words=200, overlap=1, include_title_in_text=False) -> pd.DataFrame:
    """
    Generates a DataFrame of chunks from the full article DataFrame.
    """
    chunks = []
    
    for _, row in df.iterrows():
        text = row["full_text"]
        title = row.get("title", row.get("page_id", "unknown"))

        chunk_list = chunk_by_sentences(
            text,
            max_words=max_words,
            overlap=overlap,
            append_title=title if include_title_in_text else None
        )

        for chunk_id, chunk_text in enumerate(chunk_list):
            chunks.append({
                "chunk_id": row["doc_id"] + "_" + str(chunk_id),
                "chunk_text": chunk_text,
                "source_title": title
            })

    chunk_df = pd.DataFrame(chunks)
    return chunk_df

# build dataset for training topic models from df_evidence
# keep a tracking of the ids for not adding duplicates
# we want to build a a correspondence between the claim and the doc_id relevant for each claim
# moreover, we want to build a dataframe with all the document
all_ids = []
corpus_train = []
correspondence_claim_article = []
for index, row in tqdm(df_evidence.iterrows(), total=df_evidence.shape[0]):

    this_claim_articles = []
    for sentence in row["top_sentences"]:
        
        if not sentence["doc_id"] in all_ids:
            all_ids.append(sentence["doc_id"])
            
            corpus_train.append(
                {
                    "doc_id": sentence["doc_id"],
                    "title": sentence["title"],
                    "full_text": sentence["article"],
                }
            )
            
        this_claim_articles.append(sentence["doc_id"])
    
    correspondence_claim_article.append(
        {
            "claim_id": row["claim_id"],
            "claim": row["claim"],
            "doc_ids": this_claim_articles,
        }
    )

# create dataframes
df_corpus_train = pd.DataFrame(corpus_train)
df_corpus_train.to_parquet("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/climate_fever/corpus_train.parquet")

correspondence_claim_article = pd.DataFrame(correspondence_claim_article)
correspondence_claim_article.to_json("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/climate_fever/correspondence_claim_article.json", orient="records", lines=True)

df_corpus_train_chunked = generate_chunk_dataframe(df_corpus_train, max_words=250, overlap=1, include_title_in_text=True)

df_corpus_train_chunked.to_parquet("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/climate_fever/corpus_train_chunked.parquet")
    
import pdb; pdb.set_trace()