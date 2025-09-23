import json
from tqdm import tqdm
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer, util
import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize
import pandas as pd
import re

ES_INDEX = "wikipedia_articles"
CLAIMS_JSON = "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/climate_fever/climate-fever-dataset-r1.jsonl"
TOP_K_DOCS = 100
TOP_K_SENTS = 100
TOP_K_RERANK = 50

embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
es = Elasticsearch("http://localhost:9200")

claims_df = pd.read_json(CLAIMS_JSON, lines=True)
final_results = []

def clean_wiki_text(text):
    text = re.sub(r"<ref[^>]*?>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>]*/>", "", text)
    text = re.sub(r"\{\{.*?\}\}", "", text)
    text = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"http[s]?://\S+", "", text)
    text = text.replace("|", "")
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"[!?]{2,}", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_wiki_sentence(sentence):
    return clean_wiki_text(sentence) 

for id, claim_entry in tqdm(claims_df.iterrows(), total=len(claims_df), desc="Processing claims"):
    claim_id = claim_entry["claim_id"]
    claim_text = claim_entry["claim"]
    #print(f"\033[92m{claim_text}\033[0m")

    # --- Document retrieval ---
    doc_resp = es.search(
        index=ES_INDEX,
        size=TOP_K_DOCS,
        query={"match": {"content": {"query": claim_text}}}
    )

    retrieved_articles = []
    for hit in doc_resp["hits"]["hits"]:
        doc_id = hit["_id"]
        content = hit["_source"]["content"]
        title = hit["_source"].get("title", None)
        retrieved_articles.append({
            "content": content,
            "clean_content": clean_wiki_text(content),
            "doc_id": doc_id,
            "title": title
        })

    #print(f"\033[91m Retrieved articles:\n - " + "\n - ".join(article['title'] for article in retrieved_articles) + "\033[0m")

    # --- Sentence-level matching ---
    all_sentences = []
    sentence_to_article = {}

    for article in retrieved_articles:
        sentences = sent_tokenize(article["clean_content"])
        for sent in sentences:
            cleaned = clean_wiki_sentence(sent)
            if cleaned:
                all_sentences.append(cleaned)
                sentence_to_article[cleaned] = {
                    "article": article["clean_content"],
                    "doc_id": article["doc_id"],
                    "title": article["title"]
                }

    if not all_sentences:
        continue

    claim_emb = embedding_model.encode(claim_text, convert_to_tensor=True)
    sent_embs = embedding_model.encode(all_sentences, convert_to_tensor=True)
    sim_scores = util.cos_sim(claim_emb, sent_embs)[0]
    top_k_idx = sim_scores.argsort(descending=True)[:TOP_K_SENTS]
    top_k_sents = [(all_sentences[i], sim_scores[i].item()) for i in top_k_idx]

    # --- Reranking ---
    reranked = sorted(top_k_sents, key=lambda x: x[1], reverse=True)[:TOP_K_RERANK]

    final_results.append({
        "claim_id": claim_id,
        "claim": claim_text,
        "top_sentences": [
            {
                "sentence": s,
                "score": round(score, 4),
                "article": sentence_to_article[s]["article"],
                "doc_id": sentence_to_article[s]["doc_id"],
                "title": sentence_to_article[s]["title"]
            }
            for s, score in reranked
        ]
    })

with open("evidence_candidates.json", "w", encoding="utf-8") as f:
    json.dump(final_results, f, indent=2, ensure_ascii=False)
