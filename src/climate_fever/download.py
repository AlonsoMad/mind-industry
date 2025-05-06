from pydantic import BaseModel # type: ignore
from typing import List, Tuple
import wikipedia # type: ignore
from spacy_download import load_spacy # type: ignore
import pandas as pd # type: ignore
from tqdm import tqdm # type: ignore
import hashlib
from sentence_transformers import SentenceTransformer # type: ignore
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
import warnings
from bs4 import GuessedAtParserWarning
import uuid

warnings.filterwarnings("ignore", category=GuessedAtParserWarning)


nlp = load_spacy("en_core_web_md")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
wikipedia.set_lang("en")


class QueryResponse(BaseModel):
    titles: List[str]

def extract_entities(query: str) -> List[str]:
    """Extract noun phrases from the query."""
    doc = nlp(query)
    #noun_chunks = [chunk.text for chunk in doc.noun_chunks]
    #return noun_chunks
    return list({ent.text for ent in doc.ents})

def search_wikipedia_for_entities(entities: List[str], max_results=3) -> List[str]:
    """Search Wikipedia for the given entities and return their titles. It returns a list of length max_results."""
    titles = []
    for entity in entities:
        try:
            search_results = wikipedia.search(entity, results=max_results)
            titles.extend(search_results)
        except Exception as e:
            print(f"Search failed for '{entity}': {e}")
    return list(set(titles))  # Remove duplicates


def process_query(query: str, max_results:int=3) -> QueryResponse:
    """Process the query to extract entities and search Wikipedia."""
    entities = extract_entities(query)
    if not entities:
        return QueryResponse(titles=[])
    titles = search_wikipedia_for_entities(entities, max_results)
    return QueryResponse(titles=titles)


def list_entities(query: str, max_results:int=3) -> List[str]:
    """List Wikipedia entities related to the query."""
    return process_query(query, max_results).titles


def fetch_page_content(title: str, original_query: str = None) -> Tuple[str, str]:
    try:
        page = wikipedia.page(title, auto_suggest=False)
        return page.title, page.content

    except wikipedia.DisambiguationError as e:
        
        if original_query:
            best_option = select_best_disambiguation_option(original_query, e.options)
            return fetch_page_content(best_option, original_query)
        else:
            return title, ""

    except wikipedia.PageError:
        return title, ""

    except Exception as e:
        return title, ""


def select_best_disambiguation_option(query: str, options: list) -> str:
    """
    Select the best disambiguation option based on cosine similarity of embeddings.
    """
    all_sentences = [query] + options
    embeddings = embedder.encode(all_sentences)

    query_embedding = embeddings[0]
    option_embeddings = embeddings[1:]

    similarities = cosine_similarity([query_embedding], option_embeddings)[0]

    best_idx = np.argmax(similarities)
    return options[best_idx]

def search_wikipedia_for_entities_with_tracing(entities: List[str], max_results: int = 3) -> List[Tuple[str, str]]:
    pairs = []
    for entity in entities:
        try:
            search_results = wikipedia.search(entity, results=max_results)
            for title in search_results:
                pairs.append((title, entity)) 
        except Exception as e:
            print(f"Search failed for '{entity}': {e}")
    return pairs


def get_relevant_wikipedia_texts(query: str, max_results: int = 3) -> List[Tuple[str, str]]:
    """Get relevant Wikipedia texts based on the query. It returns a list of tuples (title, content)."""
    entities = extract_entities(query)
    if not entities:
        return [] 

    results = []
    title_entity_pairs = search_wikipedia_for_entities_with_tracing(entities, max_results)

    for title, entity in title_entity_pairs:
        print(f"Fetching content for title: {title} (Entity: {entity})")
        page_title, content = fetch_page_content(title, query)
        if content:
            results.append((page_title, content, entity))
    
    import pdb; pdb.set_trace()
    return results

def title_to_id(title: str) -> str:
    return hashlib.sha1(title.encode("utf-8")).hexdigest()[:10]

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


def generate_chunk_dataframe(wiki_df: pd.DataFrame, max_words=200, overlap=1, include_title_in_text=False) -> pd.DataFrame:
    """
    Generates a DataFrame of chunks from the Wikipedia DataFrame.
    """
    chunks = []
    
    for _, row in wiki_df.iterrows():
        text = row["text"]
        title = row.get("title", row.get("page_id", "unknown"))

        chunk_list = chunk_by_sentences(
            text,
            max_words=max_words,
            overlap=overlap,
            append_title=title if include_title_in_text else None
        )

        for chunk_text in chunk_list:
            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "chunk_text": chunk_text,
                "source_title": title
            })

    chunk_df = pd.DataFrame(chunks)
    return chunk_df



import os

def main():
    seed = 1234
    sample_size = 1000
    df = pd.read_json("data/climate_fever/climate-fever-dataset-r1.jsonl", lines=True).sample(sample_size, random_state=seed)

    all_pages = {}  
    page_links = [] 

    output_dir = f"data/climate_fever/outs/{sample_size}"
    os.makedirs(output_dir, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        claim = row["claim"]
        claim_id = row.get("claim_id", f"claim_{idx}")

        print(f"\nProcessing claim: {claim} (ID: {claim_id})")

        results = get_relevant_wikipedia_texts(claim)

        for title, content, entity in results:
            page_id = title_to_id(title)
            if page_id not in all_pages:
                all_pages[page_id] = {
                    "text": content,
                    "title": title,
                    "entity": entity
                }
            page_links.append({
                "claim_id": claim_id,
                "page_id": page_id,
                "original_title": title
            })

        if idx > 0 and idx % 10 == 0:
            print(f" ** ** Saving intermediate results at step {idx}")
            wiki_df = pd.DataFrame([
                {
                    "page_id": pid,
                    "text": page_data["text"],
                    "title": page_data["title"],
                    "entity": page_data["entity"]
                }
                for pid, page_data in all_pages.items()
            ])
            links_df = pd.DataFrame(page_links)
            chunk_df = generate_chunk_dataframe(wiki_df, max_words=250, overlap=1, include_title_in_text=True)

            wiki_df.to_json(f"{output_dir}/wikipedia_pages_{seed}_{sample_size}_partial_{idx}.json", orient="records", lines=True)
            links_df.to_json(f"{output_dir}/claim_to_page_links_{seed}_{sample_size}_partial_{idx}.json", orient="records", lines=True)
            chunk_df.to_json(f"{output_dir}/wikipedia_chunks_{seed}_{sample_size}_partial_{idx}.json", orient="records", lines=True)

    print(" Saving final results...")
    wiki_df = pd.DataFrame([
        {
            "page_id": pid,
            "text": page_data["text"],
            "title": page_data["title"],
            "entity": page_data["entity"]
        }
        for pid, page_data in all_pages.items()
    ])
    links_df = pd.DataFrame(page_links)
    chunk_df = generate_chunk_dataframe(wiki_df, max_words=250, overlap=1, include_title_in_text=True)

    wiki_df.to_json(f"{output_dir}/wikipedia_pages_{seed}_{sample_size}.json", orient="records", lines=True)
    links_df.to_json(f"{output_dir}/claim_to_page_links_{seed}_{sample_size}.json", orient="records", lines=True)
    chunk_df.to_json(f"{output_dir}/wikipedia_chunks_{seed}_{sample_size}.json", orient="records", lines=True)

    print("\nFinal extracted pages:")
    print(wiki_df.head())
    print("\n Final claim ↔ page mapping:")
    print(links_df.head())


if __name__ == "__main__":
    main()
