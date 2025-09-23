from elasticsearch import Elasticsearch, helpers
import json
from tqdm import tqdm

ES_INDEX = "wikipedia_articles"
JSONL_PATH = "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/src/climate_fever/wikipedia_articles.jsonl"
ES_HOST = "http://localhost:9200"

es = Elasticsearch(
    hosts=[ES_HOST],
    request_timeout=60,
    retry_on_timeout=True
)

if es.indices.exists(index=ES_INDEX):
    es.indices.delete(index=ES_INDEX)

es.indices.create(index=ES_INDEX, body={
    "mappings": {
        "properties": {
            "title": {"type": "text"},
            "content": {"type": "text"}
        }
    }
})

print(f"Index {ES_INDEX} created.")
print(f"Counting lines in {JSONL_PATH}...")
with open(JSONL_PATH, "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)
print(f"Total lines: {total_lines}")

def generate_docs(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="Indexing in Elasticsearch"):
            try:
                data = json.loads(line)
                title = data.get("title", "").strip()
                content = data.get("text", "").strip()
                if not title or not content:
                    continue
                yield {
                    "_index": ES_INDEX,
                    "_source": {
                        "title": title,
                        "content": content
                    }
                }
            except json.JSONDecodeError:
                continue

helpers.bulk(es, generate_docs(JSONL_PATH), chunk_size=500)

print(f"Indexing completed. {total_lines} documents indexed. The index is {ES_INDEX}.")
