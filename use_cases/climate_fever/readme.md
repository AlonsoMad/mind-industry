# Applying MIND in CLIMATE FEVER

In order to download the corpus with which we have tested the effectiveness of MIND in CLIMATE FEVER, the following steps have been done:

1. Download a Wikipedia dump

    ```bash
    wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
    ````

2. Transform the XML dump into a JSON

    ```bash
    python3 src/climate_fever/wiki_xml_to_json.py
    ```

3. Index the wikipedia dump into an ElasticSearch index:

    ```bash
    python3 src/climate_fever/index_elastic.py
    ```

4. Run evidence candidate retrieval. We automatically retrieve candidate evidence from the indexed Wikipedia using a three-step ECRS pipeline (pipeline mimicking what is done in the CLIMATE FEVER paper): 
    1. Document retrieval with BM25 on the ElasticSearch index using each claim as query.

    2. Sentence retrieval using all-mpnet-base-v2 embeddings to rank candidate sentences from the top-10 documents.

    3. Sentence re-ranking using cosine similarity to select the top-5 most relevant sentences.
    
    4. The final output includes each selected sentence, its score, and the cleaned full article from which it was extracted.

    ```bash
    python3 src/climate_fever/build_corpus.py
    ```