import pandas as pd

from src.corpus_building.rosie_corpus import RosieCorpus

path_corpus_es = "/Users/lbartolome/Documents/GitHub/LinQAForge/data/source/corpus_rosie/corpus_strict_v2.0_es_compiled_passages.jsonl"
path_corpus_en = "/Users/lbartolome/Documents/GitHub/LinQAForge/data/source/corpus_rosie/corpus_strict_v3.1_en_compiled_passages_valid_links.jsonl"


rosie_corpus = RosieCorpus(path_corpus_en, path_corpus_es)

rosie_corpus.generate_tm_tr_corpus("/Users/lbartolome/Documents/GitHub/LinQAForge/data/source/corpus_rosie/df.parquet", sample=0.00001)