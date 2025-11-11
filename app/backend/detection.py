import os
import glob
import shutil
import numpy as np
import pandas as pd

from mind.cli import comma_separated_ints
from flask import Blueprint, jsonify, request
from utils import get_TM_detection, obtain_langs_TM


detection_bp = Blueprint("detection", __name__)


@detection_bp.route('detection/topickeys', methods=['GET'])
def getTopicKeys():
    try:
        data = request.get_json()
        email = data.get("email")
        dataset_name = data.get("dataset_name")
        topic_model = data.get("topic_model")

        if not email or not dataset_name or not topic_model:
            return jsonify({"error": "Missing one of the mandatory arguments"}), 400
        
        path = f'/data/{email}/3_Download/{topic_model}'

        # Read parquet to tm model
        if not os.path.exists(path):
            # print("not existing")
            print(path)
            return jsonify({"error": f'Not existing Topic Model "{topic_model}"'}), 500
        
        topic_keys = {
            "TM_name": topic_model,
            "lang": []
        }

        # Mallet files
        mallet_topic_keys = f'{path}/mallet_output/keys_*.txt'
        for file in glob.glob(mallet_topic_keys):
            if not os.path.isfile(file):
                continue

            lang = file.split('keys_')[-1].replace('.txt', '')
            topic_keys["lang"].append(lang)
            
            # "k" : {"name": title, "EN": [words], "ES: [words]"}
            with open(file, 'r') as f:
                i = 1
                keys = {}
                for topic in f:
                    # TODO future work call LLM for a title in topics
                    keys[i] = topic.replace('\n', '').split(' ')
                    i += 1
                topic_keys[lang] = keys
        
        return jsonify(topic_keys), 200
    
    except Exception as e:
        print(str(e))
        return jsonify({"error": f"ERROR: {str(e)}"}), 500

@detection_bp.route('/detection/analyse_contradiction', methods=['POST'])
def analyse_contradiction():
    try:
        data = request.get_json()
        print(data)
        email = data.get("email")
        TM = data.get("TM")
        topics = data.get("topics")
    
        print('analysing...')
        paths = get_TM_detection(email, TM)

        if isinstance(paths, tuple):
            pathTM, pathCorpus = paths[0], paths[1]
        else:
            raise Exception("Path TM failed")
        
        lang = obtain_langs_TM(pathTM)

        from mind.pipeline.pipeline import MIND

        # config part

        # source_corpus = {
        #     "corpus_path": pathCorpus,
        #     "thetas_path": f'{pathTM}/mallet_output/thetas_{lang[0]}.npz',
        #     "id_col": 'doc_id',
        #     "passage_col": 'text',
        #     "full_doc_col": 'full_doc',
        #     "language_filter": lang[0],
        #     "filter_ids": None,
        #     "load_thetas": True # Check
        # }

        source_corpus = {
            "corpus_path": pathCorpus,
            "thetas_path": f'{pathTM}/mallet_output/thetas_{lang[0]}.npz',
            "id_col": 'doc_id',
            "passage_col": 'lemmas',
            "full_doc_col": 'raw_text',
            "language_filter": lang[0],
            "filter_ids": None,
            "load_thetas": True, # Check
            "index_path": "." # Check not sure
        }

        # Seems not necessary !!!!

        # target_corpus = {
        #     "corpus_path": pathCorpus,
        #     "thetas_path": f'{pathTM}/mallet_output/thetas_{lang[1]}.npz',
        #     "id_col": 'doc_id',
        #     "passage_col": 'text',
        #     "full_doc_col": 'full_doc',
        #     "language_filter": lang[1],
        #     "filter_ids": None,
        #     "load_thetas": True # Check
        # }

        target_corpus = {
            "corpus_path": pathCorpus,
            "thetas_path": f'{pathTM}/mallet_output/thetas_{lang[1]}.npz',
            "id_col": 'doc_id',
            "passage_col": 'lemmas',
            "full_doc_col": 'raw_text',
            "language_filter": lang[1],
            "filter_ids": None,
            "load_thetas": True, # Check
            "index_path": "." # Check not sure
        }

        cfg = {
            "llm_model": "qwen:32b",
            "llm_server": "http://kumo.tsc.uc3m.es:11434",
            "source_corpus": source_corpus,
            # "target_corpus": target_corpus,
            # "dry_run": False,
            # "do_check_entailement": False,
            "config_path": '/src/config/config.yaml'
        }

        mind = MIND(**cfg)

        # run pipeline

        run_kwargs = {
            "topics": comma_separated_ints(topics), 
            "path_save": f'/data/{email}/4_Contradiction/{TM}_contradiction/mind_results.parquet', # Ver donde
            "previous_check": None
        }

        print('MIND class created. Running pipeline...')

        mind.run_pipeline(**run_kwargs)

        # import time
        # time.sleep(5)

        print('Finish pipeline')

        return jsonify({"message": f"Todo bien"}), 200
    
    except Exception as e:
        print(e)
        return jsonify({"error": f"ERROR: {str(e)}"}), 500
