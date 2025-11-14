import os
import glob
import json
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
            "filter_ids": None, # 
            "load_thetas": True, # Check
            "index_path": "." # Check not sure
        }

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
            "index_path": "pathTM" # Check not sure
        }

        cfg = {
            "llm_model": "llama3.1:8b",
            "llm_server": "http://kumo02.tsc.uc3m.es:11434",
            "source_corpus": source_corpus,
            "target_corpus": target_corpus,
            # "dry_run": False,
            # "do_check_entailement": True,
            "config_path": '/src/config/config.yaml'
        }

        # mind = MIND(**cfg)

        # run pipeline

        run_kwargs = {
            "topics": comma_separated_ints(topics), 
            "path_save": f'/data/{email}/4_Contradiction/{TM}_{topics}_contradiction/mind_results.parquet', # Ver donde
            "previous_check": None
        }

        print('MIND class created. Running pipeline...')

        # mind.run_pipeline(**run_kwargs)

        import time
        time.sleep(3)

        print('Finish pipeline')

        return jsonify({"message": f"Pipeline done correctly"}), 200
    
    except Exception as e:
        print(e)
        return jsonify({"error": f"ERROR: {str(e)}"}), 500
    
@detection_bp.route('/detection/result_mind', methods=['GET'])
def get_results_mind():
    try:
        data = request.get_json()
        print(data)
        email = data.get("email")
        TM = data.get("TM")
        topics = data.get("topics")

        df = pd.read_parquet(f'/data/{email}/4_Contradiction/{TM}_{topics}_contradiction/mind_results.parquet', engine='pyarrow')
        result_mind = df.to_dict(orient='records')
        result_columns = df.columns.tolist()

        columns_json = json.dumps([{"name": col} for col in df.columns])
        non_orderable_indices = json.dumps([i for i, col in enumerate(df.columns) if col in ['label', 'final_label']])

        return jsonify({"message": f"Results from MIND obtained correctly",
                        "result_mind": result_mind,
                        "result_columns": result_columns,
                        "columns_json": columns_json,
                        "non_orderable_indices": non_orderable_indices}), 200

    except Exception as e:
        print(e)
        return jsonify({"error": f"ERROR: {str(e)}"}), 500
    
@detection_bp.route('/detection/update_results', methods=['POST'])
def update_result_mind():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400
        
        TM = request.form.get("TM")
        topics = request.form.get("topics")
        email = request.form.get("email")
        if not TM or not topics or not email:
            return jsonify({"error": "Missing parameters"}), 400

        df = pd.read_excel(file, engine='openpyxl')
        keys = []
        for key in df.keys():
            values = key.replace('\n', '').split(' ')
            if 'label' in values:
                keys.append('label')
            elif 'final_label' in values:
                keys.append('final_label')
            else:
                keys.append(values[0])

        df.columns = keys
        df.to_parquet(f'/data/{email}/4_Contradiction/{TM}_{topics}_contradiction/mind_results.parquet', engine='pyarrow')

        return jsonify({"message": f"Results from MIND saved correctly"}), 200

    except Exception as e:
        print(e)
        return jsonify({"error": f"ERROR: {str(e)}"}), 500
