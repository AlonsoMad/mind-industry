import os
import glob
import shutil
import numpy as np
import pandas as pd

from flask import Blueprint, jsonify, request


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