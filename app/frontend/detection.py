import os
import requests

from flask import flash


MIND_WORKER_URL = os.environ.get('MIND_WORKER_URL')


def getTMDatasets(user_id: str):
    try:
        response = requests.get(f"{MIND_WORKER_URL}/datasets_detection", params={"email": user_id})
        if response.status_code == 200:
            data = response.json()
            dataset_detection = data.get("dataset_detection")
            print(dataset_detection)
            return dataset_detection

        else:
            flash(f"Error loading datasets: {response.text}", "danger")
            return {}

    except requests.exceptions.RequestException:
        flash("Backend service unavailable.", "danger")
        return {}
    
def getTMkeys(user_id: str, data_tm: dict):
    data_tm["email"] = user_id
    
    try:
        response = requests.get(f"{MIND_WORKER_URL}/detection/topickeys", json=data_tm)
        if response.status_code == 200:
            data = response.json()
            print(data)
            return data

        else:
            flash(f"Error loading topic keys: {response.get('error')}", "danger")
            return {}

    except requests.exceptions.RequestException:
        flash("Backend service unavailable.", "danger")
        return {}
    
def analyseContradiction(user_id: str, TM: str, topics: str):
    try:
        response = requests.post(f"{MIND_WORKER_URL}/detection/analyse_contradiction", json={"email": user_id, "TM": TM, "topics": topics})
        if response.status_code == 200:
            data = response.json()
            return data

        else:
            flash(f"Error loading datasets: {response.text}", "danger")
            print(response.text)
            return None

    except requests.exceptions.RequestException:
        flash("Backend service unavailable.", "danger")
        return None
    
def get_result_mind(user_id: str, TM: str, topics: str):
    try:
        response = requests.get(f"{MIND_WORKER_URL}/detection/result_mind", json={"email": user_id, "TM": TM, "topics": topics})
        if response.status_code == 200:
            data = response.json()
            return data

        else:
            flash(f"Error loading datasets: {response.text}", "danger")
            print(response.text)
            return None

    except requests.exceptions.RequestException:
        flash("Backend service unavailable.", "danger")
        return None
