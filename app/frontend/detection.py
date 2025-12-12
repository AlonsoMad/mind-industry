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
            return data

        else:
            flash(f"Error loading topic keys: {response.json().get('error')}", "danger")
            return {}

    except requests.exceptions.RequestException:
        flash("Backend service unavailable.", "danger")
        return {}
    
def getModels():    
    try:
        response = requests.get(f"{MIND_WORKER_URL}/detection/models")
        if response.status_code == 200:
            data = response.json()
            return data["models"]

        else:
            flash(f"Error loading models: {response.json().get('error')}", "danger")
            return []

    except requests.exceptions.RequestException:
        flash("Backend service unavailable.", "danger")
        return []
    
def getDocProportion(user_id: str, TM: str):
    try:
        response = requests.get(f"{MIND_WORKER_URL}/detection/doc_representation", json={"email": user_id, "TM": TM})
        if response.status_code == 200:
            data = response.json()
            return data["docs_data"]

        else:
            flash(f"Error loading doc proportion: {response.json().get('error')}", "danger")
            return {}

    except requests.exceptions.RequestException:
        flash("Backend service unavailable.", "danger")
        return {}
    
def analyseContradiction(user_id: str, TM: str, topics: str, sample_size: int, config: dict):
    try:
        response = requests.post(f"{MIND_WORKER_URL}/detection/analyse_contradiction", json={"email": user_id, "TM": TM, "topics": topics, "sample_size": sample_size, "config": config})
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
    
def get_result_mind(user_id: str, TM: str, topics: str, start: int):
    try:
        response = requests.get(f"{MIND_WORKER_URL}/detection/result_mind", json={"email": user_id, "TM": TM, "topics": topics, "start": start})
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
