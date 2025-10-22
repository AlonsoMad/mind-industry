import os
import time
import uuid
import dotenv
import requests
import threading

from tools.tools import *
from views import login_required_custom
from flask import Blueprint, render_template, request,flash, jsonify, session


preprocess = Blueprint('preprocess', __name__)
dotenv.load_dotenv()
MIND_WORKER_URL = os.environ.get('MIND_WORKER_URL')

MAX_CONCURRENT_TASKS = 4
RUNNING_TASKS = []
TASK_COUNTER = 0

tasks_lock = threading.Lock()


@preprocess.route('/preprocessing', methods=['GET'])
@login_required_custom
def preprocessing():
    user_id = session.get('user_id')

    try:
        response = requests.get(f"{MIND_WORKER_URL}/datasets", params={"email": user_id})
        if response.status_code == 200:
            data = response.json()
            datasets = data.get("datasets", [])
            names = data.get("names", [])
        else:
            flash("Error loading datasets from backend.", "danger")
            datasets, names = [], []
    except requests.exceptions.RequestException:
        flash("Backend service unavailable.", "danger")
        datasets, names = [], []

    return render_template('preprocessing.html', user_id=user_id, datasets=datasets, names=names)

def wait_for_step_completion(step_id, step_name, timeout=600, interval=5):
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{MIND_WORKER_URL}/status/{step_id}")
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", "").lower()
            message = data.get("message", "")
            
            print(f"[{step_name}] Status: {status} - {message}")
            
            if status == "completed":
                print(f"[{step_name}] Completed.")
                return True
            elif status == "error":
                raise Exception(f"Backend error: {message}")
            
            time.sleep(interval)
        except requests.RequestException as e:
            print(f"[{step_name}] Error checking status: {e}")
            time.sleep(interval)
    raise TimeoutError(f"Timeout waiting for {step_name} to complete.")

def preprocess_stage1(task_id, task_name, email, dataset, segmenter_data, translator_data, preparer_data):
    global RUNNING_TASKS

    with tasks_lock:
        task_state = next((t for t in RUNNING_TASKS if t['id'] == task_id), None)

    if not task_state:
        print(f"ERROR: Task {task_id} not found or was deleted")
        return

    print(f"Starting task: {task_name}")

    TOTAL_STEPS = 3
    if len(segmenter_data) == 0:
        TOTAL_STEPS = 2

    try:
        for step in range(1, TOTAL_STEPS + 1):
            percent = int((step / TOTAL_STEPS) * 100)
            task_state['percent'] = percent

            if step == 1:
                step_name = "Segmenting"
                task_state['message'] = f"Step {step}/{TOTAL_STEPS}: {step_name} {dataset}..."
                response = requests.post(
                    f"{MIND_WORKER_URL}/segmenter",
                    json={"email": email, "dataset": dataset, "segmenter_data": segmenter_data}
                )
                response.raise_for_status()
                data = response.json()
                print(data.get("message"))
                step_id = data.get("step_id")
                if step_id:
                    wait_for_step_completion(step_id, step_name)

            elif step == 2:
                step_name = "Translating"
                task_state['message'] = f"Step {step}/{TOTAL_STEPS}: {step_name} {dataset}..."
                response = requests.post(
                    f"{MIND_WORKER_URL}/translator",
                    json={"email": email, "dataset": dataset, "translator_data": translator_data}
                )
                response.raise_for_status()
                data = response.json()
                print(data.get("message"))
                step_id = data.get("step_id")
                if step_id:
                    wait_for_step_completion(step_id, step_name)

            elif step == 3:
                step_name = "Preparing"
                task_state['message'] = f"Step {step}/{TOTAL_STEPS}: {step_name} {dataset}..."
                response = requests.post(
                    f"{MIND_WORKER_URL}/preparer",
                    json={"email": email, "dataset": dataset, "preparer_data": preparer_data}
                )
                response.raise_for_status()
                data = response.json()
                print(data.get("message"))
                step_id = data.get("step_id")
                if step_id:
                    wait_for_step_completion(step_id, step_name)

        task_state['message'] = f"¡{task_name} completed! Results saved."
        task_state['percent'] = 100
        print(f"Completed task: {task_name}")

    except Exception as e:
        task_state['message'] = f"FATAL ERROR in {task_name}: {e}"
        task_state['percent'] = -1

    finally:
        time.sleep(3)
        with tasks_lock:
            RUNNING_TASKS = [t for t in RUNNING_TASKS if t['id'] != task_id]
        print(f"Task: {task_name} removed from the active tasks")


@preprocess.route('/preprocess/Stage1', methods=['POST'])
def start_preprocess():
    global RUNNING_TASKS, TASK_COUNTER

    data = request.get_json()
    dataset = data.get('dataset')
    segmentor_data = data.get('segmentor_data')
    translator_data = data.get('translator_data')
    preparer_data = data.get('preparer_data')

    with tasks_lock:
        if len(RUNNING_TASKS) >= MAX_CONCURRENT_TASKS:
            return jsonify({
                'success': False,
                'message': f'Límite de {MAX_CONCURRENT_TASKS} procesos concurrentes alcanzado.'
            }), 409

        TASK_COUNTER += 1
        new_task_id = str(uuid.uuid4())
        new_task_name = f"Task-{TASK_COUNTER}"

        new_task_state = {
            'id': new_task_id,
            'name': new_task_name,
            'percent': 0,
            'message': "Preprocessing data...",
            'thread': None
        }
        RUNNING_TASKS.append(new_task_state)

    thread = threading.Thread(
        target=preprocess_stage1, 
        args=(new_task_id, new_task_name, session['user_id'], dataset, segmentor_data, translator_data, preparer_data)
    )
    thread.start()
    
    return jsonify({
        'success': True, 
        'message': f"Process preprocessing dataset '{dataset}' iniciado correctamente.",
        'task_id': new_task_id
    }), 202 
    