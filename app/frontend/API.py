import os
import time
import uuid
import dotenv
import requests
import threading
from flask import Blueprint, jsonify, request


dotenv.load_dotenv()

MAX_CONCURRENT_TASKS = 4
RUNNING_TASKS = []
TASK_COUNTER = 0
MIND_WORKER_URL = os.getenv('MIND_WORKER_URL')

tasks_lock = threading.Lock()
preprocess_bp = Blueprint('preprocess_api', __name__, url_prefix='/api')


def background_preprocess_task(task_id, task_name, email, dataset, segmenter_data, translator_data, preparer_data):
    global RUNNING_TASKS
    
    with tasks_lock:
        task_state = next((t for t in RUNNING_TASKS if t['id'] == task_id), None)
    
    if not task_state:
        print(f"ERROR: Task {task_id} not found or was deleted")
        return

    print(f"Starting task: {task_name}")

    TOTAL_STEPS = 3
    if len(segmenter_data) == 0: TOTAL_STEPS = 2

    
    try:
        # Hacemos por ahora para los 3 pasos
        for step in range(1, TOTAL_STEPS + 1):
            # time.sleep(random.uniform(2, 5))
            
            percent = int((step / TOTAL_STEPS) * 100)
            
            task_state['percent'] = percent
            
            if step == 1:
                task_state['message'] = f"Step {step}/{TOTAL_STEPS}: Segmenting {dataset}..."
                response = requests.post(
                    f"{MIND_WORKER_URL}/preprocessing/segmenter",
                    json={
                        "email": email,
                        "dataset": dataset,
                        "segmenter_data": segmenter_data
                        }
                    )

            elif step == 2:
                task_state['message'] = f"Step {step}/{TOTAL_STEPS}: Translating {dataset}..."
                response = requests.post(
                    f"{MIND_WORKER_URL}/preprocessing/translator",
                    json={
                        "email": email,
                        "dataset": dataset,
                        "translator_data": translator_data
                        }
                    )
                
            elif step == 3:
                task_state['message'] = f"Step {step}/{TOTAL_STEPS}: Preparing {dataset}..."
                response = requests.post(
                    f"{MIND_WORKER_URL}/preprocessing/preparer",
                    json={
                        "email": email,
                        "dataset": dataset,
                        "preparer_data": preparer_data
                        }
                    )

            if response.status_code == 200:
                print(response.json().get("message"))
            else:
                raise Exception(f"Error in step {step}: {response.text}")
        
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


@preprocess_bp.route('/preprocess/Stage1', methods=['POST'])
def start_preprocess():
    global RUNNING_TASKS, TASK_COUNTER

    data = request.get_json()
    email = data.get('email')
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
        target=background_preprocess_task, 
        args=(new_task_id, new_task_name, email, dataset, segmentor_data, translator_data, preparer_data)
    )
    thread.start()
    
    return jsonify({
        'success': True, 
        'message': f"Process preprocessing dataset '{dataset}' iniciado correctamente.",
        'task_id': new_task_id
    }), 202 

@preprocess_bp.route('/preprocess_status')
def get_preprocess_status():
    """Endpoint consultado por el frontend para obtener el estado de todas las tareas."""
    
    with tasks_lock:
        status_list = []
        
        for task in RUNNING_TASKS:
            status_list.append({
                'id': task['id'],
                'name': task['name'],
                'percent': task['percent'],
                'message': task['message']
            })

    # 2. Devolver la respuesta
    return jsonify({
        'tasks': status_list,
        'running_count': len(status_list),
        'max_limit': MAX_CONCURRENT_TASKS
    })
