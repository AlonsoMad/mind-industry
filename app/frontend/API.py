import threading
import time
import uuid
import random
from flask import Blueprint, jsonify, request, Response


MAX_CONCURRENT_TASKS = 4
RUNNING_TASKS = []
TASK_COUNTER = 0

tasks_lock = threading.Lock()
preprocess_bp = Blueprint('preprocess_api', __name__, url_prefix='/api')

def background_preprocess_task(task_id, task_name):
    global RUNNING_TASKS
    
    with tasks_lock:
        task_state = next((t for t in RUNNING_TASKS if t['id'] == task_id), None)
    
    if not task_state:
        print(f"Error: Tarea {task_id} no encontrada al iniciar o fue eliminada.")
        return

    print(f"Iniciando tarea: {task_name}")

    TOTAL_STEPS = 5
    
    try:
        for step in range(1, TOTAL_STEPS + 1):
            time.sleep(random.uniform(2, 5)) # Simulación de trabajo
            
            percent = int((step / TOTAL_STEPS) * 100)
            
            task_state['percent'] = percent
            
            if step < TOTAL_STEPS:
                task_state['message'] = f"Paso {step}/{TOTAL_STEPS}: Ejecutando {task_name}..."
            else:
                task_state['message'] = f"¡{task_name} completada! Resultados listos."
                task_state['percent'] = 100 
                print(f"Tarea completada: {task_name}")

    except Exception as e:
        task_state['message'] = f"Error fatal en {task_name}: {e}"
        task_state['percent'] = -1 

    finally:
        time.sleep(5)
        
        with tasks_lock:
            RUNNING_TASKS = [t for t in RUNNING_TASKS if t['id'] != task_id]
        
        print(f"Tarea {task_name} removida de la lista de tareas activas.")


@preprocess_bp.route('/start_preprocess', methods=['POST'])
def start_preprocess():
    """Endpoint para iniciar un nuevo proceso de preprocesamiento."""
    global RUNNING_TASKS, TASK_COUNTER

    with tasks_lock:
        if len(RUNNING_TASKS) >= MAX_CONCURRENT_TASKS:
            return jsonify({
                'success': False,
                'message': f'Límite de {MAX_CONCURRENT_TASKS} procesos concurrentes alcanzado.'
            }), 409

        TASK_COUNTER += 1
        new_task_id = str(uuid.uuid4())
        new_task_name = f"Tarea-{TASK_COUNTER}"

        new_task_state = {
            'id': new_task_id,
            'name': new_task_name,
            'percent': 0,
            'message': f"Iniciando {new_task_name}...",
            'thread': None # Se asigna después
        }
        RUNNING_TASKS.append(new_task_state)

    thread = threading.Thread(
        target=background_preprocess_task, 
        args=(new_task_id, new_task_name)
    )
    thread.start()
    
    return jsonify({
        'success': True, 
        'message': f"Proceso '{new_task_name}' iniciado correctamente.",
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
