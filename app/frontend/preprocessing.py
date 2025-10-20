from flask import Blueprint, render_template, request,flash, jsonify, session

import dotenv
import os
import pandas as pd
import requests
from tools.tools import *
from views import login_required_custom


preprocess = Blueprint('preprocess', __name__)
dotenv.load_dotenv()
MIND_WORKER_URL = os.environ.get('MIND_WORKER_URL')
AUTH_API_URL = f"{os.environ.get('AUTH_API_URL', 'http://auth:5002/')}/auth"


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

@preprocess.route('/confirm_preprocessing_step2', methods=['POST'])
@login_required_custom
def confirm_preprocessing_step2():
    """Recibe la petición del HTML y la reenvía al Worker."""
    
    # 2. Obtener los datos del frontend HTML
    data = request.get_json()
    
    # 3. Hacer la llamada HTTP POST al servicio mind_worker
    try:
        worker_response = requests.post(
            f"{MIND_WORKER_URL}/process/preprocessing",
            json=data # Envía el JSON recibido directamente al worker
        )
        
        # 4. Devolver la respuesta del worker (status code y contenido) al frontend
        return jsonify(worker_response.json()), worker_response.status_code

    except requests.exceptions.ConnectionError:
        return jsonify({
            "status": "error",
            "message": "Error de conexión con el Worker de Procesamiento. Asegúrate de que el servicio 'mind_worker' está corriendo.",
            "next_step_available": False
        }), 503 # Service Unavailable

    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"Error desconocido en el proxy: {str(e)}",
            "next_step_available": False
        }), 500
    