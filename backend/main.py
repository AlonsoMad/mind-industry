import os
import json
from flask import Flask, request, jsonify

# Importamos las funciones de procesamiento que están montadas en /usr/src/mind
# from mind.processing_functions import run_heavy_preprocessing

# Configuraciones
app = Flask(__name__)
PORT = 5001 # Puerto de escucha del worker

@app.route('/process/preprocessing', methods=['POST'])
def handle_preprocessing():
    """Ruta que recibe la petición del Frontend y ejecuta el proceso pesado."""
    try:
        data = request.get_json()
        
        # 1. Extraer los parámetros necesarios para la función
        dataset_name = data.get('dataset_name')
        
        if not dataset_name:
            return jsonify({"status": "error", "message": "Falta 'dataset_name'."}), 400

        print(f"WORKER: Iniciando preprocesamiento para el dataset: {dataset_name}")

        # 2. Ejecutar la función pesada de tu módulo src/mind
        # run_heavy_preprocessing(dataset_name) es una función hipotética
        results = "llego"

        print(f"WORKER: Preprocesamiento completado exitosamente.")
        
        # 3. Devolver la respuesta al Frontend
        return jsonify({
            "status": "success",
            "message": f"Preprocesamiento completado para {dataset_name}.",
            "results": results
        }), 200

    except Exception as e:
        print(f"WORKER ERROR: {e}")
        return jsonify({"status": "error", "message": f"Fallo interno del Worker: {str(e)}", "next_step_available": False}), 500

if __name__ == '__main__':
    # Usamos host='0.0.0.0' para que sea accesible dentro de Docker
    app.run(host='0.0.0.0', port=PORT)
