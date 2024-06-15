"""
Main application entry point
"""
import logging
from flask import Flask, render_template, jsonify, request, redirect, url_for, session
from flask_restx import Api
from threading import Thread
import time
import requests
from pyfiglet import figlet_format
from termcolor import cprint
import os

# Import the namespace
from apis.namespace import api as active_learning_api

# Create Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Necessary for session management
# Deactivate the default mask parameter
app.config["RESTX_MASK_SWAGGER"] = False

# Initialize API and add the namespace
api = Api(
    title="EWB's Topic Modeling API",
    version='1.0',
    description='whatever',
)
api.add_namespace(active_learning_api, path='/test')
api.init_app(app)

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Global variables to control the task
task_running = False
stop_task = False
annotation_duration = 3600  # 1 hour in seconds

# Ensure task stops gracefully
def save_and_stop():
    global stop_task, task_running
    stop_task = True
    task_running = False
    logger.info("Saving state before stopping the task")
    response = requests.post('http://localhost:2095/test/SaveState/')  # Assuming the app runs on port 5000
    if response.status_code == 200:
        logger.info("State saved successfully")
    else:
        logger.error("Error saving state: %s", response.json())
    logger.info("Annotation task stopped")

# Endpoint to fetch a new document
@app.route('/get_new_document', methods=['POST'])
def get_new_document():
    if 'start_time' not in session:
        return jsonify({"error": "Annotation session not started"}), 403

    elapsed_time = time.time() - session['start_time']
    if elapsed_time > annotation_duration:
        return jsonify({"error": "Annotation session has ended"}), 403

    idx = 0  # Replace with actual logic to get document index
    logger.info("Calling getDocumentToLabel")
    response = requests.post('http://kumo01:2095/test/getDocumentToLabel/', data={'idx': idx})
    if response.status_code != 200:
        logger.error(f"Error calling getDocumentToLabel: {response.json()}")
        return jsonify({"error": "Failed to fetch document"}), 500

    document = response.json()
    logger.info(f"Fetched document: {document}")
    return jsonify(document)

# Endpoint to submit annotation
@app.route('/submit_annotation', methods=['POST'])
def submit_annotation():
    if 'start_time' not in session:
        return jsonify({"error": "Annotation session not started"}), 403

    elapsed_time = time.time() - session['start_time']
    if elapsed_time > annotation_duration:
        return jsonify({"error": "Annotation session has ended"}), 403

    label = request.form['label']
    logger.info("Calling LabelDocument")
    response = requests.post('http://kumo01:2095/test/LabelDocument/', data={'label': label})
    if response.status_code != 200:
        logger.error(f"Error calling LabelDocument: {response.json()}")
        return jsonify({"error": "Failed to label document"}), 500

    logger.info("Document labeled successfully")
    return jsonify({"message": "Document labeled successfully"}), 200

# Endpoint to start the task
@app.route('/start_annotation_task', methods=['POST'])
def start_annotation_task():
    global task_running, stop_task
    if not task_running:
        task_running = True
        stop_task = False
        session['start_time'] = time.time()
        logger.info("Starting annotation task")
        thread = Thread(target=run_annotation_task)
        thread.start()
    return redirect(url_for('annotate'))

# Endpoint to get the start time
@app.route('/get_start_time', methods=['GET'])
def get_start_time():
    if 'start_time' in session:
        return jsonify({'start_time': session['start_time']})
    return jsonify({'error': 'Annotation session not started'}), 403

# Background task to run the annotation task
def run_annotation_task(duration=3600):
    global stop_task, task_running
    start_time = time.time()
    while not stop_task and time.time() - start_time < duration:
        idx = 0  # Replace with actual logic to get document index
        logger.info("Calling getDocumentToLabel")
        response = requests.post('http://kumo01:2095/test/getDocumentToLabel/', data={'idx': idx})
        if response.status_code != 200:
            logger.error(f"Error calling getDocumentToLabel: {response.json()}")
            break

        document = response.json()
        logger.info(f"Fetched document: {document}")
        time.sleep(5)  # Adjust sleep time as needed

    if not stop_task:
        save_and_stop()

    logger.info("Annotation task stopped")
    task_running = False

# Endpoint to stop the task
@app.route('/stop_annotation_task', methods=['POST'])
def stop_annotation_task():
    global task_running, stop_task
    if task_running:
        save_and_stop()
        task_running = False
    return jsonify({"message": "Annotation task stopped"}), 200

# Serve the initial page with the "Start Annotation Task" button
@app.route('/index.html')
def index():
    return render_template('index.html')

# Serve the annotation page
@app.route('/annotate')
def annotate():
    return render_template('annotate.html')

# Serve the Swagger UI at /swaggerui.html
@app.route('/swaggerui.html')
def serve_swagger():
    return render_template('swaggerui.html')

# Redirect to Swagger UI
@app.route('/api/docs')
def swagger_ui():
    return redirect('/swaggerui.html')


if __name__ == '__main__':
    cprint(figlet_format("ROISE", font='big'), 'blue', attrs=['bold'])
    print('\n')

    logger.info("Starting the Flask app")

    try:
        app.run(host='0.0.0.0', port=2095, debug=True)
        # from waitress import serve
        # serve(app, host="0.0.0.0", port=2092)
    except Exception as e:
        logger.error(
            "Error occurred while running the Flask app", exc_info=True)
