import logging
from flask import Flask, render_template, jsonify, request, redirect, url_for, session
from flask_restx import Api
import requests
from pyfiglet import figlet_format
from termcolor import cprint
import os

# Import the namespace
from apis.namespace_eval import api as active_learning_api, labeled_ids

# Create Flask app
app = Flask(__name__)
# Set the secret key for session management
app.config['SECRET_KEY'] = os.urandom(24)
# Deactivate the default mask parameter
app.config["RESTX_MASK_SWAGGER"] = False

# Initialize API and add the namespace
api = Api(
    title="Rosie Corpus evaluation",
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
global_idx = 0
global_doc_id = None
doc_ids_to_label = None

# Ensure task stops gracefully
def save_and_stop():
    global stop_task, task_running
    stop_task = True
    task_running = False
    logger.info("Saving state before stopping the task")
    response = requests.post('http://app_eval_container:5003/test/SaveState/')
    logger.info(response)
    if response.status_code == 200:
        logger.info("State saved successfully")
    else:
        logger.error("Error saving state: %s", response.json())
    logger.info("Annotation task stopped")

# Endpoint to fetch a new document
@app.route('/get_new_document', methods=['POST'])
def get_new_document():
    global global_idx, global_doc_id, task_running
    logger.info("Fetching new document")
    logger.info("GLOBAL DOC ID: %s", global_doc_id)

    if global_doc_id is None or global_idx >= len(doc_ids_to_label):
        # Stop the task
        save_and_stop()
        logger.info("No more documents to label. Task completed.")
        return jsonify({"message": "No more documents to label. Task completed."}), 200

    while global_doc_id not in labeled_ids:
        idx = global_doc_id  # Use the global doc_id
        logger.info(f"Fetching document at index {idx}")
        logger.info("Calling GetDocumentEval")
        response = requests.post(f'http://app_eval_container:5003/test/GetDocumentEval/', data={'idx': idx})
        logger.info(f"Response Status Code: {response.status_code}")
        logger.info(f"Response Content: {response.content}")
        if response.status_code == 200:
            document = response.json()
            logger.info(f"Fetched document: {document}")
            global_idx += 1
            if global_idx < len(doc_ids_to_label):
                global_doc_id = doc_ids_to_label[global_idx]
            else:
                global_doc_id = None
            return jsonify(document)
        
        global_idx += 1
        if global_idx < len(doc_ids_to_label):
            global_doc_id = doc_ids_to_label[global_idx]
        else:
            global_doc_id = None
            break

    # Stop the task if no more documents are available
    save_and_stop()
    logger.info("No more documents to label. Task completed.")
    return jsonify({"message": "No more documents to label. Task completed."}), 200

@app.route('/submit_annotation', methods=['POST'])
def submit_annotation():
    label = request.form['label']
    idx = request.form['idx']
    logger.info("Calling LabelDocument")
    response = requests.post('http://app_eval_container:5003/test/LabelDocument/', data={'label': label, 'idx': idx})
    if response.status_code != 200:
        logger.error(f"Error calling LabelDocument: {response.json()}")
        return jsonify({"error": "Failed to label document"}), 500

    logger.info("Document labeled successfully")
    return jsonify({"message": "Document labeled successfully"}), 200

# Endpoint to start the task
@app.route('/start_annotation_task', methods=['POST'])
def start_annotation_task():
    global task_running, stop_task, doc_ids_to_label, global_idx, global_doc_id
    if not task_running:
        doc_ids_to_label_resp = requests.post('http://app_eval_container:5003/test/GetIdDocsToLabel/')
        if doc_ids_to_label_resp.status_code == 200:
            doc_ids_to_label = doc_ids_to_label_resp.json()["docs"]
        else:
            return jsonify({"error": "Failed to fetch document IDs"}), 500

        logger.info(doc_ids_to_label)
        
        global_idx = 0
        if doc_ids_to_label:
            global_doc_id = doc_ids_to_label[global_idx]
        else:
            return jsonify({"error": "No documents to label"}), 404
        
        task_running = True
        stop_task = False
        logger.info("Fetched document IDs to label")
        logger.info(f"Document IDs: {doc_ids_to_label}")
        logger.info("Starting annotation task")
    return redirect(url_for('annotate'))

# Endpoint to stop the task
@app.route('/stop_annotation_task', methods=['POST'])
def stop_annotation_task():
    global task_running, stop_task
    if task_running:
        save_and_stop()
        task_running = False
        if session.get("save_results"):
            return jsonify({"redirect": url_for('final_results')})
        else:
            return jsonify({"error": "Failed to save results"}), 500
    return jsonify({"message": "Annotation task is not running"}), 200

@app.route('/get_document_count', methods=['GET'])
def get_document_count():
    global doc_ids_to_label
    total_documents = len(doc_ids_to_label)
    remaining_documents = total_documents - len(labeled_ids)
    return jsonify({
        'total_documents': total_documents,
        'remaining_documents': remaining_documents
    })

# Serve the initial page with the "Start Annotation Task" button
@app.route('/index.html')
def index():
    return render_template('index.html')

# Serve the annotation page
@app.route('/annotate')
def annotate():
    return render_template('annotate_second_eval.html')

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
        app.run(host='0.0.0.0', port=5003, debug=True)
        # from waitress import serve
        # serve(app, host="0.0.0.0", port=2092)
    except Exception as e:
        logger.error("Error occurred while running the Flask app", exc_info=True)
