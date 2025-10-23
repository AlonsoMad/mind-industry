import os
import dotenv
import requests

from enum import Enum
from tools.tools import *
from functools import wraps
from flask import Blueprint, render_template, request, flash, jsonify, session, send_file


class LastInstruction(str, Enum):
    idle = "Idle"
    explore_topics = "Explore topics"
    analyze_contradictions = "Analyze contradictions"

views = Blueprint('views', __name__)
dotenv.load_dotenv()

current_instruction = {"instruction": LastInstruction.idle, "last_updated": None}
MIND_WORKER_URL = os.environ.get('MIND_WORKER_URL')
AUTH_API_URL = f"{os.environ.get('AUTH_API_URL', 'http://auth:5002/')}/auth"    


def login_required_custom(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please log in to access this page.", "warning")
            return render_template("home.html")
        return f(*args, **kwargs)
    return decorated_function

@views.route('/')
def home():
    user_id = session.get('user_id')
    return render_template("home.html", user_id=user_id)

@views.route('/about_us')
def about_us():
    user_id = session.get('user_id')
    return render_template("about_us.html", user_id=user_id)

@views.get('/get_instruction')
def get_last_instruction():
    print("Fetching last instruction:", current_instruction["instruction"])
    return jsonify({
        "instruction": current_instruction["instruction"].value, 
        "last_updated": current_instruction["last_updated"]
    })

@views.route('/dataset_selection', methods=['POST'])
@login_required_custom
def dataset_selection():
    data = request.get_json()
    dataset = data.get('dataset')
    mind_api_url = f"{os.getenv('MIND_API_URL', 'http://mind:93')}"

    if not dataset:
        flash('No dataset provided', 'danger')
        return jsonify({'error': 'No dataset provided'}), 400

    print("Received dataset:", dataset)
    session['dataset'] = dataset

    response = requests.post(f'{mind_api_url}/initialize', json={'dataset': dataset})
    
    return jsonify({'message': 'Dataset received', 'dataset': dataset})

@views.route('/topic_selection', methods=['GET', 'POST'])
@login_required_custom
def topic_selection():
    data = request.get_json()
    topic_id = data.get('topic_id')
    mind_api_url = f"{os.getenv('MIND_API_URL', 'http://mind:93')}"

    if not topic_id:
        flash('No topic ID provided', 'danger')
        return jsonify({'error': 'No topic ID provided'}), 400

    print("Received topic ID:", topic_id)
    response = requests.get(f'{mind_api_url}/topic_documents', params={'topic_id': topic_id})
    
    return jsonify(response.json())


@views.route('/get_pyldavis', methods=['GET', 'POST'])
@login_required_custom
def get_pyldavis():

    #Get path to mallet ds
    mallet_folder = f"{os.getenv('TM_PATH', '/Data/mallet_folder')}"
    dataset = session['dataset'] if 'dataset' in session else 'en_2025_06_05_matched'

    pyLDAvis_path = os.path.join(mallet_folder, dataset, 'n_topics_50', 'mallet_output')

    if not os.path.exists(pyLDAvis_path):
        flash(f"Path {pyLDAvis_path} does not exist.", "danger")
        return jsonify({"error": "Mallet output path does not exist."}), 404
    else:
        vis_inputs = read_mallet(pyLDAvis_path)
        print(vis_inputs)

        # visuals = vis.prepare(topic_term_dists=vis_inputs['topic_term_dists_en'],
        #                     doc_topic_dists=vis_inputs['doc_topic_dists_en'],
        #                     doc_lengths=vis_inputs['doc_lengths_en'],
        #                     vocab=vis_inputs['vocab_en'],
        #                     term_frequency=vis_inputs['term_frequency_en'])

        # response = vis.prepared_data_to_json(visuals)


    return jsonify(response.json())

@views.route('/analyze_topic', methods=['GET', 'POST'])
@login_required_custom
def analyze_topic():
    '''
    Stores the selected topic ID in the session for later use.
    '''
    data = request.get_json()
    topic_id = data.get('topic_id')

    if not topic_id:
        return jsonify({"error": "No topic ID provided"}), 400
    
    session['selected_topic'] = topic_id  
    print(f"Selected topic ID stored in session: {topic_id}")
    return jsonify({"message": "Topic stored, awaiting sample count."})

@views.route('/submit_analysis', methods=['GET','POST'])
@login_required_custom
def submit_analysis():
    data = request.get_json()
    topic_id = session.get('selected_topic')
    n_samples = data.get("n_samples")
    mind_api_url = os.getenv("MIND_API_URL", "http://mind:93")
    print(f"Submitting analysis for topic ID: {topic_id} with n_samples: {n_samples}")

    if not topic_id or not n_samples:
        flash("Missing topic or sample count.", "danger")
        return jsonify({"error": "Missing topic or sample count."}), 400

    session['n_samples'] = n_samples

    try:
        response = requests.post(
            f"{mind_api_url}/run",
            params={"topic_number": topic_id, "n_sample": int(n_samples)}
        )

        if response.ok:
            flash("Topic analysis started!", "success")
        else:
            flash(f"Failed to start analysis: {response.text}", "danger")
    except Exception as e:
        flash(f"Error contacting MIND: {str(e)}", "danger")

    return jsonify({"message": "Sample recieved, starting analysis."})

@views.route('/mode_selection', methods=['GET', 'POST'])
@login_required_custom
def mode_selection():
    data = request.get_json()
    mode = data.get('instruction')
    mind_api_url = f"{os.getenv('MIND_API_URL', 'http://mind:93')}"
    print("Received mode:", mode)

    if not mode:
        flash('No mode provided', 'danger')
        return jsonify({'error': 'No mode provided'}), 400
    
    status_resp = requests.get(f"{mind_api_url}/status")
    status_data = status_resp.json()
    status = status_data.get("state", "unknown")
    
    if status in ["initialized", 'topic_exploration']:
        print(f"Current MIND status: {status}")
        if mode == "Explore topics":
            current_instruction["instruction"] = LastInstruction.explore_topics
            response = requests.get(f'{mind_api_url}/explore')
            return jsonify(response.json())
        
        elif mode == "Analyze contradictions":
            current_instruction["instruction"] = LastInstruction.analyze_contradictions
            response = requests.get(f'{mind_api_url}/explore')
            return jsonify(response.json())
        
        else:
            flash('Invalid mode selected', 'danger')
            return jsonify({'error': 'Invalid mode selected'}), 400
    else:
        flash(f"Select a dataset before exploring!", "warning")
        return jsonify({'error': f'MIND is not initialized. Current status: {status}'})

@views.route('/detection', methods=['GET', 'POST'])
@login_required_custom
def detection():
    user_id = session.get('user_id')
    mind_api_url = f"{os.getenv('MIND_API_URL', 'http://mind:93')}"
    dataset_path = os.getenv("DATASET_PATH", "/Data/3_joined_data")

    status = "idle"
    mind_info = {}
    ds_tuple = ( [], [], [])

    try:
        data = request.get_json()
        dataset = data.get('dataset', None) if data else None
        print(data)
    except Exception as e:
        dataset = None
    try:
        # Check current MIND status
        status_resp = requests.get(f"{mind_api_url}/status")
        status_data = status_resp.json()
        print(status_data)
        status = status_data.get("state", "unknown")
        #force completed
        # status = "completed"
        print(status)

        if status in ["idle", "failed"]:
            # init_resp = requests.post(f"{mind_api_url}/initialize")
            # flash(init_resp.json().get("message"), "info")
            ds_tuple = load_datasets(dataset_path)
            if ds_tuple[0]:
                flash("Datasets loaded successfully!", "success")
                flash("MIND is idle, please wait...", "warning")

            else:
                flash("No datasets found or error loading datasets.", "warning")

            if dataset:
                response = requests.post(f'{mind_api_url}/initialize', json={'dataset': dataset})

        elif status == "initializing":

            flash("MIND is initializing, please wait...", "warning")
        elif status == "initialized":
            ds_tuple = load_datasets(dataset_path)
            flash("MIND already initialized.", "success")
        elif status == "topic_exploration":
            try:
                explore_resp = requests.get(f'{mind_api_url}/explore')
                if explore_resp.status_code == 200:
                    mind_info = explore_resp.json().get("topic_information", {})
                else:
                    flash("Failed to explore topics.", "warning")
            except Exception as e:
                flash(f"Error contacting MIND: {str(e)}", "danger")
        elif status == "running":

            flash("MIND is currently processing.", "info")
        elif status == "completed":
            dataset_path = os.getenv("OUTPUT_PATH", "/Data/mind_folder")
            topic_id = session.get('selected_topic') or 5 #Just for testing
            n_samples = session.get('n_samples') or 5
            og_dataset = session.get('dataset') or 'en_2025_06_05_matched'
            try:
                response = requests.get(f"{MIND_WORKER_URL}/final_results/{og_dataset}")
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                else:
                    flash(f"Error loading final results: {response.text}", "danger")
                    results = []
            except requests.exceptions.RequestException:
                flash("Backend service unavailable.", "danger")
                results = []

            print(mind_info)
            # ds_tuple = (og_ds, [og_dataset], [shapes])

            # full_path = os.path.join(dataset_path,'final_results',f"topic_{topic_id}",f'samples_len_{n_samples}', 'results.parquet')

            # ds = pd.read_parquet(full_path)
            # mind_info = ds.to_dict(orient='records')

        else:
            flash("Unknown MIND state.", "danger")

    except requests.RequestException as e:
        flash(f"Error connecting to MIND: {e}", "danger")

    return render_template("detection.html", user_id=user_id, status=status, ds_tuple=ds_tuple, mind_info=mind_info)

@views.route('/get_results', methods=['GET','POST'])
def get_results():
    dataset_path = os.getenv("OUTPUT_PATH", "/Data/mind_folder")
    topic_id = session.get('selected_topic')
    n_samples = session.get('n_samples')
    if not topic_id or not n_samples:
        topic_id = 5  # Default topic ID for testing
        n_samples = 5 
    full_path = os.path.join(dataset_path,'final_results',f"topic_{topic_id}",f'samples_len_{n_samples}', 'results.parquet')
    return send_file(full_path, as_attachment=True, download_name=f"results_topic_{topic_id}_samples_{n_samples}.parquet")
