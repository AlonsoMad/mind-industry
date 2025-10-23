import os
import dotenv
import requests

from tools.tools import *
from auth import validate_password
from views import login_required_custom
from flask import Blueprint, render_template, request, flash, jsonify, session


profile_bp = Blueprint('profile', __name__)
dotenv.load_dotenv()
MIND_WORKER_URL = os.environ.get('MIND_WORKER_URL')
AUTH_API_URL = f"{os.environ.get('AUTH_API_URL', 'http://auth:5002/')}/auth"


@profile_bp.route('/profile', methods=['GET', 'POST'])
@login_required_custom
def profile():
    user_id = session.get('user_id')
    username = session.get('username')

    datasets = []
    dataset_path = os.path.join(os.getenv("OUTPUT_PATH", "/Data/mind_folder"), 'final_results')

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file == "results.parquet":
                full_path = os.path.join(root, file)
                try:
                    df = pd.read_parquet(full_path)
                    datasets.append({
                        "name": f'results_topic_{extract_topic_id(root)}_samples_{extract_sample_len(root)}',
                        "topic": extract_topic_id(root),
                        "sample_len": extract_sample_len(root),
                        "data": df
                    })
                except Exception as e:
                    print(f"Error reading {full_path}: {e}")

    if request.method == 'POST':

        new_email = request.form.get('email')
        new_username = request.form.get('username')
        new_password = request.form.get('password')
        new_password_rep = request.form.get('password_rep')
        
        update_payload = {}
        if new_email and new_email != session.get('email'):
            update_payload['email'] = new_email
        if new_username and new_username != session.get('username'):
            update_payload['username'] = new_username
        if new_password == new_password_rep and new_password != '' and new_password_rep != '':
            if validate_password(new_password, new_password_rep)[0]:
                update_payload['password'] = new_password
                update_payload['password_rep'] = new_password_rep

        if update_payload:
            try:
                response = requests.put(f"{AUTH_API_URL}/user/{session['user_id']}", json=update_payload)
                if response.status_code == 200:
                    if 'email' in update_payload:
                        session['user_id'] = update_payload['email']
                        user_id = update_payload['email']
                    if 'username' in update_payload:
                        session['username'] = update_payload['username']
                        username = update_payload['username']
                    flash("Profile updated successfully!", "success")
                else:
                    flash(response.json().get('error', 'Error updating profile'), "danger")
            except requests.exceptions.RequestException:
                flash("Authentication service unavailable", "danger")
        else:
            flash("No changes made.", "info")
    return render_template("profile.html", user_id=user_id, username=username, datasets=datasets)

@profile_bp.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    action = request.form.get('action')
    if not action:
        return jsonify({'error': 'No action selected'}), 400

    filename = file.filename
    temp_path = f"temp_{filename}"
    file.save(temp_path)

    email = session.get("user_id") 
    output_dir = f'{email}/{action}'
    output_file_path = os.path.join(output_dir, filename.split(".")[0])

    try:
        with open(temp_path, 'rb') as f:
            files = {'file': (filename, f)}
            data = {
                'path': output_file_path,
                'email': email,
                'stage': int(action.split('_')[0]),
                'dataset_name': filename 
            }
            resp = requests.post(f"{MIND_WORKER_URL}/upload_dataset", files=files, params=data)

            if not resp.ok:
                backend_error_message = resp.json().get('error', 'Backend error message not found')
                return jsonify({'error': backend_error_message}), resp.status_code

            backend_response = resp.json().get('message', 'Success')
    
    except requests.RequestException as e:
        return jsonify({'error': f'Backend request failed: {e}'}), 500
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return jsonify({'message': backend_response}), 200
