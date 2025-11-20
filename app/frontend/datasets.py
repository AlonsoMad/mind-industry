import os
import dotenv
import requests

from views import login_required_custom
from flask import Blueprint, render_template, flash, session


dataset_bp = Blueprint('dataset', __name__)
dotenv.load_dotenv()
MIND_WORKER_URL = os.environ.get('MIND_WORKER_URL')


@dataset_bp.route('/datasets')
@login_required_custom
def datasets():
    user_id = session.get('user_id')

    try:
        response = requests.get(f"{MIND_WORKER_URL}/datasets", params={"email": user_id})
        if response.status_code == 200:
            data = response.json()
            datasets = data.get("datasets", [])
            names = data.get("names", [])
            shapes = data.get("shapes", [])
            stages = data.get("stages", [])
            # flash(f"Datasets loaded successfully!", "success")
        else:
            flash(f"Error loading datasets: {response.text}", "danger")
            datasets, names, shapes, stages = [], [], [], []
    except requests.exceptions.RequestException:
        flash("Backend service unavailable.", "danger")
        datasets, names, shapes, stages = [], [], [], []

    return render_template("datasets.html", user_id=user_id, datasets=datasets, names=names, shape=shapes, stages=stages, zip=zip)
