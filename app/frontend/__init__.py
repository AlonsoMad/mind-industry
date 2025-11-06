import os

from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
from flask_session import Session


load_dotenv()  # Load environment variables from .env file


def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="../app/static")
    app.config['SECRET_KEY'] = os.getenv("WEB_APP_KEY")
    app.config['SESSION_TYPE'] = 'filesystem'

    Session(app)

    CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes

    from auth import auth
    from views import views
    from profile import profile_bp
    from datasets import dataset_bp
    from preprocessing import preprocess

    app.register_blueprint(preprocess, url_prefix='/')
    app.register_blueprint(profile_bp, url_prefix='/')
    app.register_blueprint(dataset_bp, url_prefix='/')
    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')

    @app.after_request
    def add_cache_control(response):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    
    return app
