from dotenv import load_dotenv
from flask_cors import CORS
from flask import Flask

import os


load_dotenv()  # Load environment variables from .env file


def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="../app/static")
    app.config['SECRET_KEY'] = os.getenv("WEB_APP_KEY")

    CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes

    from views import views
    from auth import auth
    from API import preprocess_bp

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')
    app.register_blueprint(preprocess_bp)

    @app.after_request
    def add_cache_control(response):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    
    return app
