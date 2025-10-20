from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    app.config['SQLALCHEMY_DATABASE_URI'] = \
        "postgresql://auth_user:auth_pass@db:5432/auth_db"
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY'] = "supersecretkey"

    db.init_app(app)

    from auth_routes import auth_bp
    app.register_blueprint(auth_bp, url_prefix='/auth')

    return app
