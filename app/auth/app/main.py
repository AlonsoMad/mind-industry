import dotenv

from flask import Flask
from routes import auth_bp
from database import init_db, db


def create_app():
    app = Flask(__name__)
    init_db(app)
    app.register_blueprint(auth_bp, url_prefix="/auth")

    with app.app_context():
        db.create_all()

    print(app.url_map)

    return app

if __name__ == "__main__":
    dotenv.load_dotenv()
    app = create_app()
    app.run(host="0.0.0.0", port=5002)
