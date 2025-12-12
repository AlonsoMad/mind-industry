import os
import dotenv
import requests

from database import db
from models import User
from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash


dotenv.load_dotenv()
auth_bp = Blueprint("auth", __name__)
MIND_WORKER_URL = os.getenv("MIND_WORKER_URL")


@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    email = data.get("email")
    username = data.get("username")
    password = data.get("password")
    password_rep = data.get("password_rep")

    if not email or not password:
        return jsonify({"error": "Missing fields"}), 400
    if password != password_rep:
        return jsonify({"error": "Passwords do not match"}), 400

    hashed_pw = generate_password_hash(password)
    new_user = User(email=email, username=username, password=hashed_pw)
    
    try:
        db.session.add(new_user)
        db.session.commit()
    except Exception as e:
        print(str(e))
        return jsonify({"error": "Failed to insert user: User already exists"}), 500        

    try:
        response = requests.post(
            f"{MIND_WORKER_URL}/create_user_folders",
            json={"email": email},
            timeout=10
        )
        if response.status_code != 200:
            raise Exception(f"Error from backend: {response.text}")

    except Exception as e:
        db.session.delete(new_user)
        db.session.commit()
        return jsonify({"error": f"Failed to create user folders: {str(e)}"}), 500

    return jsonify({"message": "User created successfully"}), 201

@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    user = User.query.filter_by(email=email).first()
    if not user or not check_password_hash(user.password, password):
        return jsonify({"error": "Invalid credentials"}), 401

    return jsonify({"message": "Login successful", "user_id": user.email, "username": user.username}), 200

@auth_bp.route("/user/<user_id>", methods=["PUT"])
def update_user(user_id):
    """
    Update User
    """
    data = request.get_json()
    email = data.get("email")
    username = data.get("username")
    password = data.get("password")
    password_rep = data.get("password_rep")

    user = User.query.filter_by(email=user_id).first()
    if not user:
        return jsonify({"error": "User not found"}), 404

    updated = False

    if email and email != user.email:
        old_email = user.email
        user.email = email
        updated = True
        try:
            response = requests.post(
                f"{MIND_WORKER_URL}/update_user_folders",
                json={"old_email": old_email, "new_email": email},
                timeout=10
            )
            if response.status_code != 200:
                raise Exception(response.text)
        except Exception as e:
            user.email = old_email
            db.session.commit()
            return jsonify({"error": f"Failed to update backend: {str(e)}"}), 500

    if username and username != user.username:
        user.username = username
        updated = True

    if password or password_rep:
        if not password or not password_rep:
            return jsonify({"error": "Both password and password_rep are required"}), 400
        if password != password_rep:
            return jsonify({"error": "Passwords do not match"}), 400
        user.password = generate_password_hash(password)
        updated = True

    if updated:
        db.session.commit()
        return jsonify({"message": "User updated successfully"}), 200
    else:
        return jsonify({"message": "No changes made"}), 200
