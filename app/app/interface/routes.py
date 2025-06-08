from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, make_response, send_file
from datetime import datetime, timedelta
import requests
import pstats
import os
import uuid
import zipfile
import shutil
from werkzeug.utils import secure_filename
from random import random


interface = Blueprint('interface', __name__)

FASTAPI_URL = "http://localhost:8000"


UPLOAD_FOLDER = 'app/static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
UPLOAD_DIR = "static/uploaded_datasets"
MODEL_DIR = "static/models"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


@interface.route('/')
@interface.route('/home')
def home():
    return render_template('index.html')


@interface.route('/predict2', methods=['GET', 'POST'])
def predict2():
    if request.method == 'POST':
        # Retrieve the uploaded image
        image = request.files.get('image')

        if not image:
            flash('Please upload an image.', 'danger')
            return redirect(url_for('interface.predict'))

        try:
            # Save the uploaded image locally in the static/uploads folder
            unique_filename = f"{uuid.uuid4()}_{image.filename}"
            saved_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            image.save(saved_path)

            # Send image to FastAPI for prediction
            files = {'image': (image.filename, open(saved_path, 'rb'), image.mimetype)}
            response = requests.post(f"{FASTAPI_URL}/predict", files=files)
            response.raise_for_status()  
            prediction_data = response.json()

            # Extract prediction results
            label = prediction_data.get('label', 'Unknown')
            confidence = prediction_data.get('confidence', 0.0)

            return render_template(
                'predict.html',
                prediction=label,
                confidence=f"{confidence:.2f}%",
                uploaded_image_url=url_for('static', filename=f"uploads/{unique_filename}")
            )

        except requests.exceptions.RequestException as e:
            flash(f"Error connecting to prediction service: {e}", 'danger')
            return redirect(url_for('interface.predict'))

    return render_template('predict.html', prediction=None)


@interface.route('/predict', methods=['GET', 'POST'])
def predict():
    models = []
    prediction_result = None

    # Fetch the available models
    try:
        response = requests.get(f"{FASTAPI_URL}/list-models/")
        response.raise_for_status()
        data = response.json()
        models = data.get("models", [])
    except requests.exceptions.RequestException as e:
        flash(f"Error fetching models: {str(e)}", "danger")

    if request.method == 'POST':
        image = request.files.get('image')
        selected_model = request.form.get('model')

        if not image or not selected_model:
            flash("Please upload an image and select a model.", "danger")
            return render_template("predict.html", models=models)


        try:
            files = {'image': (image.filename, image.stream, image.mimetype)}
            form_data = {'model_name': selected_model}  # Pass model name as form data
            response = requests.post(
                f"{FASTAPI_URL}/predict/",
                files=files,
                data=form_data,  # Include model name in the request
            )
            response.raise_for_status()
            prediction_result = response.json()
        except requests.exceptions.RequestException as e:
            flash(f"Error during prediction: {str(e)}", "danger")

    return render_template("predict.html", models=models, prediction_result=prediction_result)


"""

@interface.route('/predict', methods=['GET', 'POST'])
def predict():
    models = []
    prediction_result = None

    # Fetch available models
    try:
        response = requests.get(f"{FASTAPI_URL}/list-models/")
        response.raise_for_status()
        data = response.json()
        models = data.get("models", [])
    except requests.exceptions.RequestException as e:
        flash(f"Error fetching models: {str(e)}", "danger")

    if request.method == 'POST':
        image = request.files.get('image')
        selected_model = request.form.get('model')

        if not image or not selected_model:
            flash("Please upload an image and select a model.", "danger")
            return render_template("predict.html", models=models)

        try:
            # Save the uploaded image inside app/static/uploads
            upload_folder = os.path.join("app", "static", "uploads")
            os.makedirs(upload_folder, exist_ok=True)  # Create the folder if it doesn't exist
            image_path = os.path.join(upload_folder, image.filename)
            image.save(image_path)  # Save the image locally

            # Send the image and selected model to FastAPI
            files = {'image': (image.filename, image.stream, image.mimetype)}
            form_data = {'model_name': selected_model}
            response = requests.post(
                f"{FASTAPI_URL}/predict/",
                files=files,
                data=form_data,
            )
            response.raise_for_status()
            data = response.json()

            # Include the uploaded image path in the prediction result
            prediction_result = {
                "label": data["label"],
                "confidence": data["confidence"],
                "uploaded_image": f"uploads/{image.filename}",  # Path relative to app/static
            }

        except requests.exceptions.RequestException as e:
            flash(f"Error during prediction: {str(e)}", "danger")

    return render_template("predict.html", models=models, prediction_result=prediction_result)
"""

@interface.route('/model')
def model():
    return render_template('model.html')


@interface.route('/train-model', methods=['GET', 'POST'])
def train_model():
    model_info = None
    plot_url = None
    confusion_matrix_url = None

    if request.method == 'POST':
        zip_file = request.files.get('zip_file')

        if not zip_file:
            flash("Please upload a zip file.", "danger")
            return redirect(url_for("interface.train_model"))

        try:
            # Send the zip file to the FastAPI backend
            files = {'zip_file': (zip_file.filename, zip_file.stream, zip_file.mimetype)}
            response = requests.post(f"{FASTAPI_URL}/train-model/", files=files)
            response.raise_for_status()

            # Parse response from FastAPI
            data = response.json()
            flash("Model trained successfully!", "success")

            # Store the FastAPI response for rendering
            model_info = {
                "message": data.get("message"),
                "validation_accuracy": f"{data.get('validation_accuracy', 0.0):.2f}",
                "validation_loss": f"{data.get('validation_loss', 0.0):.2f}",
                "model_path": data.get("saved_model"),
            }

            # URLs for plots
            plot_url = f"{FASTAPI_URL}/get-training-plot/"
            confusion_matrix_url = f"{FASTAPI_URL}/get-confusion-matrix/"

        except requests.exceptions.RequestException as e:
            flash(f"Error during training: {str(e)}", "danger")

        return render_template("train.html", model_info=model_info, plot_url=plot_url, confusion_matrix_url=confusion_matrix_url)

    return render_template("train.html", model_info=model_info, plot_url=plot_url, confusion_matrix_url=confusion_matrix_url)




@interface.route("/view-training-plot/")
def view_training_plot():
    # Redirect to the FastAPI endpoint
    return redirect(f"{FASTAPI_URL}/get-training-plot/")

@interface.route("/view-confusion-matrix/")
def view_confusion_matrix():
    # Redirect to the FastAPI endpoint
    return redirect(f"{FASTAPI_URL}/get-confusion-matrix/")