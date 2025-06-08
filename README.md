# Early Detection of Gastric Cancer using Deep Learning

## Overview
The Early Gastric Cancer Detection System leverages a machine learning pipeline to assist healthcare professionals in predicting whether endoscopic images indicate the presence of cancerous lesions. This project includes a web-based FastAPI backend for prediction and model training and is built with a focus on scalability, accuracy, and user accessibility.

## Features
Image Classification: Predicts if an uploaded endoscopic image indicates gastric cancer.
Model Training: Enables users to train or retrain models using custom datasets.
Interactive Visualizations: Displays model performance metrics such as training loss, accuracy, and confusion matrices.
REST API: Supports endpoint-based interaction for seamless integration with external services.


## Installation Instructions

### Prerequisites

- Python 3.10 or higher
- Virtual environment setup (venv or conda)
- Docker (optional for deployment)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Mwangi-dan/early-gastric-cancer-detection_MLOP_Summative.git
cd early-gastric-cancer-detection_MLOP_Summative
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate 
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## How to Use

### 1. Start the FastAPI Server

Run the following command in the project directory:

```bash
uvicorn fast:app --reload
```

The server will start on [http://127.0.0.1:8000](http://127.0.0.1:8000).

### 2. Endpoints Overview

- **Prediction**: `/predict/`
    - Upload an endoscopic image and select a trained model to predict its class.
- **Model Training**: `/train-model/`
    - Upload a zipped folder with labeled images (cancerous, non-cancerous) to train a new model.
- **Model Listing**: `/list-models/`
    - Fetch all available models for prediction.
