# Edge IoMT Health Status Prediction API

A FastAPI-based machine learning microservice for predicting patient health status using real-time IoMT sensor data.

## Features

- 🚀 FastAPI REST API
- 🧠 Trained RandomForest model
- ⚖️ SMOTE-balanced classification
- 🐳 Dockerized for edge/cloud deployment

## Usage

### 🔧 Run Locally
```bash
uvicorn app:app --reload
```

### 🐳 Run with Docker
```bash
docker build -t edge-iomt-api .
docker run -p 8000:8000 edge-iomt-api
```

Access Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
