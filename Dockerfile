FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt ./
COPY app.py ./
COPY random_forest_model.pkl ./
COPY scaler.pkl ./
COPY feature_names.txt ./
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
