FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY vmatch ./vmatch
COPY data ./data
COPY scripts ./scripts
COPY model ./model
# downloads the model only if ./model is empty
RUN python scripts/get_model.py
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "vmatch.api:app", "--host", "0.0.0.0", "--port", "8000"]
