FROM python:3.11-slim
WORKDIR /app

COPY model/requirements.txt ./model/
RUN pip install --no-cache-dir -r model/requirements.txt

COPY model/ ./model/
RUN mkdir -p /app/data

CMD ["python", "model/train.py"]