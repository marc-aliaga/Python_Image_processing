FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000
# timeout mayor que ALGORITHM_TIMEOUT para que gunicorn no corte la petición
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "180", "app:app"]
