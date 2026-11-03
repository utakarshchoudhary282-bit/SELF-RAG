# Deploying FastAPI with Uvicorn

FastAPI is an ASGI (Asynchronous Server Gateway Interface) framework. To run it, you need an ASGI web server such as Uvicorn or Hypercorn.

## Running in Development
To run FastAPI locally in development with auto-reload:

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
- `main`: refers to the Python file `main.py`.
- `app`: refers to the object `app = FastAPI()` created inside `main.py`.
- `--reload`: automatically restarts the server whenever code changes are saved.

## Production Deployment with Gunicorn / Multiple Workers
For production deployments, Uvicorn can be managed with multiple worker processes to handle high traffic and utilize multiple CPU cores:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Running Inside Docker
A standard Dockerfile for deploying FastAPI:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```
This container listens on port 8000 and serves all incoming API requests reliably.

