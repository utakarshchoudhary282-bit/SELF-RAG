# Middleware and CORS in FastAPI

Middleware is a function that runs before every request is processed by a specific path operation, and after every response is generated.

## What is CORS?
CORS (Cross-Origin Resource Sharing) is a security mechanism enforced by web browsers. It restricts web pages running on one domain/origin (e.g. `http://localhost:3000`) from making HTTP requests to a different domain/origin (e.g. `http://localhost:8000`).

## Enabling CORS in FastAPI
FastAPI provides built-in `CORSMiddleware` to configure cross-origin access easily:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Error Handling with HTTPException
To return HTTP error responses with status codes to the client, raise `HTTPException`:

```python
from fastapi import FastAPI, HTTPException, status

app = FastAPI()

items = {"foo": "The Foo item"}

@app.get("/items/{item_id}")
def read_item(item_id: str):
    if item_id not in items:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Item not found in database"
        )
    return {"item": items[item_id]}
```
When an `HTTPException` is raised, FastAPI automatically stops execution and returns a JSON response containing the status code and detail message.

