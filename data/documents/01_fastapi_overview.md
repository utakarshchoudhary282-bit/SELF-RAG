# FastAPI Overview and Core Concepts

FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.8+ based on standard Python type hints.

## Key Features
- **Fast**: Very high performance, on par with NodeJS and Go (thanks to Starlette and Pydantic). One of the fastest Python frameworks available.
- **Fast to code**: Increases the speed to develop features by about 200% to 300%.
- **Fewer bugs**: Reduces about 40% of developer-induced errors.
- **Intuitive**: Great editor support with autocompletion everywhere and less time debugging.
- **Easy**: Designed to be easy to use and learn. Less time reading docs.
- **Short**: Minimize code duplication. Multiple features from each parameter declaration.
- **Standards-based**: Based on and fully compatible with the open standards for APIs: OpenAPI (previously known as Swagger) and JSON Schema.

## Creating a Basic Application
To create a basic FastAPI app, instantiate the `FastAPI` class:

```python
from fastapi import FastAPI

app = FastAPI(title="Documentation API", version="1.0.0")

@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI"}
```

## Path Parameters and Type Hints
You can declare path parameters with standard Python type hints:

```python
@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}
```
FastAPI automatically performs data parsing, validation, and produces automatic Swagger documentation.

## Query Parameters
When you declare other function parameters that are not part of the path, they are automatically interpreted as "query" parameters:

```python
@app.get("/items/")
def read_items(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}
```
Query parameters can have default values or be optional using `Optional[str] = None`.

