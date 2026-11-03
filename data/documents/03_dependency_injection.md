# Dependency Injection in FastAPI

FastAPI features a very powerful and intuitive Dependency Injection system. It allows developers to declare dependencies needed by path operations, plugins, database sessions, and authentication guards.

## Why Use Dependency Injection?
- **Code reuse**: Share common logic (e.g., database connection, user authentication) across multiple routes.
- **Decoupled architecture**: Easily swap dependencies or mock them during unit and integration testing.
- **Automatic hierarchy**: Dependencies can themselves declare sub-dependencies, forming an execution graph.

## Basic Syntax with `Depends`
To declare a dependency, import `Depends` from `fastapi` and pass a callable function:

```python
from fastapi import Depends, FastAPI

app = FastAPI()

def common_pagination_params(skip: int = 0, limit: int = 100):
    return {"skip": skip, "limit": limit}

@app.get("/items/")
def read_items(pagination: dict = Depends(common_pagination_params)):
    return pagination

@app.get("/users/")
def read_users(pagination: dict = Depends(common_pagination_params)):
    return pagination
```

## Database Session Example
A standard pattern for managing database connections with `Depends` and `yield`:

```python
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/users/{user_id}")
def get_user(user_id: int, db: Session = Depends(get_db)):
    return db.query(User).filter(User.id == user_id).first()
```
The code before the `yield` runs before the request, and the code after `yield` runs after the response is sent, guaranteeing resource cleanup.

