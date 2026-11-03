# Request Body and Pydantic Data Validation

In FastAPI, when you need to send data from a client to your API, you send it as a request body. To declare a request body, you use Pydantic models with all their power and benefits.

## What is Pydantic?
Pydantic is a data validation library that uses Python type annotations. It enforces type hints at runtime and provides user-friendly error messages when data is invalid.

## Defining a Pydantic Model
To declare a request body, import `BaseModel` from `pydantic` and create your data schema class:

```python
from typing import Optional
from pydantic import BaseModel, Field

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float = Field(gt=0, description="The price must be greater than zero")
    tax: Optional[float] = None
```

## Using the Model in Path Operations
You declare the model as a parameter in your route function:

```python
from fastapi import FastAPI, status

app = FastAPI()

@app.post("/items/", status_code=status.HTTP_201_CREATED)
def create_item(item: Item):
    total = item.price + (item.tax or 0)
    return {"name": item.name, "total_price": total}
```

## Automatic Benefits
1. **Read request body as JSON**: Converts incoming JSON payload into your Python object.
2. **Type validation**: Validates types (e.g., ensuring `price` is a valid float). If validation fails, returns a standard `422 Unprocessable Entity` HTTP error with exact details.
3. **OpenAPI Schema generation**: Generates JSON Schema definitions for your models in the interactive Swagger documentation.

