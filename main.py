from typing import Union
from fastapi.responses import StreamingResponse
from fastapi import FastAPI
from pydantic import BaseModel
import json
import rtx_api as rtx_api

app = FastAPI()


class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}

@app.get("/libros/{id}")
def mostrar_libro(id: int):
    return {"data": id}

@app.post("/chatwithrtx")
def recibir_respuesta(pregunta: str):
    def event_generator():
        for response in rtx_api.send_message(pregunta):
            yield f"data: {json.dumps(response)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")