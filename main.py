from typing import Union
from fastapi.responses import StreamingResponse
from fastapi import FastAPI
from pydantic import BaseModel
import json
import rtx_api as rtx_api

app = FastAPI()

@app.post("/chatwithrtx")
def recibir_respuesta(pregunta: str):
    def event_generator():
        response = rtx_api.send_message(pregunta)
        return json.dumps(response)

    return StreamingResponse(event_generator(), media_type="text/event-stream")