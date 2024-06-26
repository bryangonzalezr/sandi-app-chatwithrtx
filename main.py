from typing import Union
from fastapi.responses import StreamingResponse, Response
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import json
import rtx_api as rtx_api
# from models.userData import UserData
# from models.types import Types
import httpx


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chatwithrtx")
def recibir_respuesta(pregunta: str):
    
    def event_generator():

        response = rtx_api.send_message(pregunta)
        response = dict(response)

        if response['type'] == "General Query":
            yield json.dumps({"response": response['response']})

        elif response['type'] == 'Recipe':
            data = json.dumps(response)
            r = httpx.post("http://localhost:8080/receta/api/", json=response)
            r.raise_for_status()
            yield json.dumps(r.json())
        
        else:
            yield json.dumps({"error": "Invalid query type"})

    return StreamingResponse(event_generator(), media_type="text/event-stream")