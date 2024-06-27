from typing import Union
from fastapi.responses import StreamingResponse, Response
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import json
import rtx_api as rtx_api
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
            yield json.dumps(response)

        elif response['type'] == 'Recipe':
            r = httpx.post("http://localhost:8080/receta/api/", json=response)
            r.raise_for_status()
            yield json.dumps(r.json())

        elif response['type'] == 'Daily Menu':
            r = httpx.post("http://localhost:8080/daymenu/generate/", json=response, follow_redirects=True)
            r.raise_for_status()
            yield json.dumps(r.json())

        elif response['type'] == 'General Menu':
            r = httpx.post("http://localhost:8080/menu/generate/", params={"timespan": response['timespan']}, json={"query": response['query']}, follow_redirects=True, timeout=60.0)
            r.raise_for_status()
            yield json.dumps(r.json())

        else:
            yield json.dumps({"error": "Invalid query type"})

    return StreamingResponse(event_generator(), media_type="text/event-stream")

""" def read_file_content(filepath: str) -> str:
    try:
        with open(filepath, 'r') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return "The file was not found."
    except IOError:
        return "An I/O error occurred."

file_content = ""

@app.on_event("startup")
def startup_event():
    global file_content
    file_content = read_file_content('instructions_recipes.txt')


@app.get("/file-content/")
def get_file_content():
    return {"file_content": file_content} """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)