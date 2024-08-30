from typing import Union
from fastapi.responses import StreamingResponse, Response
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import json
import rtx_api as rtx_api
from models.userData import UserData
from models.types import Types
import httpx
from fastapi import FastAPI
import deepl

app = FastAPI()

#translator
auth_key = "a7578d8e-684a-49ae-8df7-986236244665:fx"
translator = deepl.Translator(auth_key)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.get("/translate_esp/{text}")
def translate_text_esp(text: str):
    # Add your translation logic here
    translated_text = translator.translate_text(text, source_lang="EN", target_lang="ES")
    return {"original_text": text, "translated_text": translated_text}

@app.get("/translate_eng/{text}")
def translate_text_eng(text: str):
    # Add your translation logic here
    translated_text = translator.translate_text(text, source_lang="ES", target_lang="EN-US")
    return {"original_text": text, "translated_text": translated_text}

@app.post("/chatwithrtx")
def recibir_respuesta(pregunta: str):

    def event_generator():

        translated_question = translator.translate_text(pregunta, source_lang="ES", target_lang="EN-US").text
        response = rtx_api.send_message(translated_question)

        if response is None or 'error' in response:
            response = {'type': ""}

        else:
            response = {'query': response}

            translated_response = translator.translate_text(response.get('query', ''), source_lang="EN", target_lang="ES").text
            response['query'] = translated_response

        yield json.dumps(response)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

def read_file_content(filepath: str) -> str:
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
    return {"file_content": file_content}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)