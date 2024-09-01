from fastapi import APIRouter, HTTPException
from app.aux_fun.modelo import clasificador_pregunta
import json
from pydantic import BaseModel
from app.config import Config



router = APIRouter()

class PreguntaUsuario(BaseModel):
    pregunta: str

@router.post("/pregunta_usuario")
def pregunta_usuario(
    pregunta: PreguntaUsuario
):
    try:
        print(pregunta)
        respuesta = clasificador_pregunta(pregunta.pregunta)
        return json.loads(respuesta)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.put("/ngrok_url")
def ngrok_url(
    url: str
):
    Config.update_ngrok_url(url)
    return {"url": url}

@router.get("/ngrok_url")
def get_ngrok_url():
    return {"url": Config.get_ngrok_url()}