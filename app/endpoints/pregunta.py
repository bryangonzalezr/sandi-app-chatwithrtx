from fastapi import APIRouter, HTTPException
from app.aux.modelo import clasificador_pregunta
import json
from pydantic import BaseModel

router = APIRouter()

class PreguntaUsuario(BaseModel):
    pregunta: str

@router.post("/pregunta_usuario")
def pregunta_usuario(
    pregunta: PreguntaUsuario
):
    try:
        respuesta = clasificador_pregunta(pregunta.pregunta)
        return json.loads(respuesta)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))