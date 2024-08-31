from fastapi import APIRouter
from app.aux.translator import translate_text_from_to
from fastapi.responses import StreamingResponse
from app.aux.rtx_fun import send_message
import json

router = APIRouter()

@router.post("/chatwithrtx")
def recibir_respuesta(pregunta: str):

    def event_generator():

        translated_question = translate_text_from_to(pregunta, source_lang="ES", target_lang="EN-US")
        response = send_message(translated_question)

        if response is None or 'error' in response:
            response = {'type': ""}

        else:
            response = {'query': response}

            translated_response = translate_text_from_to(response.get('query', ''), source_lang="EN-US", target_lang="ES")
            response['query'] = translated_response

        yield json.dumps(response)

    return StreamingResponse(event_generator(), media_type="text/event-stream")