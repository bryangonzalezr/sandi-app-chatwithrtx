from fastapi import APIRouter
from app.aux_fun.translator import translate_text_from_to

router = APIRouter()


@router.get("/en_to_es/{text}")
def translate_text_esp(text: str):
    # Add your translation logic here
    translated_text = translate_text_from_to(text, source_lang="EN", target_lang="ES")
    return {"original_text": text, "translated_text": translated_text}

@router.get("/es_to_en/{text}")
def translate_text_eng(text: str):
    # Add your translation logic here
    translated_text = translate_text_from_to(text, source_lang="ES", target_lang="EN-US")
    return {"original_text": text, "translated_text": translated_text}
