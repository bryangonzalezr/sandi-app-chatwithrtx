import deepl
from dotenv import load_dotenv
import os

load_dotenv()

auth_key = os.getenv("DEEPL_AUTH_KEY")
translator = deepl.Translator(auth_key)

def translate_text_from_to(text: str, source_lang: str, target_lang: str):
    translated_text = translator.translate_text(text, source_lang=source_lang, target_lang=target_lang).text
    return translated_text