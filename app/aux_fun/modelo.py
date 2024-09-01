from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from difflib import get_close_matches
import json
from dotenv import load_dotenv
import os
from app.aux_fun.rtx_fun import send_message
from app.aux_fun.translator import translate_text_from_to
import gdown
import zipfile


load_dotenv()

datos = "app/aux_fun/data/cocina.json"
with open(datos, "r") as file:
    data = json.load(file)

ingredientes = data["ingredientes"]
recetas = data["recetas"]
model_path = "app/aux_fun/data/model/checkpoint-830"

def model_check_existence():
    route = 'app/aux_fun/data/model/checkpoint-830'
    if os.path.exists(route):
        return True
    else:
        return False
    
def load_model():

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if model_path is None:
        raise ValueError("No se ha definido la variable de entorno MODEL_PACIENTE")
    if model_check_existence() is False:
        # Descargar el modelo de google drive
        google_drive_url = "https://drive.google.com/uc?id=1ZuvEGZrzvU9uO3cFWsqNWM-AVWj9BQ42"
        #download model
        gdown.download(google_drive_url, 'app/aux_fun/data/model/model.zip', quiet=False)
        #unzip downloaded file
        with zipfile.ZipFile('app/aux_fun/data/model/model.zip', 'r') as zip_ref:
            zip_ref.extractall('app/aux_fun/data/model')

    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Asegúrate de que el tokenizador tenga un token de relleno
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model

tokenizer, model = load_model()

def identificar_ingredientes(prompt):
    palabras = prompt.lower().split()
    ingredientes_encontrados = set()
    
    for palabra in palabras:
        # Buscamos coincidencias exactas o cercanas con una diferencia de una letra
        coincidencias = get_close_matches(palabra, ingredientes, n=1, cutoff=0.8)
        if coincidencias:
            ingredientes_encontrados.add(coincidencias[0])
    
    return list(ingredientes_encontrados)

def identificar_recetas(prompt):
    palabras = prompt.lower().split()
    recetas_encontradas = set()
    
    for palabra in palabras:
        # Buscamos coincidencias exactas o cercanas con una diferencia de una letra
        coincidencias = get_close_matches(palabra, recetas, n=1, cutoff=0.8)
        if coincidencias:
            recetas_encontradas.add(coincidencias[0])
    
    return list(recetas_encontradas)



def clasificador_pregunta(prompt: str):

    # 3. Tokenizar el prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128)
    # 4. Hacer una predicción
    with torch.no_grad():
        outputs = model(**inputs)
    # 5. Interpretar los resultados
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    # Mapeo inverso de etiquetas
    label_map_inverse = {
        0: "solicitud_receta",
        1: "solicitud_menu",
        2: "ninguna",
        3: "pregunta_cocina"
    }
    predicted_label = label_map_inverse[predicted_class]

    ingredientes_detectados = identificar_ingredientes(prompt)
    recetas_detectadas = identificar_recetas(prompt)

    if predicted_label == "solicitud_receta":
        #conexion con api juandex
        result = {
        "query": recetas_detectadas,
        "ingredientes": ingredientes_detectados,
        "type": predicted_label,
    }
    elif predicted_label == "solicitud_menu":
        #conexion api juandex
        result = {
        "query": recetas_detectadas,
        "ingredientes": ingredientes_detectados,
        "type": predicted_label,
        "time": 1,
    }
    elif predicted_label == "pregunta_cocina":
        
        #traducir a ingles
        translated_prompt = translate_text_from_to(prompt, source_lang="ES", target_lang="EN-US")
        #chatrtx
        response = send_message(translated_prompt)
        print(response)
        if(response == None):
            return {
                "query": prompt,
                "response": "Hubo un error al intentar comunicarse con el servidor de RTX",
                "type": predicted_label,
                    }
        translated_response = translate_text_from_to(response, source_lang="EN", target_lang="ES")
        #responder
        result = {
        "query": prompt,
        "response": translated_response,
        "type": predicted_label,
    }
    else:
        result = {
        "type": predicted_label,
    }
    # Convertir a JSON
    result_json = json.dumps(result, indent=4)
    return result_json

