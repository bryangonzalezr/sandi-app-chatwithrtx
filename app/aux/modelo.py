from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from difflib import get_close_matches
import json
from dotenv import load_dotenv
import os
from app.aux.rtx_fun import send_message
from app.aux.translator import translate_text_from_to

load_dotenv()

datos = os.getenv("DATA_CUOSINE")
with open(datos, "r") as file:
    data = json.load(file)

ingredientes = data["ingredientes"]
recetas = data["recetas"]
model_path = os.getenv("MODEL_PACIENTE")
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Asegúrate de que el tokenizador tenga un token de relleno
tokenizer.pad_token = tokenizer.eos_token

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
        #traducir a español
        translated_response = translate_text_from_to(response, source_lang="EN-US", target_lang="ES")
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