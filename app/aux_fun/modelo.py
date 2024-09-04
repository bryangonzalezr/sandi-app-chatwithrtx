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
import re
import requests


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

def extraer_tiempo(prompt):
    # Diccionario para convertir palabras a días
    periodos = {
        "semanal": 7,
        "semana": 7,
        "quincena": 15,
        "quincenal": 15,
        "mensual": 30,
        "mes": 30,
        "hoy": 1,
        "diario": 1,
        "día": 1,
        "dias": 1,  # Considerar faltas ortográficas
        "dia": 1    # Considerar faltas ortográficas
    }

    # Buscar patrones numéricos como "5 días" o "10 dias"
    match = re.search(r'(\d+)\s*(días|dias|dia)', prompt)
    if match:
        return int(match.group(1))

    # Buscar palabras que representan periodos
    for palabra, dias in periodos.items():
        if palabra in prompt.lower():
            return dias

    # Si no se encuentra nada, retornar un valor predeterminado, por ejemplo, 1 día
    return 1

def obtener_token(email, password):
    url = "http://localhost:8080/api/login"
    payload = {
        "email": email,
        "password": password
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Levanta una excepción para códigos de respuesta no exitosos
        data = response.json()
        return data['data']['token']
    except requests.exceptions.RequestException as e:
        print(f"Error al obtener el token: {e}")
        return None
    
def obtener_receta(token, query):
    url = "http://localhost:8080/api/receta/generar"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    body = {
        "query": query
    }

    try:
        response = requests.post(url, headers=headers, json=body)
        response.raise_for_status()  # Levanta una excepción para códigos de respuesta no exitosos
        data = response.json()
        
        # Extraer la información requerida
        recipe = data.get("recipe", {})
        label = recipe.get("label", "")
        ingredient_lines = recipe.get("ingredientLines", [])
        calories = recipe.get("calories", 0)
        ingredientes = ", ".join(ingredient_lines)
        instrucciones = send_message(f"Give me instructions for {label} using the following ingredients: {ingredientes}")

        # Traducir los campos
        label_translated = translate_text_from_to(label, source_lang="EN", target_lang="ES")
        ingredient_lines_translated = [translate_text_from_to(line, source_lang="EN", target_lang="ES") for line in ingredient_lines]
        instrucciones_traducidas = translate_text_from_to(instrucciones, source_lang="EN", target_lang="ES")
        
        # Crear el JSON final con las traducciones
        translated_result = {
            "receta": label_translated,
            "ingredientes": ingredient_lines_translated,
            "calorias": calories,
            "instrucciones": instrucciones_traducidas
        }
        
        return translated_result
    
    except requests.exceptions.RequestException as e:
        print(f"Error al obtener la receta: {e}")
        return None
    
def generar_menu(token, query, time):
    # Verificamos si el valor de time es 1 para consumir el endpoint
    if time == 1:
        url = "http://localhost:8080/api/menu-diario/generar"
        body = {
            "query": query
        }
    elif time > 1:
        url = "http://localhost:8080/api/menu/generar"
        body = {
            "query": query,
            "timespan": time
        }
    else:
        print("Inserta un valor válido")
        return None
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    try:
        # Hacer la solicitud al endpoint
        response = requests.post(url, headers=headers, json=body)
        response.raise_for_status()  # Lanzar una excepción si la respuesta tiene un error
        data = response.json()

        # Procesar los datos para extraer la información relevante
        if time == 1:
            recipes = data["recipes"]
        else:
            recipes = data["menus"]
        total_calories = data["total_calories"]

        recetas_traducidas = []

        for receta in recipes:
            # ingredientes = ", ".join(receta["ingredientLines"])
            # instrucciones = send_message(f"Give me instructions for {receta['label']} using the following ingredients: {ingredientes}")
            # print(f"Give me instructions for {receta['label']} using the following ingredients: {ingredientes}")

            # Traducir label, ingredientLines y mealType
            label_traducido = translate_text_from_to(receta["label"], source_lang="EN", target_lang="ES")
            ingredient_lines_traducido = [translate_text_from_to(ingredient, source_lang="EN", target_lang="ES") for ingredient in receta["ingredientLines"]]
            meal_type_traducido = [translate_text_from_to(meal, source_lang="EN", target_lang="ES") for meal in receta["mealType"]]
            # instrucciones_traducidas = translate_text_from_to(instrucciones, source_lang="EN", target_lang="ES")
            
            # Agregar la receta traducida con calorías sin traducir
            receta_traducida = {
                "receta": label_traducido,
                "ingredientes": ingredient_lines_traducido,
                "calorias": receta["calories"], 
                "tipo": meal_type_traducido
                # "instrucciones": instrucciones_traducidas
            }
            
            recetas_traducidas.append(receta_traducida)

        # Resultado final con total_calories sin traducir
        resultado = {
            "recipes": recetas_traducidas,
            "total_calories": total_calories 
        }

        return json.dumps(resultado, indent=4)

    except requests.exceptions.RequestException as e:
        print(f"Error al hacer la solicitud: {e}")
        return None

def identificar_ingredientes(prompt):
    palabras = prompt.lower().split()
    ingredientes_encontrados = set()
    
    for palabra in palabras:
        # Buscamos coincidencias exactas o cercanas con una diferencia de una letra
        coincidencias = get_close_matches(palabra, ingredientes, n=1, cutoff=0.8)
        if coincidencias:
            ingredientes_encontrados.add(coincidencias[0])
    return ", ".join(ingredientes_encontrados)

def identificar_recetas(prompt):
    palabras = prompt.lower().split()
    recetas_encontradas = set()
    
    for palabra in palabras:
        # Buscamos coincidencias exactas o cercanas con una diferencia de una letra
        coincidencias = get_close_matches(palabra, recetas, n=1, cutoff=0.8)
        if coincidencias:
            recetas_encontradas.add(coincidencias[0])
    return ", ".join(recetas_encontradas)

def texto_query(ingredientes, recetas):
    if ingredientes == "" and recetas == "":
        return "Debes ingresar ingredientes y/o recetas"
    elif ingredientes == "" and recetas != "":
        return recetas
    elif ingredientes != "" and recetas == "":
        return ingredientes
    elif ingredientes != "" and recetas != "":
        return f"{recetas} con {ingredientes}"

def clasificador_pregunta(prompt: str):
    token = obtener_token("sandi@test.cl", "sandi.,2024")
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

    if predicted_label == "solicitud_receta" or predicted_label == "solicitud_menu":
        ingredientes_detectados = identificar_ingredientes(prompt)
        recetas_detectadas = identificar_recetas(prompt)
        resultado = texto_query(ingredientes_detectados, recetas_detectadas)
        resultado = translate_text_from_to(resultado, source_lang="ES", target_lang="EN-US")

    if predicted_label == "solicitud_receta":
        query = obtener_receta(token, resultado)
        result = {
        "receta": query['receta'],
        "ingredientes": query['ingredientes'],
        "calorias": query['calorias'],
        "instrucciones": query['instrucciones'],
        "type": predicted_label
    }
    elif predicted_label == "solicitud_menu":
        # Aquí usamos la función extraer_tiempo
        dias = extraer_tiempo(prompt)
        query = generar_menu(token, resultado, dias)
        query = json.loads(query)
        result = {
            "recetas": query['recipes'],
            "total_calorias": query['total_calories'],
            # "instrucciones": query['instrucciones'],
            "type": predicted_label,
            "time": dias
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