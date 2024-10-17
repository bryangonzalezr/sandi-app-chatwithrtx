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

api_url = os.getenv("SANDI_API_HOST")
api_port = os.getenv("SANDI_API_PORT")

datos = "app/aux_fun/data/cocina.json"
with open(datos, "r") as file:
    data = json.load(file)

ingredientes = data["ingredientes"]
recetas = data["recetas"]
model_path = os.getenv("MODEL_PATH")

def model_check_existence():
    route = model_path
    if os.path.exists(route):
        return True
    else:
        return False
    
def load_model():

    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    if model_path is None:
        raise ValueError("No se ha definido la variable de entorno MODEL_PACIENTE")

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
    url = "http://"+api_url+":"+api_port+"/api/login"
    payload = {
        "email": email,
        "password": password
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Levanta una excepción para códigos de respuesta no exitosos
        data = response.json()
        return data['token']
    except requests.exceptions.RequestException as e:
        print(f"Error al obtener el token: {e}")
        return None
    
def obtener_receta(token, query, id_usuario):
    url = "http://"+api_url+":"+api_port+"/api/receta/generar"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    body = {
        "query": query,
        "user_id": id_usuario
    }

    try:
        response = requests.post(url, headers=headers, json=body)
        response.raise_for_status()  # Levanta una excepción para códigos de respuesta no exitosos
        data = response.json()
        
        # Extraer la información requerida
        recipe = data.get("recipe", {})
        label = recipe.get("label", "")
        dietLabels = recipe.get("dietLabels", [])
        healthLabels = recipe.get("healthLabels", [])
        cautions = recipe.get("cautions", [])
        ingredient_lines = recipe.get("ingredientLines", [])
        calories = recipe.get("calories", 0)
        totalTime = recipe.get("totalTime", 0)
        mealType = recipe.get("mealType", [])
        dishType = recipe.get("dishType", [])
        ingredientes = ", ".join(ingredient_lines)
        instrucciones = send_message(f"Give me instructions for {label} using the following ingredients: {ingredientes}")

        # Traducir los campos
        label_translated = translate_text_from_to(label, source_lang="EN", target_lang="ES")
        diet_labels_translated = [translate_text_from_to(label, source_lang="EN", target_lang="ES") for label in dietLabels]
        health_labels_translated = [translate_text_from_to(label, source_lang="EN", target_lang="ES") for label in healthLabels]
        cautions_translated = [translate_text_from_to(caution, source_lang="EN", target_lang="ES") for caution in cautions]
        ingredient_lines_translated = [translate_text_from_to(line, source_lang="EN", target_lang="ES") for line in ingredient_lines]
        meal_type_translated = [translate_text_from_to(type, source_lang="EN", target_lang="ES") for type in mealType]
        dish_type_translated = [translate_text_from_to(type, source_lang="EN", target_lang="ES") for type in dishType]
        instrucciones_traducidas = translate_text_from_to(instrucciones, source_lang="EN", target_lang="ES")
        
        # Crear el JSON final con las traducciones
        translated_result = {
            "receta": label_translated,
            "dietas": diet_labels_translated,
            "salud": health_labels_translated,
            "precauciones":  cautions_translated,
            "ingredientes": ingredient_lines_translated,
            "calorias": calories,
            "tiempoTotal": totalTime,
            "tipo_comida": meal_type_translated,
            "tipo_plato": dish_type_translated,
            "instrucciones": instrucciones_traducidas
        }
        
        return translated_result
    
    except requests.exceptions.RequestException as e:
        print(f"Error al obtener la receta: {e}")
        return None
    
def generar_menu(token, query, time, id_usuario):
    # Verificamos si el valor de time es 1 para consumir el endpoint
    if time == 1:
        url = "http://"+api_url+":"+api_port+"/api/menu-diario/generar"
        body = {
            "query": query,
            "user_id": id_usuario
        }
    elif time > 1:
        url = "http://"+api_url+":"+api_port+"/api/menu/generar"
        body = {
            "query": query,
            "time": str(time),
            "user_id": id_usuario
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
            ingredientes = ", ".join(receta["ingredientLines"])
            instrucciones = send_message(f"Give me instructions for {receta['label']} using the following ingredients: {ingredientes}")
            # print(f"Give me instructions for {receta['label']} using the following ingredients: {ingredientes}")

            # Traducir label, ingredientLines y mealType
            label_traducido = translate_text_from_to(receta["label"], source_lang="EN", target_lang="ES")
            diet_labels_translated = [translate_text_from_to(label, source_lang="EN", target_lang="ES") for label in receta["dietLabels"]]
            health_labels_translated = [translate_text_from_to(label, source_lang="EN", target_lang="ES") for label in receta["healthLabels"]]
            cautions_translated = [translate_text_from_to(caution, source_lang="EN", target_lang="ES") for caution in receta["cautions"]]
            ingredient_lines_traducido = [translate_text_from_to(ingredient, source_lang="EN", target_lang="ES") for ingredient in receta["ingredientLines"]]
            meal_type_translated = [translate_text_from_to(meal, source_lang="EN", target_lang="ES") for meal in receta["mealType"]]
            dish_type_translated = [translate_text_from_to(type, source_lang="EN", target_lang="ES") for type in receta["dishType"]]
            instrucciones_traducidas = translate_text_from_to(instrucciones, source_lang="EN", target_lang="ES")
            
            # Agregar la receta traducida con calorías sin traducir
            receta_traducida = {
                "label": label_traducido,
                "dietLabels": diet_labels_translated,
                "healthLabels": health_labels_translated,
                "cautions":  cautions_translated,
                "ingredientLines": ingredient_lines_traducido,
                "calories": receta["calories"],
                "totalTime": receta["totalTime"],
                "mealType": meal_type_translated,
                "dishType": dish_type_translated, 
                "instructions": instrucciones_traducidas
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

def clasificador_pregunta(prompt: str, id_usuario: int):
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
        query = obtener_receta(token, resultado, id_usuario= id_usuario)
        result = {
        "label": query['receta'],
        "dietLabels": query['dietas'],
        "healthLabels": query["salud"],
        "cautions": query["precauciones"],
        "ingredientLines": query['ingredientes'],
        "calories": query['calorias'],
        "totalTime": query["tiempoTotal"],
        "mealType": query["tipo_comida"],
        "dishType": query["tipo_plato"],
        "instructions": query['instrucciones'],
        "type": predicted_label,
        "user_id": id_usuario
    }
    elif predicted_label == "solicitud_menu":
        # Aquí usamos la función extraer_tiempo
        dias = extraer_tiempo(prompt)
        query = generar_menu(token, resultado, dias, id_usuario)
        query = json.loads(query)
        result = {
            "recipes": query['recipes'],
            "total_calories": query['total_calories'],
            # "instrucciones": query['instrucciones'],
            "type": predicted_label,
            "time": dias,
            "user_id": id_usuario
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
                "id_usuario": id_usuario
                    }
        translated_response = translate_text_from_to(response, source_lang="EN", target_lang="ES")
        #responder
        result = {
        "query": prompt,
        "response": translated_response,
        "type": predicted_label,
        "id_usuario": id_usuario
    }
    else:
        result = {
        "type": predicted_label,
    }
    # Convertir a JSON
    result_json = json.dumps(result, indent=4)
    return result_json