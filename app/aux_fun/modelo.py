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

# def obtener_token(email, password):
#     url = "http://"+api_url+":"+api_port+"/api/login"
#     payload = {
#         "email": email,
#         "password": password
#     }
#     try:
#         response = requests.post(url, json=payload)
#         response.raise_for_status()  # Levanta una excepción para códigos de respuesta no exitosos
#         data = response.json()
#         return data['token']
#     except requests.exceptions.RequestException as e:
#         print(f"Error al obtener el token: {e}")
#         return None
    
def obtener_receta(token, query, restrictions):
    url = "http://"+api_url+":"+api_port+"/api/receta/generar"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    if restrictions != "" and query != "":
        body = {
            "query": query,
            "excluded": restrictions
        }
    elif restrictions != "" and query == "":
        body = {
            "excluded": restrictions
        }
    else:
        body = {
            "query": query
        }
    print(body)

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
        
        # Extraer y traducir ingredientes con cantidad
        ingredients = [
            {
                "food": translate_text_from_to(ingredient.get("food"), source_lang="EN", target_lang="ES"),
                "quantity": float(ingredient.get("quantity")) if ingredient.get("quantity") is not None else None,
                "measure": "" if ingredient.get("measure") is None else translate_text_from_to(ingredient.get("measure"), source_lang="EN", target_lang="ES")
            }
            for ingredient in recipe.get("ingredients", [])
        ]
        print(ingredients)
        
        calories = recipe.get("calories", 0) 
        totalTime = recipe.get("totalTime", 0)
        mealType = recipe.get("mealType", [])
        dishType = recipe.get("dishType", [])
        ingredientes = ", ".join(ingredient_lines)
        instrucciones = send_message(f"Give me instructions for {label} using the following ingredients: {ingredientes}")
        print(instrucciones)

        # Traducir los campos restantes
        label_translated = translate_text_from_to(label, source_lang="EN", target_lang="ES")
        diet_labels_translated = [translate_text_from_to(label, source_lang="EN", target_lang="ES") for label in dietLabels]
        health_labels_translated = [translate_text_from_to(label, source_lang="EN", target_lang="ES") for label in healthLabels]
        cautions_translated = [translate_text_from_to(caution, source_lang="EN", target_lang="ES") for caution in cautions]
        ingredient_lines_translated = [translate_text_from_to(line, source_lang="EN", target_lang="ES") for line in ingredient_lines]
        meal_type_translated = [translate_text_from_to(type, source_lang="EN", target_lang="ES") for type in mealType]
        dish_type_translated = [translate_text_from_to(type, source_lang="EN", target_lang="ES") for type in dishType] if dishType else []
        instrucciones_traducidas = translate_text_from_to(instrucciones, source_lang="EN", target_lang="ES")
        
        # Crear el JSON final con las traducciones
        translated_result = {
            "receta": label_translated,
            "dietas": diet_labels_translated,
            "salud": health_labels_translated,
            "precauciones": cautions_translated,
            "ingredientes": ingredient_lines_translated,
            "nombre_ingredientes": ingredients,
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
    
def generar_menu(token, query, time, restrictions):
    if restrictions != "" and query != "":
        body = {
            "query": query,
            "excluded": restrictions,
            "timespan": str(time)
        }
    elif restrictions != "" and query == "":
        body = {
            "excluded": restrictions,
            "timespan": str(time)
        }
    else:
        body = {
            "query": query,
            "timespan": str(time)
        }

    # Verificamos si el valor de time es 1, 7 o 30 para definir el tipo de menú
    if time == 1:
        url = "http://"+api_url+":"+api_port+"/api/menu-diario/generar"
        menu_type = "diario"
        if restrictions != "" and query != "":
            body = {
                "query": query,
                "excluded": restrictions
            }
        elif restrictions != "" and query == "":
            body = {
                "excluded": restrictions
            }
        else:
            body = {
                "query": query
            }
    elif time == 7:
        url = "http://"+api_url+":"+api_port+"/api/menu/generar"
        menu_type = "semanal"
    elif time == 30:
        url = "http://"+api_url+":"+api_port+"/api/menu/generar"
        menu_type = "mensual"
    else:
        print("Inserta un valor válido para 'time' (1 para diario, 7 para semanal, 30 para mensual)")
        return None

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    print(body)

    try:
        # Hacer la solicitud al endpoint
        response = requests.post(url, headers=headers, json=body)
        response.raise_for_status()  # Lanzar una excepción si la respuesta tiene un error
        data = response.json()

        # Procesar los datos para extraer la información relevante
        if time == 1:
            # Cuando el tiempo es diario, data["recipes"] es una lista de recetas
            menus = data["recipes"]
        else:
            # Cuando el tiempo es semanal o mensual, data["menus"] es un diccionario
            menus = data["menus"]
        
        total_calories = data["total_calories"]
        menus_traducidos = {}

        if time == 1:
            # Para el menú diario, procesamos cada receta en la lista
            recetas_traducidas = []
            for receta in menus:
                ingredientes = ", ".join(receta["ingredientLines"])
                instrucciones = send_message(f"Give me instructions for {receta['label']} using the following ingredients: {ingredientes}")
                # Extraer y traducir ingredientes con cantidad
                ingredients = [
                    {
                        "food": translate_text_from_to(ingredient.get("food"), source_lang="EN", target_lang="ES"),
                        "quantity": float(ingredient.get("quantity")) if ingredient.get("quantity") is not None else None,
                        "measure": "" if ingredient.get("measure") is None else translate_text_from_to(ingredient.get("measure"), source_lang="EN", target_lang="ES")
                    }
                    for ingredient in receta.get("ingredients", [])
                ]

                # Traducir label, ingredientLines y mealType
                label_traducido = translate_text_from_to(receta["label"], source_lang="EN", target_lang="ES")
                diet_labels_translated = [translate_text_from_to(label, source_lang="EN", target_lang="ES") for label in receta["dietLabels"]]
                health_labels_translated = [translate_text_from_to(label, source_lang="EN", target_lang="ES") for label in receta["healthLabels"]]
                cautions_translated = [translate_text_from_to(caution, source_lang="EN", target_lang="ES") for caution in receta["cautions"]]
                ingredient_lines_traducido = [translate_text_from_to(ingredient, source_lang="EN", target_lang="ES") for ingredient in receta["ingredientLines"]]
                meal_type_translated = [translate_text_from_to(meal, source_lang="EN", target_lang="ES") for meal in receta["mealType"]]
                dish_type_translated = [translate_text_from_to(type, source_lang="EN", target_lang="ES") for type in receta["dishType"]] if receta["dishType"] else []
                instrucciones_traducidas = translate_text_from_to(instrucciones, source_lang="EN", target_lang="ES")

                # Agregar la receta traducida
                receta_traducida = {
                    "label": label_traducido,
                    "dietLabels": diet_labels_translated,
                    "healthLabels": health_labels_translated,
                    "cautions": cautions_translated,
                    "ingredientLines": ingredient_lines_traducido,
                    "ingredients": ingredients,
                    "calories": receta["calories"],
                    "totalTime": receta["totalTime"],
                    "mealType": meal_type_translated,
                    "dishType": dish_type_translated,
                    "instructions": instrucciones_traducidas
                }

                recetas_traducidas.append(receta_traducida)

            # Guardamos todas las recetas traducidas en menus_traducidos para el menú diario
            menus_traducidos = recetas_traducidas
        else:
            # Para menú semanal o mensual, iteramos sobre los días
            for dia, recipes in menus.items():
                recetas_traducidas = []
                for receta in recipes:
                    ingredientes = ", ".join(receta["ingredientLines"])
                    instrucciones = send_message(f"Give me instructions for {receta['label']} using the following ingredients: {ingredientes}")
                    # Extraer y traducir ingredientes con cantidad
                    ingredients = [
                        {
                            "food": translate_text_from_to(ingredient.get("food"), source_lang="EN", target_lang="ES"),
                            "quantity": float(ingredient.get("quantity")) if ingredient.get("quantity") is not None else None,
                            "measure": "" if ingredient.get("measure") is None else translate_text_from_to(ingredient.get("measure"), source_lang="EN", target_lang="ES")
                        }
                        for ingredient in receta.get("ingredients", [])
                    ]

                    # Traducir label, ingredientLines y mealType
                    label_traducido = translate_text_from_to(receta["label"], source_lang="EN", target_lang="ES")
                    diet_labels_translated = [translate_text_from_to(label, source_lang="EN", target_lang="ES") for label in receta["dietLabels"]]
                    health_labels_translated = [translate_text_from_to(label, source_lang="EN", target_lang="ES") for label in receta["healthLabels"]]
                    cautions_translated = [translate_text_from_to(caution, source_lang="EN", target_lang="ES") for caution in receta["cautions"]]
                    ingredient_lines_traducido = [translate_text_from_to(ingredient, source_lang="EN", target_lang="ES") for ingredient in receta["ingredientLines"]]
                    meal_type_translated = [translate_text_from_to(meal, source_lang="EN", target_lang="ES") for meal in receta["mealType"]]
                    dish_type_translated = [translate_text_from_to(type, source_lang="EN", target_lang="ES") for type in receta["dishType"]] if receta["dishType"] else []
                    instrucciones_traducidas = translate_text_from_to(instrucciones, source_lang="EN", target_lang="ES")

                    # Agregar la receta traducida
                    receta_traducida = {
                        "label": label_traducido,
                        "dietLabels": diet_labels_translated,
                        "healthLabels": health_labels_translated,
                        "cautions": cautions_translated,
                        "ingredientLines": ingredient_lines_traducido,
                        "ingredients": ingredients,
                        "calories": receta["calories"],
                        "totalTime": receta["totalTime"],
                        "mealType": meal_type_translated,
                        "dishType": dish_type_translated,
                        "instructions": instrucciones_traducidas
                    }

                    recetas_traducidas.append(receta_traducida)

                # Añadir las recetas traducidas bajo el día correspondiente
                menus_traducidos[dia] = recetas_traducidas

        # Resultado final con total_calories sin traducir y el tipo de menú
        resultado = {
            "menus": menus_traducidos,
            "type": menu_type,
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

def clasificador_pregunta(prompt: str, token: str):
    # token = obtener_token("sandi@test.cl", "sandi.,2024")
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

    restricciones_detectadas = ""
    if predicted_label == "solicitud_receta" or predicted_label == "solicitud_menu":
        if " sin " in prompt:
            partes = prompt.split(" sin ", 1)
            resto = partes[0].strip()
            restriccion = partes[1].strip()
            
            restricciones_detectadas = translate_text_from_to(identificar_ingredientes(restriccion), source_lang="ES", target_lang="EN-US")
            restricciones_detectadas = [ing.strip() for ing in restricciones_detectadas.split(",")]
            ingredientes_detectados = identificar_ingredientes(resto)
            recetas_detectadas = identificar_recetas(resto)

            if ingredientes_detectados == "" and recetas_detectadas == "":
                resultado = ""
            else:
                resultado = texto_query(ingredientes_detectados, recetas_detectadas)
                resultado = translate_text_from_to(resultado, source_lang="ES", target_lang="EN-US")
        else:
            ingredientes_detectados = identificar_ingredientes(prompt)
            recetas_detectadas = identificar_recetas(prompt)
            resultado = texto_query(ingredientes_detectados, recetas_detectadas)
            resultado = translate_text_from_to(resultado, source_lang="ES", target_lang="EN-US")

    if predicted_label == "solicitud_receta":
        query = obtener_receta(token, resultado, restricciones_detectadas)
        result = {
        "label": query['receta'],
        "dietLabels": query['dietas'],
        "healthLabels": query["salud"],
        "cautions": query["precauciones"],
        "ingredientLines": query['ingredientes'],
        "ingredients": query['nombre_ingredientes'],
        "calories": query['calorias'],
        "totalTime": query["tiempoTotal"],
        "mealType": query["tipo_comida"],
        "dishType": query["tipo_plato"],
        "instructions": query['instrucciones'],
        "type": predicted_label
    }
    elif predicted_label == "solicitud_menu":
        # Aquí usamos la función extraer_tiempo
        dias = extraer_tiempo(prompt)
        query = generar_menu(token, resultado, dias, restricciones_detectadas)
        query = json.loads(query)
        result = {
            "menus": query['menus'],
            "type": query['type'],
            "total_calories": query['total_calories'],
            "type_query": predicted_label,
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
                "type": predicted_label
                    }
        translated_response = translate_text_from_to(response, source_lang="EN", target_lang="ES")
        #responder
        result = {
        "query": prompt,
        "response": translated_response,
        "type": predicted_label
    }
    else:
        result = {
        "type": predicted_label,
    }
    # Convertir a JSON
    result_json = json.dumps(result, indent=4)
    return result_json