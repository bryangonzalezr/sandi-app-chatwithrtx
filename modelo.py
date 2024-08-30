from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from difflib import get_close_matches
import json

ingredientes = [
    "aceite de oliva", "ajo", "albahaca", "alcachofa", "alcaparras", "almendras", "anchoas", "arándanos", "arroz", 
    "atún", "avellanas", "aguacate", "palta", "azúcar", "bacalao", "bacon", "batata", "berenjena", "berros", "brócoli", 
    "calabacín", "zapallo italiano", "calabaza", "zapallo", "caldo de pollo", "canela", "cangrejo", "carne de cerdo", 
    "carne de res", "cebolla", "cebollín", "celery", "apio", "cereza", "cerdo", "chancho", "champiñones", "chayote", 
    "chía", "chile", "ají", "chocolate", "cilantro", "ciruela", "clavo de olor", "coco", "col", "coliflor", "comino", 
    "corazón de alcachofa", "costillas", "crema de leche", "cúrcuma", "durazno", "endibia", "eneldo", "espárragos", 
    "espinacas", "estragón", "fideos", "frijoles", "porotos", "garbanzos", "gelatina", "jengibre", "granos de pimienta", 
    "guacamole", "guayaba", "harina", "higos", "huevo", "jamón", "jengibre", "judías verdes", "kale", "kiwi", 
    "langosta", "lentejas", "lechuga", "limón", "lima", "maíz", "mango", "maní", "manzana", "manteca", "mantequilla", 
    "maracuyá", "mayonesa", "mejillones", "melón", "menta", "mermelada", "miel", "mostaza", "naranja", "nuez moscada", 
    "nueces", "orégano", "pan", "papas", "paprika", "parmesano", "pasas", "pasta", "pepino", "perejil", "pimentón", 
    "pimiento rojo", "pimiento verde", "piña", "pistachos", "pollo", "prosciutto", "queso cheddar", "queso crema", 
    "queso feta", "queso mozzarella", "quinoa", "rábano", "ramitas de canela", "remolacha", "betarraga", "romero", 
    "rúcula", "salmón", "sal", "salsa de soja", "salsa de tomate", "sardinas", "semillas de girasol", "semillas de sésamo", 
    "setas", "soja", "spaghetti", "tomate", "tomate cherry", "tomillo", "tocino", "trigo", "trucha", "uva", "vainilla", 
    "vinagre", "vinagre balsámico", "vinagre de manzana", "yogur", "zanahoria", "zucchini", "aceitunas", 
    "almendra molida", "arroz basmati", "arroz integral", "azafrán", "azúcar glas", "bicarbonato de sodio", 
    "cacao en polvo", "café", "calamares", "camarones", "carnes rojas", "caviar", "champiñón", "chorizo", 
    "coco rallado", "curry", "datiles", "endibias", "farro", "garam masala", "harina de avena", "harina de maíz", 
    "hierbabuena", "hojas de laurel", "hongos", "jalapeños", "lasaña", "lechuga romana", "levadura", "macarrones", 
    "mandarina", "mango verde", "masa de hojaldre", "menta fresca", "merluza", "mostaza Dijon", "panceta", 
    "papas fritas", "pargo", "pasta de curry", "pasta filo", "pechuga de pollo", "pepperoni", "peras", "piñones", 
    "pitahaya", "polenta", "queso brie", "queso gouda", "queso manchego", "queso ricotta", "requesón", "rosbif", 
    "sal marina", "salami", "salmón ahumado", "salsa de ostras", "salsa inglesa", "sandía", "sepia", "soja texturizada", 
    "solomillo", "tapioca", "tofu", "camote", "arúgula", "chirimoya", "jícama", "acelga", "cebollas caramelizadas", 
    "frijoles negros", "caldo de verduras", "chalotas", "champiñones portobello", "carne molida", "aceite de coco", 
    "queso parmesano", "espaguetis", "estragón fresco", "almejas", "hojuelas de chile", "pepinillos", "perejil fresco", 
    "cáscara de limón", "pimienta negra", "curry en polvo", "semillas de chía", "mantequilla de maní", 
    "carne de cordero", "aguaymanto", "harina de almendra", "leche de almendra", "salchicha", "clara de huevo", 
    "yema de huevo", "miel de maple", "melocotón", "coco deshidratado", "pepitas de calabaza", "quiche", "jalapeño", 
    "vino rosado", "jarabe de maíz", "miso", "fresas", "frambuesas", "grosellas", "arándanos rojos", "gelatina sin sabor", 
    "caldo de carne", "nata", "crema agria", "carne de pato", "carne de venado", "carne de faisán", "caviar negro", 
    "caviar rojo", "paté", "trufas", "sésamo", "calabacita", "cuscús", "baba ganoush", "tahini", "chile jalapeño", 
    "chile serrano", "chile habanero", "chile poblano", "chipotle", "sriracha", "mostaza amarilla", "manzana verde", 
    "mermelada de naranja", "mermelada de frambuesa", "compota de manzana", "queso camembert", "queso gruyere", 
    "queso azul", "queso cottage", "queso fresco", "pechuga de pavo", "jamón serrano", "carne en conserva", 
    "cordero", "jamón ibérico", "chuletas de cerdo", "longaniza", "fuet", "sopa de miso", "brotes de soya", 
    "brotes de alfalfa", "champiñones shiitake", "champiñones cremini", "edamame", "tempura", "wasabi", 
    "jengibre encurtido", "vino de arroz", "vinagre de arroz", "sake", "mirin", "dashi", "udon", "soba", 
    "ramen", "mochi", "nata líquida", "queso mascarpone", "queso pecorino", "gorgonzola", "queso de cabra", 
    "nuez de macadamia", "leche de coco", "nata montada", "queso halloumi", "queso de oveja", "bocadillo", 
    "almendras tostadas"
]

recetas = ["pizza"]

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

# 1. Cargar el modelo entrenado y el tokenizador
model_path = "./content/results/checkpoint-830"  # Asegúrate de que esta ruta sea correcta
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Asegúrate de que el tokenizador tenga un token de relleno
tokenizer.pad_token = tokenizer.eos_token

# 2. Preparar tu prompt
prompt = "Dame una receta de pizza"

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

print(f"Prompt: {prompt}")
print(f"Predicción: {predicted_label}")
print("Probabilidades:")
for i, prob in enumerate(probabilities[0]):
    print(f"  {label_map_inverse[i]}: {prob.item():.4f}")

confidence_threshold = 0.6  # Establece un umbral de confianza

max_probability = torch.max(probabilities).item()

if max_probability < confidence_threshold:
    print("La predicción no es lo suficientemente confiable.")
else:
    print(f"Predicción: {predicted_label} (Confianza: {max_probability:.4f})")

ingredientes_detectados = identificar_ingredientes(prompt)
recetas_detectadas = identificar_recetas(prompt)

if predicted_label == "solicitud_receta":
    result = {
    "query": recetas_detectadas,
    "ingredientes": ingredientes_detectados,
    "type": predicted_label,
}
elif predicted_label == "solicitud_menu":
    result = {
    "query": recetas_detectadas,
    "ingredientes": ingredientes_detectados,
    "type": predicted_label,
    "time": 1,
}
elif predicted_label == "pregunta_cocina":
    result = {
    "query": prompt,
    "type": predicted_label,
}
else:
    result = {
    "type": predicted_label,
}

# Convertir a JSON
result_json = json.dumps(result, indent=4)
print(result_json)