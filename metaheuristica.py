# Archivo: metaheuristica.py
import sys
import requests
import random
import json
import re
import time  
from metricas.bert import calcular_metricas_bert_huggingface
from operadores_geneticos.mutacion import mutacion
from operadores_geneticos.crossover import crossover

LLM_API_URL = "http://localhost:11434"

#LLM_API_URL = "http://localhost:11434/api/chat"

# Esta función se encargará de generar los tweets. En base a los prompts creados y a los textos de referencia.
# Tambien es mejorable su estructura, para que sea más eficiente y entienda mejor el proceso.
def consultar_llm(prompt, texto_referencia):
    """
    Consulta al modelo LLM para generar un texto basado en el prompt y el texto de referencia.
    """
    payload = {
        "model": "llama3.1",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an assistant specializing in generating synthetic data for training AI systems. "
                    "Your task is to create short tweets based on the provided context. Respond with the tweet only, "
                    "without any additional text or explanations, and avoid emojis or hashtags."
                    
                )
            },
            {
                "role": "user",
                "content": (
                    f"Generate a tweet based on the following prompt:\n"
                    f"Prompt: {prompt}\n"
                    f"Reference text: {texto_referencia}"
                    f"dont respond with the reference text"
                )
            }
        ],
        "stream": False,
        "temperature": 0.8
    }

    try:
        response = requests.post(LLM_API_URL, json=payload)
        if response.status_code == 200:
            content = response.json().get("message", {}).get("content", "").strip()
            tweet = extraer_tweet(content)
            return tweet
        else:
            print(f"Error al conectarse al LLM: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error de conexión: {e}")
        return None
## Esta función me funcionó para extraer el tweet de un texto, pero no es muy eficiente. Me di cuenta que el modelo que use respondia siempre de la misma manera y por eso me sirvio.
def extraer_tweet(texto):
    """
    Extrae únicamente el tweet de un texto que puede incluir prefijos o explicaciones adicionales.
    """
    match = re.search(r'"(.*?)"', texto)
    if match:
        return match.group(1)
    posibles_tweets = re.split(r"Here's a tweet based on the provided context:|vs", texto, maxsplit=1)
    if len(posibles_tweets) > 1:
        return posibles_tweets[1].strip()
    return texto.strip()

# Guardar progreso en JSON
def guardar_progreso_json(poblacion, generacion, archivo_json="progreso_generaciones.json"):
    """
    Guarda el progreso de cada generación en un archivo JSON.
    Este archivo es clave para ver la evolución del fitness y poder hacer el análisis del comportamiento de la metaheuristica.
    """
    try:
        progreso = []
        try:
            with open(archivo_json, "r") as file:
                progreso = json.load(file)
        except (IOError, json.JSONDecodeError):
            pass
        progreso.append({
            "generacion": generacion,
            "individuos": [
                {"fitness": individuo["fitness"], "prompt": individuo["prompt"]}
                for individuo in poblacion
            ]
        })
        with open(archivo_json, "w") as file:
            json.dump(progreso, file, indent=4)
        print(f"Progreso de la generación {generacion} guardado en {archivo_json}.")
    except Exception as e:
        print(f"Error al guardar el progreso en JSON: {e}")

# Esto me sirvio en algun punto, se puede eliminar si no se usa.
def guardar_progreso_txt(poblacion, generacion, archivo_txt="progreso_generaciones.txt"):
    """
    Guarda el progreso de cada generación en un archivo TXT.
    """
    try:
        with open(archivo_txt, "a") as file:
            file.write(f"\n=== Generación {generacion} ===\n")
            for i, individuo in enumerate(poblacion, 1):
                file.write(f"Individuo {i}: Fitness: {individuo['fitness']} Prompt: {individuo['prompt']}\n")
        print(f"Progreso de la generación {generacion} guardado en {archivo_txt}.")
    except Exception as e:
        print(f"Error al guardar el progreso en TXT: {e}")

# Cargar datos desde JSON
def cargar_datos_json(ruta_archivo):
    try:
        with open(ruta_archivo, "r") as file:
            return json.load(file)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error al cargar el archivo JSON {ruta_archivo}: {e}")
        return None

# Evaluar población
def generar_y_evaluar(poblacion_actual, texto_referencia, consultar_llm, calcular_metricas_bert_huggingface):
    """
    Evalúa la población actual generando respuestas y calculando métricas.
    """
    for entrada in poblacion_actual:
        prompt = entrada["prompt"]
        respuesta = consultar_llm(prompt, texto_referencia)
        if respuesta:
            fitness = calcular_metricas_bert_huggingface({
                "data_generada": respuesta,
                "texto_referencia": texto_referencia
            })["f1"]
            entrada["data_generada"] = respuesta
            entrada["fitness"] = fitness
        else:
            entrada["data_generada"] = ""
            entrada["fitness"] = 0

# Selección por torneo
def torneo(poblacion, k):
    participantes = random.sample(poblacion, k)
    return max(participantes, key=lambda x: x["fitness"])

# Generar nueva población
def generar_nueva_poblacion(poblacion_actual, texto_referencia, k, num_elitismo, consultar_llm, calcular_metricas_bert_huggingface, p_cross=0.9, p_mut=0.01):
    
    nueva_poblacion = [] 
    poblacion_actual.sort(key=lambda x: x["fitness"], reverse=True)
    ## scamos los mejores individuos por elitismo.
    elite = poblacion_actual[:num_elitismo]
    nueva_poblacion.extend(elite)
    candidatos_torneo = poblacion_actual
    
    ## Aplicamos Crossover si p menor que pro_cross
    while len(nueva_poblacion) < len(poblacion_actual): ###iteramos hasta que tengamos la cantidad de individuos que necesitamos.
        if random.random() < p_cross: ## este random es por para ver si aplicamos crossover 
            padre1 = torneo(candidatos_torneo, k)
            padre2 = torneo(candidatos_torneo, k)
            hijo = crossover(padre1, padre2)
            respuesta = consultar_llm(hijo["prompt"], texto_referencia)
             ### el individuo generado por crossover se evalua de inmediato para tener su fitness
            if respuesta:
                fitness = calcular_metricas_bert_huggingface({
                    "data_generada": respuesta,
                    "texto_referencia": texto_referencia
                })["f1"]
                hijo["data_generada"] = respuesta
                hijo["fitness"] = fitness

        if random.random() < p_mut: ## este random es por para ver si aplicamos mutation
            hijo = mutacion(hijo)
            respuesta = consultar_llm(hijo["prompt"], texto_referencia) 
            ### el individuo generado por mutación se evalua de inmediato para tener su fitness
            if respuesta:
                fitness = calcular_metricas_bert_huggingface({
                    "data_generada": respuesta,
                    "texto_referencia": texto_referencia
                })["f1"]
                hijo["data_generada"] = respuesta
                hijo["fitness"] = fitness
        nueva_poblacion.append(hijo)
    return nueva_poblacion


########################################
#   while len(nueva_poblacion) < len(poblacion_actual): ###iteramos hasta que tengamos la cantidad de individuos que necesitamos.
#        if random.random() < 0.15: ## este random es por para ver la mutación, del proceso reproductivo 15% será por mutación
#            individuo = torneo(candidatos_torneo, k)
#            mutado = mutacion(individuo)
#            respuesta = consultar_llm(mutado["prompt"], texto_referencia) 
#            ### el individuo generado por mutación se evalua de inmediato para tener su fitness
#            if respuesta:
#                fitness = calcular_metricas_bert_huggingface({
#                    "data_generada": respuesta,
#                    "texto_referencia": texto_referencia
#                })["f1"]
#                mutado["data_generada"] = respuesta
#                mutado["fitness"] = fitness
#                nueva_poblacion.append(mutado)
#        else: ##prob restante es para crossover. En el estado del arte se aplica mucho mas crossover que mutación. Eventualmente se podrian probar nuevos valores
#            padre1 = torneo(candidatos_torneo, k)
#            padre2 = torneo(candidatos_torneo, k)
#            hijo = crossover(padre1, padre2)
#            respuesta = consultar_llm(hijo["prompt"], texto_referencia)
#             ### el individuo generado por crossover se evalua de inmediato para tener su fitness
#            if respuesta:
#                fitness = calcular_metricas_bert_huggingface({
#                    "data_generada": respuesta,
#                    "texto_referencia": texto_referencia
#                })["f1"]
#                hijo["data_generada"] = respuesta
#                hijo["fitness"] = fitness
#                nueva_poblacion.append(hijo)
#   return nueva_poblacion
##################################################

# Flujo principal
def main():
    start_time = time.time() 
    args = sys.argv[1:]
    mode = args[0] ##Este parametro lo use en una implementación de distintos crossovers, sin embargo, en la versión final no lo ocupe.
    num_generaciones = int(args[1])
    num_elitismo = int(args[2])
    archivo_json = args[3]
    texto_referencia = args[4]
    k = int(args[5])  

    # Cargamos los prompts inciales para evaluarlos
    poblacion_actual = cargar_datos_json(archivo_json)

    # Evaluar la población inicial
    generar_y_evaluar(poblacion_actual, texto_referencia, consultar_llm, calcular_metricas_bert_huggingface)


    #Hasta este punto, tenemos los individuos completos guardados en el archivo json, es decir prompt, data generada y fitness.
    # A continuación, empieza el proceso evolutivo.
    for generacion in range(1, num_generaciones + 1):
        print(f"\n=== GENERACIÓN {generacion} ===")
        poblacion_actual = generar_nueva_poblacion(
            poblacion_actual, texto_referencia, k=k, num_elitismo=num_elitismo,
            consultar_llm=consultar_llm, calcular_metricas_bert_huggingface=calcular_metricas_bert_huggingface
        )
        guardar_progreso_txt(poblacion_actual, generacion)
        guardar_progreso_json(poblacion_actual, generacion)

    with open("data_final.json", "w") as file:
        json.dump(poblacion_actual, file, indent=4)

    # Calcular y mostrar el tiempo de ejecución total
    end_time = time.time()  # Registrar el tiempo final
    elapsed_time = end_time - start_time
    print(f"\nTiempo total de ejecución: {elapsed_time:.2f} segundos.")

if __name__ == "__main__":
    main()