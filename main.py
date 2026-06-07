import functions_framework
import pandas as pd
import numpy as np
import re
import pickle
import json
from datetime import datetime
import os

# Nombres de los archivos de modelos entrenados
VECTORIZER_FILE = 'tfidf_vectorizer.pkl'
SUPERVISED_MODEL_FILE = 'supervised_model.pkl'
PCA_MODEL_FILE = 'pca_model.pkl'
KMEANS_MODEL_FILE = 'kmeans_model.pkl'
CLUSTER_NAMES_FILE = 'cluster_names.json'

# Variables globales para almacenar los modelos en memoria
# Esto evita recargar los modelos en cada invocación
vectorizer = None
supervised_model = None
pca_model = None
kmeans_model = None
cluster_names = None

def load_models():
    """
    Carga los modelos entrenados desde los archivos pickle.
    Los modelos se cargan una sola vez en memoria para mejorar el rendimiento.
    """
    global vectorizer, supervised_model, pca_model, kmeans_model, cluster_names
    
    if vectorizer is None:
        with open(VECTORIZER_FILE, 'rb') as f:
            vectorizer = pickle.load(f)
            
    if supervised_model is None:
        with open(SUPERVISED_MODEL_FILE, 'rb') as f:
            supervised_model = pickle.load(f)
            
    if pca_model is None:
        with open(PCA_MODEL_FILE, 'rb') as f:
            pca_model = pickle.load(f)
            
    if kmeans_model is None:
        with open(KMEANS_MODEL_FILE, 'rb') as f:
            kmeans_model = pickle.load(f)
            
    if cluster_names is None:
        with open(CLUSTER_NAMES_FILE, 'r') as f:
            cluster_names = json.load(f)

def clean_text(text):
    """
    Realiza la limpieza del texto para mejorar la calidad de las variables.
    Se eliminan URLs, menciones, caracteres especiales y números que no aportan
    información relevante para el análisis de sentimientos.
    """
    if not isinstance(text, str):
        return ""
    
    # Convertir a minúsculas para normalizar el texto
    text = text.lower()
    
    # Eliminar URLs y enlaces web
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Eliminar menciones a otros usuarios
    text = re.sub(r'\@\w+', '', text)
    
    # Eliminar el símbolo de hashtag pero conservar la palabra
    text = re.sub(r'\#', '', text)
    
    # Eliminar caracteres especiales y números
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Normalizar espacios en blanco
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

@functions_framework.http
def predict_sentiment(request):
    """
    Cloud Function HTTP que realiza predicción de sentimiento y clustering de tweets.
    
    Entrada (JSON):
        {
            "tweet_text": "Texto del tweet a analizar"
        }
    
    Salida (JSON):
        {
            "status": "success",
            "data": {
                "timestamp": "Marca de tiempo",
                "student_name": "Nombre del estudiante",
                "proba_positive": Probabilidad de sentimiento positivo,
                "proba_negative": Probabilidad de sentimiento negativo,
                "proba_neutral": Probabilidad de sentimiento neutro,
                "proba_mixed": Probabilidad de sentimiento mixto,
                "class": "Clase predicha",
                "cluster": "Cluster al que pertenece"
            }
        }
    """
    
    # Cargar los modelos entrenados
    try:
        load_models()
    except Exception as e:
        return {"status": "error", "message": f"Error al cargar los modelos: {str(e)}"}, 500

    # Parsear el JSON de entrada
    request_json = request.get_json(silent=True)
    
    # Validar que el JSON contenga el campo requerido
    if not request_json or 'tweet_text' not in request_json:
        return {"status": "error", "message": "El JSON debe contener el campo 'tweet_text'"}, 400
        
    tweet_text = request_json['tweet_text']
    
    # ========================================================================
    # INGENIERÍA DE VARIABLES
    # ========================================================================
    
    # Realizar la limpieza del texto
    cleaned_text = clean_text(tweet_text)
    
    # Validar que el texto no esté vacío después de la limpieza
    if not cleaned_text:
        return {"status": "error", "message": "El texto está vacío después de la limpieza"}, 400
    
    # Aplicar la vectorización TF-IDF para transformar el texto en variables numéricas
    X_vec = vectorizer.transform([cleaned_text])
    
    # ========================================================================
    # MODELO SUPERVISADO - PREDICCIÓN DE SENTIMIENTO
    # ========================================================================
    
    # Obtener las probabilidades de cada clase de sentimiento
    probas = supervised_model.predict_proba(X_vec)[0]
    classes = supervised_model.classes_
    
    # Mapear las probabilidades a los nombres de las clases
    proba_dict = {f"proba_{c.lower()}": float(p) for c, p in zip(classes, probas)}
    
    # Obtener la clase predicha (la de mayor probabilidad)
    predicted_class = supervised_model.predict(X_vec)[0]
    
    # ========================================================================
    # MODELO NO SUPERVISADO - CLUSTERING
    # ========================================================================
    
    # Aplicar la reducción de dimensiones con PCA
    X_pca = pca_model.transform(X_vec.toarray())
    
    # Predecir el cluster al que pertenece el tweet
    cluster_id = int(kmeans_model.predict(X_pca)[0])
    cluster_name = cluster_names.get(str(cluster_id), f"Cluster_{cluster_id}")
    
    # ========================================================================
    # FORMATEAR RESPUESTA
    # ========================================================================
    
    # Generar marca de tiempo en formato DD/MM/YYYYHH:MM:SS
    timestamp = datetime.now().strftime("%d/%m/%Y%H:%M:%S")
    
    # Construir el JSON de respuesta con el formato requerido
    response = {
        "status": "success",
        "data": {
            "timestamp": timestamp,
            "student_name": "Uriel Ramos Calleja",
            "proba_positive": proba_dict.get("proba_positive", 0.0),
            "proba_negative": proba_dict.get("proba_negative", 0.0),
            "proba_neutral": proba_dict.get("proba_neutral", 0.0),
            "proba_mixed": proba_dict.get("proba_mixed", 0.0),
            "class": predicted_class,
            "cluster": cluster_name
        }
    }
    
    return response, 200
