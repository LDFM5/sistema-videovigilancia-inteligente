"""
config.py

Archivo de configuración global del sistema.

Define constantes, parámetros y rutas utilizadas en el proyecto, tales como:
- Ruta del modelo entrenado
- Carpeta de almacenamiento de evidencias
- Umbrales de detección
- Parámetros de ventana temporal
- Duración de grabación
- Índices de cámaras
- Colores utilizados en visualización

Centralizar la configuración permite modificar el comportamiento del sistema
sin alterar la lógica de los módulos principales.
"""

import os
import json

# =========================
# RUTAS BASE
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_JSON_CONFIG = os.path.join(BASE_DIR, "config_usuario.json")

MODELS_DIR = os.path.join(BASE_DIR, "models")
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidences")

# Crear carpeta de evidencias si no existe
os.makedirs(EVIDENCE_DIR, exist_ok=True)

# Rutas de los modelos neuronales entrenados
WEAPON_MODEL_PATH = os.path.join(MODELS_DIR, "Modelo_objetos_sospechosos.pt")
POSE_MODEL_PATH = os.path.join(MODELS_DIR, "yolo11n-pose.pt")
BEHAVIOR_MODEL_PATH = os.path.join(MODELS_DIR, "comportamiento.pth")

# =========================
# VENTANAS TEMPORALES
# =========================
WINDOW_SECONDS = 1.5           
ACTIVATION_THRESHOLD = 15     

# =========================
# PARAMETROS DE GRABACION FIJOS
# =========================
RECORDING_FPS = 15

# =========================
# CÁMARAS
# =========================
CAMERA_INDEXES = {
    "webcam": 0,
    #"phone": 1
}

# =========================
# CLASES DE DETECCIÓN DE ARMAS
# =========================
CLASES_ARMAS_ALERTA = ["firearm", "melee_weapon"]

# =========================
# TELEGRAM
# =========================
try:
    from config_local import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
except ImportError:
    TELEGRAM_TOKEN = None
    TELEGRAM_CHAT_ID = None


# =========================================================================
# SISTEMA DE PERSISTENCIA Y ADAPTACION COMPARTIDA VIA JSON
# =========================================================================

# Valores por defecto para la primera inicializacion automatica de fabrica
VALORES_FABRICA = {
    "cfg_armas": True,
    "cfg_comportamiento": False,
    "cfg_confianza": 0.50,
    "cfg_prebuffer": 10,
    "cfg_postbuffer": 15,
    "cfg_debug": False 
}

def cargar_configuracion_inicial():
    """
    Lee el archivo de datos JSON en el arranque. Si no se localiza en el disco,
    genera uno nuevo de forma automatica utilizando los valores predeterminados.
    """
    if not os.path.exists(RUTA_JSON_CONFIG):
        with open(RUTA_JSON_CONFIG, 'w', encoding='utf-8') as f:
            json.dump(VALORES_FABRICA, f, indent=2)
        return VALORES_FABRICA
    try:
        with open(RUTA_JSON_CONFIG, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return VALORES_FABRICA

# Cargar la configuracion dinamica desde el almacenamiento persistente
_config_disco = cargar_configuracion_inicial()

# Asignacion automatica de variables en memoria leidas desde el archivo JSON
ACTIVAR_MODELO_ARMAS = bool(_config_disco.get("cfg_armas", True))
ACTIVAR_MODELO_COMPORTAMIENTO = bool(_config_disco.get("cfg_comportamiento", False))
CONF_WEAPON = float(_config_disco.get("cfg_confianza", 0.50))
PRE_BUFFER_SECONDS = int(_config_disco.get("cfg_prebuffer", 10))
POST_BUFFER_SECONDS = int(_config_disco.get("cfg_postbuffer", 15))
MODO_DEBUG = bool(_config_disco.get("cfg_debug", False))  # 🚨 Se lee del JSON como booleano puro


def guardar_configuracion_disco(nuevos_valores):
    """
    Guarda las modificaciones del panel web directamente sobre el archivo JSON,
    y actualiza simultaneamente la memoria del modulo para el ciclo de la IA.
    """
    with open(RUTA_JSON_CONFIG, 'w', encoding='utf-8') as f:
        json.dump(nuevos_valores, f, indent=2)
    
    import sys
    modulo = sys.modules[__name__]
    setattr(modulo, "ACTIVAR_MODELO_ARMAS", bool(nuevos_valores["cfg_armas"]))
    setattr(modulo, "ACTIVAR_MODELO_COMPORTAMIENTO", bool(nuevos_valores["cfg_comportamiento"]))
    setattr(modulo, "CONF_WEAPON", float(nuevos_valores["cfg_confianza"]))
    setattr(modulo, "PRE_BUFFER_SECONDS", int(nuevos_valores["cfg_prebuffer"]))
    setattr(modulo, "POST_BUFFER_SECONDS", int(nuevos_valores["cfg_postbuffer"]))
    setattr(modulo, "MODO_DEBUG", bool(nuevos_valores["cfg_debug"]))  # 🚨 Mantiene actualizada la RAM del módulo


def restaurar_valores_fabrica():
    """
    Restablece el archivo JSON y las variables del sistema a los parametros de fabrica.
    """
    guardar_configuracion_disco(VALORES_FABRICA)
    return VALORES_FABRICA