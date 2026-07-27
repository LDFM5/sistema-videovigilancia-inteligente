"""
config.py

Archivo de configuración global del sistema.

Centraliza constantes, rutas de modelos, parámetros de fábrica y 
el motor de persistencia/sincronización dinámica mediante JSON.
"""

import os
import json
import sys

# =========================
# RUTAS BASE Y DIRECTORIOS
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_JSON_CONFIG = os.path.join(BASE_DIR, "config_usuario.json")

MODELS_DIR = os.path.join(BASE_DIR, "models")
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidences")

# Garantizar la existencia de carpetas operativas esenciales
os.makedirs(EVIDENCE_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Rutas de los modelos neuronales entrenados
WEAPON_MODEL_PATH = os.path.join(MODELS_DIR, "Modelo_objetos_sospechosos.pt")
POSE_MODEL_PATH = os.path.join(MODELS_DIR, "yolo11n-pose.pt")
BEHAVIOR_MODEL_PATH = os.path.join(MODELS_DIR, "comportamiento.pth")

# =========================
# VENTANAS TEMPORALES Y PARÁMETROS FIJOS
# =========================
WINDOW_SECONDS = 1.5           
ACTIVATION_THRESHOLD = 15     
RECORDING_FPS = 15

# =========================
# CÁMARAS Y CANALES
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
# CREDENCIALES DE TELEGRAM
# =========================
try:
    from config_local import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
except ImportError:
    TELEGRAM_TOKEN = None
    TELEGRAM_CHAT_ID = None


# =========================================================================
# SISTEMA DE PERSISTENCIA Y ADAPTACIÓN COMPARTIDA VÍA JSON
# =========================================================================

# Valores predeterminados de fábrica
VALORES_FABRICA = {
    "cfg_armas": True,
    "cfg_comportamiento": False,
    "cfg_confianza_armas": 0.50,
    "cfg_confianza_comportamiento": 0.50,
    "cfg_prebuffer": 10,
    "cfg_postbuffer": 15,
    "cfg_debug": False 
}

def cargar_configuracion_inicial():
    """
    Carga la configuración desde el disco JSON.
    Si no existe o está corrupto, lo regenera con los valores de fábrica.
    """
    if not os.path.exists(RUTA_JSON_CONFIG):
        guardar_configuracion_disco(VALORES_FABRICA)
        return dict(VALORES_FABRICA)
        
    try:
        with open(RUTA_JSON_CONFIG, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        # Si el JSON estaba corrupto, se sobreescribe con seguridad para reparar el sistema
        guardar_configuracion_disco(VALORES_FABRICA)
        return dict(VALORES_FABRICA)


# Cargar configuración activa al inicializar el módulo
_config_disco = cargar_configuracion_inicial()

# Asignación de variables globales en memoria RAM del módulo
ACTIVAR_MODELO_ARMAS = bool(_config_disco.get("cfg_armas", True))
ACTIVAR_MODELO_COMPORTAMIENTO = bool(_config_disco.get("cfg_comportamiento", False))

CONF_WEAPON = float(_config_disco.get("cfg_confianza_armas", 0.50))
CONF_BEHAVIOR = float(_config_disco.get("cfg_confianza_comportamiento", 0.50))

PRE_BUFFER_SECONDS = int(_config_disco.get("cfg_prebuffer", 10))
POST_BUFFER_SECONDS = int(_config_disco.get("cfg_postbuffer", 15))
MODO_DEBUG = bool(_config_disco.get("cfg_debug", False))


def guardar_configuracion_disco(nuevos_valores):
    """
    Aplica una escritura atómica en disco para prevenir la corrupción del archivo JSON
    y actualiza en caliente los atributos globales del módulo config.
    """
    temp_path = RUTA_JSON_CONFIG + ".tmp"
    
    # 1. Escritura segura en archivo temporal
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(nuevos_valores, f, indent=2)
        
    # 2. Reemplazo atómico garantizado por el sistema operativo
    os.replace(temp_path, RUTA_JSON_CONFIG)
    
    # 3. Sincronización de variables en la memoria RAM de este módulo
    modulo = sys.modules[__name__]
    setattr(modulo, "ACTIVAR_MODELO_ARMAS", bool(nuevos_valores.get("cfg_armas", True)))
    setattr(modulo, "ACTIVAR_MODELO_COMPORTAMIENTO", bool(nuevos_valores.get("cfg_comportamiento", False)))
    
    setattr(modulo, "CONF_WEAPON", float(nuevos_valores.get("cfg_confianza_armas", 0.50)))
    setattr(modulo, "CONF_BEHAVIOR", float(nuevos_valores.get("cfg_confianza_comportamiento", 0.50)))
    
    setattr(modulo, "PRE_BUFFER_SECONDS", int(nuevos_valores.get("cfg_prebuffer", 10)))
    setattr(modulo, "POST_BUFFER_SECONDS", int(nuevos_valores.get("cfg_postbuffer", 15)))
    setattr(modulo, "MODO_DEBUG", bool(nuevos_valores.get("cfg_debug", False)))


def restaurar_valores_fabrica():
    """
    Restablece el archivo JSON y las variables del sistema a los parámetros de fábrica.
    """
    guardar_configuracion_disco(VALORES_FABRICA)
    return dict(VALORES_FABRICA)