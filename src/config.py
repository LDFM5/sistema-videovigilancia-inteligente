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

# Crear las carpetas de trabajo si no existen.
os.makedirs(EVIDENCE_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Rutas de los modelos neuronales
WEAPON_DEFAULT_MODEL = "Modelo_objetos_sospechosos.pt"
BEHAVIOR_DEFAULT_MODEL = "comportamiento.pth"
POSE_DEFAULT_MODEL = "yolo11n-pose.pt"

POSE_MODEL_PATH = os.path.join(MODELS_DIR, POSE_DEFAULT_MODEL)

def resolver_ruta_modelo(nombre_o_ruta, tipo="armas"):
    """Resuelve la ruta absoluta de un archivo de modelo a partir de su nombre o ruta relativa/absoluta."""
    if not nombre_o_ruta:
        nombre_o_ruta = WEAPON_DEFAULT_MODEL if tipo == "armas" else BEHAVIOR_DEFAULT_MODEL
    
    nombre_str = str(nombre_o_ruta).strip().strip('"\'')
    if not nombre_str:
        nombre_str = WEAPON_DEFAULT_MODEL if tipo == "armas" else BEHAVIOR_DEFAULT_MODEL

    # Si es ruta absoluta existente o no, mantener la ruta absoluta indicada por el usuario
    if os.path.isabs(nombre_str):
        return os.path.normpath(nombre_str)
        
    # Si existe directamente relativo al directorio de ejecución
    if os.path.exists(nombre_str):
        return os.path.abspath(nombre_str)

    # Si existe en models/
    en_models = os.path.join(MODELS_DIR, os.path.basename(nombre_str))
    if os.path.exists(en_models):
        return os.path.normpath(en_models)
        
    # Si existe relativo a BASE_DIR
    en_root = os.path.join(BASE_DIR, nombre_str)
    if os.path.exists(en_root):
        return os.path.normpath(en_root)
        
    return os.path.normpath(os.path.join(MODELS_DIR, nombre_str))

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
    "cfg_armas": False,
    "cfg_comportamiento": False,
    "cfg_confianza_armas": 0.50,
    "cfg_confianza_comportamiento": 0.50,
    "cfg_prebuffer": 10,
    "cfg_postbuffer": 15,
    "cfg_debug": False,
    "cfg_modelo_armas": WEAPON_DEFAULT_MODEL,
    "cfg_modelo_comportamiento": BEHAVIOR_DEFAULT_MODEL,
    "camaras": {
        "webcam": 0
    }
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
            data = json.load(f)
            # Asegurar que la clave 'camaras' exista
            if "camaras" not in data:
                data["camaras"] = dict(VALORES_FABRICA["camaras"])
            return data
    except Exception:
        # Si el JSON estaba corrupto, se sobreescribe con seguridad para reparar el sistema
        guardar_configuracion_disco(VALORES_FABRICA)
        return dict(VALORES_FABRICA)


# Cargar configuración activa al inicializar el módulo
_config_disco = cargar_configuracion_inicial()

# Asignación de variables globales en memoria RAM del módulo
ACTIVAR_MODELO_ARMAS = bool(_config_disco.get("cfg_armas", False))
ACTIVAR_MODELO_COMPORTAMIENTO = bool(_config_disco.get("cfg_comportamiento", False))

MODELO_ARMAS_NOMBRE = str(_config_disco.get("cfg_modelo_armas", WEAPON_DEFAULT_MODEL))
MODELO_COMPORTAMIENTO_NOMBRE = str(_config_disco.get("cfg_modelo_comportamiento", BEHAVIOR_DEFAULT_MODEL))

WEAPON_MODEL_PATH = resolver_ruta_modelo(MODELO_ARMAS_NOMBRE, "armas")
BEHAVIOR_MODEL_PATH = resolver_ruta_modelo(MODELO_COMPORTAMIENTO_NOMBRE, "comportamiento")

CONF_WEAPON = float(_config_disco.get("cfg_confianza_armas", 0.50))
CONF_BEHAVIOR = float(_config_disco.get("cfg_confianza_comportamiento", 0.50))

PRE_BUFFER_SECONDS = int(_config_disco.get("cfg_prebuffer", 10))
POST_BUFFER_SECONDS = int(_config_disco.get("cfg_postbuffer", 15))
MODO_DEBUG = bool(_config_disco.get("cfg_debug", False))
CAMERA_INDEXES = dict(_config_disco.get("camaras", {"webcam": 0}))


def obtener_camaras_configuradas():
    """Retorna el diccionario actual de cámaras desde la configuración persistida."""
    cfg = cargar_configuracion_inicial()
    return cfg.get("camaras", CAMERA_INDEXES)


def guardar_configuracion_disco(nuevos_valores):
    """
    Aplica una escritura atómica en disco para prevenir la corrupción del archivo JSON
    y actualiza en caliente los atributos globales del módulo config.
    """
    # Preservar la sección de cámaras si no viene en los nuevos valores recibidos
    if "camaras" not in nuevos_valores:
        cfg_actual = cargar_configuracion_inicial()
        nuevos_valores["camaras"] = cfg_actual.get("camaras", VALORES_FABRICA["camaras"])

    temp_path = RUTA_JSON_CONFIG + ".tmp"
    
    # 1. Escribir los datos en un archivo temporal.
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(nuevos_valores, f, indent=2)
        
    # 2. Reemplazar el archivo de configuración de forma atómica.
    os.replace(temp_path, RUTA_JSON_CONFIG)
    
    # 3. Sincronización de variables en la memoria RAM de este módulo
    modulo = sys.modules[__name__]
    setattr(modulo, "ACTIVAR_MODELO_ARMAS", bool(nuevos_valores.get("cfg_armas", False)))
    setattr(modulo, "ACTIVAR_MODELO_COMPORTAMIENTO", bool(nuevos_valores.get("cfg_comportamiento", False)))
    
    nombre_armas = str(nuevos_valores.get("cfg_modelo_armas", getattr(modulo, "MODELO_ARMAS_NOMBRE", WEAPON_DEFAULT_MODEL)))
    nombre_comp = str(nuevos_valores.get("cfg_modelo_comportamiento", getattr(modulo, "MODELO_COMPORTAMIENTO_NOMBRE", BEHAVIOR_DEFAULT_MODEL)))
    setattr(modulo, "MODELO_ARMAS_NOMBRE", nombre_armas)
    setattr(modulo, "MODELO_COMPORTAMIENTO_NOMBRE", nombre_comp)
    setattr(modulo, "WEAPON_MODEL_PATH", resolver_ruta_modelo(nombre_armas, "armas"))
    setattr(modulo, "BEHAVIOR_MODEL_PATH", resolver_ruta_modelo(nombre_comp, "comportamiento"))

    setattr(modulo, "CONF_WEAPON", float(nuevos_valores.get("cfg_confianza_armas", 0.50)))
    setattr(modulo, "CONF_BEHAVIOR", float(nuevos_valores.get("cfg_confianza_comportamiento", 0.50)))
    
    setattr(modulo, "PRE_BUFFER_SECONDS", int(nuevos_valores.get("cfg_prebuffer", 10)))
    setattr(modulo, "POST_BUFFER_SECONDS", int(nuevos_valores.get("cfg_postbuffer", 15)))
    setattr(modulo, "MODO_DEBUG", bool(nuevos_valores.get("cfg_debug", False)))
    if "camaras" in nuevos_valores:
        setattr(modulo, "CAMERA_INDEXES", dict(nuevos_valores["camaras"]))


def restaurar_valores_fabrica():
    """
    Restablece el archivo JSON y las variables del sistema a los parámetros de fábrica.
    """
    guardar_configuracion_disco(VALORES_FABRICA)
    return dict(VALORES_FABRICA)

