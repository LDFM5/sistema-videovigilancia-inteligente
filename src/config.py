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

# =========================
# RUTAS BASE
# =========================

# Carpeta raíz del proyecto (sube desde src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(BASE_DIR, "models")
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidences")

# Crear carpeta de evidencias si no existe
os.makedirs(EVIDENCE_DIR, exist_ok=True)

# Ruta del modelo entrenado
MODEL_PATH = os.path.join(MODELS_DIR, "best.pt")
POSE_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n-pose.pt")


# =========================
# DETECCIÓN
# =========================

CONF_WEAPON = 0.5   # Umbral de confianza mínimo


# =========================
# VENTANAS TEMPORALES
# =========================
WINDOW_SECONDS = 1.5          
ACTIVATION_THRESHOLD = 15     

BEHAVIOR_WINDOW_SECONDS = 4.0      
BEHAVIOR_ACTIVATION_THRESHOLD = 30 # Requiere ~2.5 seg para asaltos

# Umbral ultra rápido para eventos impulsivos (Golpes)
GOLPE_ACTIVATION_THRESHOLD = 2     # Con solo 2 frames de movimiento brusco, dispara


# =========================
# GRABACIÓN
# =========================

PRE_BUFFER_SECONDS = 10    # Segundos guardados ANTES de que se detecte el arma
POST_BUFFER_SECONDS = 15   # Segundos de grabación DESPUÉS de que el arma desaparece
RECORDING_FPS = 15

# =========================
# CÁMARAS
# =========================

CAMERA_INDEXES = {
    "webcam": 0,
    #"usb": 2
}

# =========================
# COLORES (BGR)
# =========================

COLOR_GUN = (0, 0, 255)       # Rojo
COLOR_KNIFE = (0, 255, 255)   # Amarillo
COLOR_ALERT = (0, 0, 255)

# =========================
# COMPORTAMIENTOS
# =========================
# Velocidad de golpe: Cuántas veces la longitud de su propio torso 
# recorrería la mano en 1 segundo si mantuviera esa velocidad.
UMBRAL_VELOCIDAD_GOLPE = 10.0

# Velocidad a la que los hombros deben "desplomarse" hacia el piso.
# Un valor de 2.0 o 2.5 suele ser perfecto para ignorar a alguien sentándose rápido.
UMBRAL_VELOCIDAD_CAIDA = 2.0

# =========================
# TELEGRAM
# =========================

try:
    from config_local import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
except:
    TELEGRAM_TOKEN = None
    TELEGRAM_CHAT_ID = None
