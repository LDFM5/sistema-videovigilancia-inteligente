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
# VENTANA TEMPORAL
# =========================

WINDOW_SECONDS = 1.5          # Persistencia requerida en segundos
ACTIVATION_THRESHOLD = 25     # Detecciones necesarias para activar alerta


# =========================
# GRABACIÓN
# =========================

PRE_BUFFER_SECONDS = 10    # Segundos guardados ANTES de que se detecte el arma
POST_BUFFER_SECONDS = 15   # Segundos de grabación DESPUÉS de que el arma desaparece
RECORDING_FPS = 20

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
# TELEGRAM
# =========================

try:
    from config_local import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
except:
    TELEGRAM_TOKEN = None
    TELEGRAM_CHAT_ID = None
