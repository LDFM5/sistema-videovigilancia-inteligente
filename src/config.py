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

RECORD_DURATION = 5  # Duración del clip en segundos


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

TELEGRAM_TOKEN = "8560474015:AAE7wU-1n-y3Z2gl5LmyKEy-IdFzlTDPPOA"
TELEGRAM_CHAT_ID = "7179295943"
