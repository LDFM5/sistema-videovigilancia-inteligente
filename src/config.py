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
# MODO DESARROLLADOR
# =========================
# True: Muestra TODAS las clases (celulares, etc.) en pantalla para pruebas. 
#       (Nota: Las alertas/grabaciones seguirán disparándose SOLO con las armas reales).
# False: MODO PRODUCCIÓN. Solo dibuja las armas en pantalla. Ignora lo demás.
MODO_DEBUG = False

# =========================
# CONTROL DE MÓDULOS DE IA (FEATURE FLAGS)
# =========================
# Pon en 'False' el que no necesites para ahorrar RAM y CPU
ACTIVAR_MODELO_ARMAS = True
ACTIVAR_MODELO_COMPORTAMIENTO = True

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
WEAPON_MODEL_PATH = os.path.join(MODELS_DIR, "Modelo_objetos_sospechosos.pt")
POSE_MODEL_PATH = os.path.join(MODELS_DIR, "yolo11n-pose.pt")
BEHAVIOR_MODEL_PATH = os.path.join(MODELS_DIR, "behavior_gru.pth")


# =========================
# DETECCIÓN
# =========================

CONF_WEAPON = 0.5   # Umbral de confianza mínimo


# =========================
# VENTANAS TEMPORALES
# =========================
WINDOW_SECONDS = 1.5          
ACTIVATION_THRESHOLD = 15     

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
    "phone": 1,
    #"usb": 1
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
except:
    TELEGRAM_TOKEN = None
    TELEGRAM_CHAT_ID = None
