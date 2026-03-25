"""
cameras.py

Módulo responsable de la gestión de cámaras.

Funciones principales:
- Inicializar cámaras definidas en la configuración.
- Obtener resolución y FPS reales por cámara.
- Leer frames en cada iteración del sistema.

Permite soportar múltiples cámaras de forma independiente y
mantener el sistema robusto ante desconexiones o errores de captura.
"""

import cv2
from config import CAMERA_INDEXES


# =========================
# INICIALIZAR CÁMARAS
# =========================

def initialize_cameras():
    """
    Inicializa todas las cámaras definidas en config.
    
    Retorna:
        cameras (dict)
        camera_resolutions (dict)
        camera_fps (dict)
    """

    cameras = {}
    camera_resolutions = {}
    camera_fps = {}

    for cam_name, cam_index in CAMERA_INDEXES.items():

        cap = cv2.VideoCapture(cam_index)

        if not cap.isOpened():
            print(f"❌ No se pudo abrir la cámara: {cam_name}")
            continue

        # ==================================================
        # NUEVO: Forzar resolución panorámica (Alta calidad)
        # ==================================================
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)   # Ancho deseado
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)   # Alto deseado
        # (Si tu cámara soporta 1080p, puedes intentar 1920 y 1080)
        # ==================================================

        cameras[cam_name] = cap

        # Obtener la resolución REAL que la cámara aceptó darnos
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        camera_resolutions[cam_name] = (width, height)

        # FPS real
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 1:  # algunas cámaras reportan 0 o 1
            fps = 30

        camera_fps[cam_name] = fps

        print(f"📷 {cam_name} resolución: {width}x{height}")
        print(f"🎥 {cam_name} FPS: {fps}")

    return cameras, camera_resolutions, camera_fps


# =========================
# LEER FRAMES
# =========================

def read_frames(cameras):
    """
    Lee un frame de cada cámara activa.

    Retorna:
        frames (dict) con {nombre_camara: frame}
    """

    frames = {}

    for cam_name, cap in cameras.items():
        ret, frame = cap.read()
        if ret:
            frames[cam_name] = frame
        else:
            print(f"⚠️ No se pudo leer frame de {cam_name}")

    return frames
