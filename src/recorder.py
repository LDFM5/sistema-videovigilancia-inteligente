"""
recorder.py

Módulo encargado de la grabación de evidencia en video.

Funciones principales:
- Inicializar el estado de grabación por cámara.
- Crear archivos de video cuando se activa una alerta.
- Escribir frames durante el periodo definido.
- Cerrar correctamente los archivos al finalizar la grabación.

Este módulo garantiza que los eventos detectados sean almacenados
como evidencia sin necesidad de grabación continua.
"""

import os
import time
import cv2
from config import EVIDENCE_DIR


# =========================
# INICIALIZAR ESTADO
# =========================

def initialize_recording_state(cameras):
    """
    Crea estructura de estado por cámara.
    """

    return {
        cam_name: {
            "recording": False,
            "start_time": None,
            "writer": None
        }
        for cam_name in cameras
    }


# =========================
# MANEJO DE GRABACIÓN
# =========================

def handle_recording(
    cam_name,
    frame,
    camera_resolutions,
    camera_fps,
    recording_state,
    record_duration,
    alert_triggered
):
    """
    Controla la grabación de clips por cámara.
    """

    state = recording_state[cam_name]

    # =========================
    # ACTIVAR GRABACIÓN
    # =========================
    if alert_triggered and not state["recording"]:

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(
            EVIDENCE_DIR,
            f"{cam_name}_{timestamp}.mp4"
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        w, h = camera_resolutions[cam_name]
        fps = camera_fps[cam_name]

        writer = cv2.VideoWriter(
            filename,
            fourcc,
            fps,
            (w, h)
        )

        if not writer.isOpened():
            print(f"❌ Error creando archivo para {cam_name}")
            return

        print(f"🎥 Grabando evidencia ({cam_name})")

        state["recording"] = True
        state["start_time"] = time.time()
        state["writer"] = writer

    # =========================
    # ESCRIBIR FRAME
    # =========================
    if state["recording"]:

        w, h = camera_resolutions[cam_name]
        frame_to_write = cv2.resize(frame, (w, h))

        state["writer"].write(frame_to_write)

        elapsed = time.time() - state["start_time"]

        if elapsed >= record_duration:
            print(f"✅ Clip guardado ({cam_name})")

            state["writer"].release()
            state["writer"] = None
            state["recording"] = False
            state["start_time"] = None
