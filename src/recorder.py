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
from collections import deque
from config import EVIDENCE_DIR, RECORDING_FPS

# =========================
# INICIALIZAR ESTADO
# =========================

def initialize_recording_state(cameras, pre_buffer_seconds):
    """
    Crea la estructura de estado por cámara, incluyendo el pre-buffer circular.
    """
    state = {}
    for cam_name in cameras:
        # Usamos el FPS real en lugar del teórico de la cámara
        buffer_size = int(RECORDING_FPS * pre_buffer_seconds)
        
        state[cam_name] = {
            "recording": False,
            "writer": None,
            "frame_buffer": deque(maxlen=buffer_size),
            "post_buffer_start_time": None
        }
    return state

# =========================
# MANEJO DE GRABACIÓN
# =========================

def handle_recording(
    cam_name,
    frame,
    camera_resolutions,
    recording_state,
    post_buffer_seconds,
    alert_triggered,
    amenaza_presente
):
    """
    Controla la grabación dinámica con pre-buffer y post-buffer preciso.
    """
    state = recording_state[cam_name]
    w, h = camera_resolutions[cam_name]
    frame_resized = cv2.resize(frame, (w, h))

    # =========================
    # ESTADO 1: NO ESTAMOS GRABANDO
    # =========================
    if not state["recording"]:
        state["frame_buffer"].append(frame_resized)

        if alert_triggered:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(EVIDENCE_DIR, f"{cam_name}_{timestamp}.mp4")
            
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            fps = RECORDING_FPS

            writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))

            if not writer.isOpened():
                print(f"❌ Error creando archivo para {cam_name}")
                return

            print(f"🎥 Alerta! Iniciando grabación ({cam_name}) - Volcando pre-buffer...")

            state["recording"] = True
            state["writer"] = writer
            state["post_buffer_start_time"] = None

            for b_frame in state["frame_buffer"]:
                writer.write(b_frame)
            state["frame_buffer"].clear()

    # =========================
    # ESTADO 2: ESTAMOS GRABANDO EL EVENTO
    # =========================
    else:
        state["writer"].write(frame_resized)

        # Usamos la detección directa de YOLO, NO la alerta temporal
        if amenaza_presente:
            # Mientras el arma esté en el frame actual, el post-buffer NO avanza
            state["post_buffer_start_time"] = None
        else:
            # El arma NO está en este frame exacto. Iniciamos o continuamos el conteo
            if state["post_buffer_start_time"] is None:
                state["post_buffer_start_time"] = time.time()
            else:
                elapsed = time.time() - state["post_buffer_start_time"]
                
                if elapsed >= post_buffer_seconds:
                    print(f"✅ Evento finalizado. Clip guardado exitosamente ({cam_name})")
                    
                    state["writer"].release()
                    state["writer"] = None
                    state["recording"] = False
                    state["post_buffer_start_time"] = None