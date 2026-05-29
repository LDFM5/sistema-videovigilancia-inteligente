"""
recorder.py

Módulo de instrumentación industrial para la captura, compresión y persistencia de evidencia digital.
Optimizado para el volcado asíncrono de buffers volátiles pre/post-evento.
"""

import os
import time
import cv2
import threading
from collections import deque
from config import EVIDENCE_DIR, RECORDING_FPS
from telegram_bot import send_video_sync

# =========================================================================
# PROCESAMIENTO Y CODIFICACIÓN DE VIDEO
# =========================================================================

def _comprimir_video(input_path, output_path, shared_state):
    """
    Efectúa la transcodificación y reducción espacial/temporal del archivo binario.
    Normaliza la resolución a 640x360 y el muestreo a 10 FPS para la optimización del ancho de banda.
    """
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    target_width, target_height = 640, 360
    out = cv2.VideoWriter(output_path, fourcc, 10, (target_width, target_height))
    
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break
        
        if count % 2 == 0:
            frame_small = cv2.resize(frame, (target_width, target_height))
            out.write(frame_small)
        count += 1
        
    cap.release()
    out.release()
    return output_path


def _procesar_y_subir_evidencia(filepath, caption, shared_state):
    """
    Subproceso asíncrono: Ejecuta la compresión del contenedor MP4, evalúa la carga útil 
    técnica en memoria y gestiona la inyección asíncrona hacia los gateways de notificación.
    """
    temp_path = filepath.replace(".mp4", "_lite.mp4")
    
    try:
        if shared_state:
            shared_state.emitir_evento_dashboard('system_log', {
                "type": "info", 
                "message": "COMPRESOR_VIDEO: REDUCIENDO RESOLUCIÓN DE VIDEO PARA TRANSMISIÓN..."
            })
            
        _comprimir_video(filepath, temp_path, shared_state)
        
        size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        
        if shared_state:
            shared_state.emitir_evento_dashboard('system_log', {
                "type": "success", 
                "message": f"COMPRESOR_VIDEO: COMPRESIÓN EXITOSA ({size_mb:.2f} MB). INICIANDO ENVÍO..."
            })

        # Transmisión síncrona dentro de este hilo secundario aislado
        send_video_sync(temp_path, caption, shared_state=shared_state)
        
        # Liberación de memoria física en disco (Copia ligera temporal)
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    except Exception as e:
        if shared_state:
            shared_state.emitir_evento_dashboard('system_log', {
                "type": "error", 
                "message": f"ALMACENAMIENTO_ERROR: FALLO EN PROCESAMIENTO MULTIMEDIA -> REF: {str(e).upper()}"
            })


# =========================================================================
# GESTIÓN Y RESERVA DE ESTADOS VOLÁTILES
# =========================================================================

def initialize_recording_state(cameras, pre_buffer_seconds):
    """
    Aprovisiona de forma estricta los buffers de cola circular indexados para cada canal activo.
    """
    state = {}
    for cam_name in cameras:
        cam_upper = cam_name.upper()
        buffer_size = int(RECORDING_FPS * pre_buffer_seconds)
        state[cam_upper] = {
            "recording": False,
            "writer": None,
            "frame_buffer": deque(maxlen=buffer_size),
            "post_buffer_start_time": None,
            "current_file": None
        }
    return state


# =========================================================================
# CONTROLADOR OPERATIVO DE GRABACIÓN DE INCIDENTES
# =========================================================================

def handle_recording(cam_name, frame, camera_resolutions, recording_state, post_buffer_seconds, alert_triggered, amenaza_presente, shared_state=None):
    """
    Máquina de estados determinista para la persistencia de video de seguridad.
    Sincroniza el volcado FIFO (Pre-buffer) y la extensión por persistencia de riesgo (Post-buffer).
    """
    cam_upper = cam_name.upper()
    state = recording_state[cam_upper]
    w, h = camera_resolutions[cam_upper]
    frame_resized = cv2.resize(frame, (w, h))

    # ESTADO 1: MONITOREO PASIVO - ACUMULANDO BUFFER VOLÁTIL
    if not state["recording"]:
        state["frame_buffer"].append(frame_resized)

        if alert_triggered:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(EVIDENCE_DIR, f"{cam_upper}_{timestamp}.mp4")
            
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            fps = RECORDING_FPS
            writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))

            if not writer.isOpened():
                if shared_state:
                    shared_state.emitir_evento_dashboard('system_log', {
                        "type": "error", 
                        "message": f"ALMACENAMIENTO_ERROR: NO SE PUDO CREAR ARCHIVO DE VIDEO PARA: {cam_upper}"
                    })
                return

            if shared_state:
                shared_state.emitir_evento_dashboard('system_log', {
                    "type": "warn", 
                    "message": f"ALMACENAMIENTO: INTERRUPCIÓN POR ALERTA -> INICIANDO GRABACIÓN EN CANAL: {cam_upper}"
                })

            state["recording"] = True
            state["writer"] = writer
            state["post_buffer_start_time"] = None
            state["current_file"] = filename

            # Volcado inmediato de la traza histórica guardada en la cola circular
            for b_frame in state["frame_buffer"]:
                writer.write(b_frame)
            state["frame_buffer"].clear()

    # ESTADO 2: CAPTURA ACTIVA - GRABACIÓN EN CURSO
    else:
        state["writer"].write(frame_resized)

        if amenaza_presente:
            state["post_buffer_start_time"] = None
        else:
            if state["post_buffer_start_time"] is None:
                state["post_buffer_start_time"] = time.time()
            else:
                elapsed = time.time() - state["post_buffer_start_time"]
                
                if elapsed >= post_buffer_seconds:                    
                    state["writer"].release()
                    state["writer"] = None
                    state["recording"] = False
                    state["post_buffer_start_time"] = None

                    archivo_guardado = state["current_file"]
                    
                    # 🔹 REPORTE MULTIMEDIA SIMPLIFICADO Y ENTENDIBLE (ESPAÑOL)
                    telemetria_caption = (
                        f"📹 VIDEO DE EVIDENCIA\n"
                        f"──────────────────────\n"
                        f"CÁMARA: {cam_upper}\n"
                        f"EVENTO: ALERTA DE SEGURIDAD\n"
                        f"ESTADO: ARCHIVADO EN EL SISTEMA"
                    )
                    
                    hilo_upload = threading.Thread(
                        target=_procesar_y_subir_evidencia, 
                        args=(archivo_guardado, telemetria_caption, shared_state),
                        daemon=True,
                        name=f"Uploader-{cam_upper}"
                    )
                    hilo_upload.start()

                    if shared_state:
                        shared_state.emitir_evento_dashboard('system_log', {
                            "type": "success", 
                            "message": f"ALMACENAMIENTO: GRABACIÓN FINALIZADA. ARCHIVO GUARDADO EN DISCO: {cam_upper}"
                        })