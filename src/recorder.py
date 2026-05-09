"""
recorder.py

Módulo encargado de la grabación y compresión de evidencia en video.
"""
import os
import time
import cv2
import threading
from collections import deque
from config import EVIDENCE_DIR, RECORDING_FPS
from telegram_bot import send_video_sync

# =========================
# PROCESAMIENTO DE VIDEO
# =========================

def _comprimir_video(input_path, output_path):
    """Reduce resolución a 640x360 y FPS a 10 para hacer el video ligero."""
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    target_width, target_height = 640, 360
    out = cv2.VideoWriter(output_path, fourcc, 10, (target_width, target_height))
    
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if count % 2 == 0:
            frame_small = cv2.resize(frame, (target_width, target_height))
            out.write(frame_small)
        count += 1
        
    cap.release()
    out.release()
    return output_path

def _procesar_y_subir_evidencia(filepath, caption):
    """Hilo secundario: Comprime el video y lo sube a Telegram."""
    temp_path = filepath.replace(".mp4", "_lite.mp4")
    
    try:
        print("⚙️ Generando versión ligera para Telegram...")
        _comprimir_video(filepath, temp_path)
        
        tamaño_mb = os.path.getsize(temp_path) / (1024 * 1024)
        print(f"📊 Video comprimido: {tamaño_mb:.2f} MB. Subiendo a Telegram...")

        # Subir usando el bot (síncrono, pero ya estamos en un hilo secundario)
        send_video_sync(temp_path, caption)
        
        # Limpiar: borrar la copia ligera
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    except Exception as e:
        print(f"❌ Error procesando evidencia: {e}")

# =========================
# INICIALIZAR ESTADO
# =========================

def initialize_recording_state(cameras, pre_buffer_seconds):
    state = {}
    for cam_name in cameras:
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

def handle_recording(cam_name, frame, camera_resolutions, recording_state, post_buffer_seconds, alert_triggered, amenaza_presente):
    state = recording_state[cam_name]
    w, h = camera_resolutions[cam_name]
    frame_resized = cv2.resize(frame, (w, h))

    # ESTADO 1: NO ESTAMOS GRABANDO
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
            state["current_file"] = filename

            for b_frame in state["frame_buffer"]:
                writer.write(b_frame)
            state["frame_buffer"].clear()

    # ESTADO 2: ESTAMOS GRABANDO EL EVENTO
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

                    # ==========================================
                    # DISPARAR COMPRESIÓN Y ENVÍO EN SEGUNDO PLANO
                    # ==========================================
                    archivo_guardado = state["current_file"]
                    hilo_upload = threading.Thread(
                        target=_procesar_y_subir_evidencia, 
                        args=(archivo_guardado, "🎥 Evidencia capturada: Actividad Sospechosa")
                    )
                    hilo_upload.start()

                    print(f"✅ Evento finalizado. Clip guardado exitosamente ({cam_name})")