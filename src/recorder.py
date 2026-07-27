"""
recorder.py

Módulo de captura y persistencia de evidencia digital con temporización
estricta por delta de tiempo para garantizar velocidad de reproducción 1:1 real.
"""

import os
import time
import cv2
import threading
from collections import deque
from config import EVIDENCE_DIR, RECORDING_FPS
from telegram_bot import send_video_sync


def _comprimir_video(input_path, output_path, shared_state):
    """
    Transcodifica el video para reducir su tamaño manteniendo la tasa de tiempo real.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return output_path

    target_width, target_height = 640, 360
    fps_salida = max(1.0, RECORDING_FPS / 2.0)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps_salida, (target_width, target_height))

    try:
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            
            if count % 2 == 0:
                frame_small = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                out.write(frame_small)
            count += 1
    finally:
        cap.release()
        out.release()

    return output_path


def _procesar_y_subir_evidencia(filepath, caption, shared_state):
    base, ext = os.path.splitext(filepath)
    temp_path = f"{base}_lite{ext}"

    try:
        if shared_state:
            shared_state.emitir_evento_dashboard('system_log', {
                "type": "info", 
                "message": "COMPRESOR_VIDEO: PREPARANDO ARCHIVO PARA TRANSMISIÓN..."
            })

        _comprimir_video(filepath, temp_path, shared_state)

        if os.path.exists(temp_path):
            size_mb = os.path.getsize(temp_path) / (1024 * 1024)
            if shared_state:
                shared_state.emitir_evento_dashboard('system_log', {
                    "type": "success", 
                    "message": f"COMPRESOR_VIDEO: COMPRESIÓN EXITOSA ({size_mb:.2f} MB). ENVIANDO..."
                })

            send_video_sync(temp_path, caption, shared_state=shared_state)

    except Exception as e:
        if shared_state:
            shared_state.emitir_evento_dashboard('system_log', {
                "type": "error", 
                "message": f"ALMACENAMIENTO_ERROR: FALLO EN PROCESAMIENTO MULTIMEDIA -> REF: {str(e).upper()}"
            })
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def initialize_recording_state(cameras, pre_buffer_seconds):
    """
    Inicializa los buffers y los registros de marcas de tiempo por cámara.
    """
    state = {}
    for cam_name in cameras:
        cam_upper = cam_name.upper()
        buffer_size = max(1, int(RECORDING_FPS * pre_buffer_seconds))
        state[cam_upper] = {
            "recording": False,
            "writer": None,
            "frame_buffer": deque(maxlen=buffer_size),
            "post_buffer_start_time": None,
            "last_frame_time": 0.0,  # 🎯 Marca de tiempo para regular la tasa de escrituras
            "current_file": None,
            "lock": threading.Lock()
        }
    return state


def _volcar_buffer_asincrono(writer, buffer_frames, lock):
    """
    Escribe el pre-búfer de forma progresiva sin bloquear el hilo principal.
    """
    for b_frame in buffer_frames:
        with lock:
            if writer is not None and writer.isOpened():
                writer.write(b_frame)
        time.sleep(0.001)  # Micro-pausa para ceder el control del bus de CPU


def handle_recording(cam_name, frame, camera_resolutions, recording_state, post_buffer_seconds, alert_triggered, amenaza_presente, shared_state=None):
    """
    Máquina de estados determinista para la persistencia de video de seguridad.
    Garantiza una tasa constante de 15 FPS regulando el tiempo transcurrido
    y duplicando fotogramas si la IA sufre caídas de FPS.
    """
    cam_upper = cam_name.upper()
    state = recording_state[cam_upper]
    w, h = camera_resolutions[cam_upper]

    ahora = time.time()
    intervalo_objetivo = 1.0 / RECORDING_FPS  # Ej: 1 / 15 = 0.0666 segundos

    # Escalado a la resolución destino solo si es necesario
    if frame.shape[1] != w or frame.shape[0] != h:
        frame_resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        frame_resized = frame

    # =========================================================================
    # 🎯 CONTROL DE TEMPORIZACIÓN CON COMPENSACIÓN DE CAÍDAS DE FPS
    # =========================================================================
    time_elapsed = ahora - state["last_frame_time"]

    if time_elapsed >= intervalo_objetivo:
        # Calcular cuántos fotogramas de 15 FPS representan el tiempo real transcurrido
        cuadros_a_insertar = int(time_elapsed / intervalo_objetivo)
        # Limitar la repetición máxima a 3 cuadros para evitar acumulación si hay un congelamiento brusco
        cuadros_a_insertar = min(cuadros_a_insertar, 3)

        # ESTADO 1: MONITOREO PASIVO (Rellenar Pre-búfer)
        if not state["recording"]:
            for _ in range(cuadros_a_insertar):
                state["frame_buffer"].append(frame_resized)
            state["last_frame_time"] = ahora

            if alert_triggered:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(EVIDENCE_DIR, f"{cam_upper}_{timestamp}.mp4")

                fourcc = cv2.VideoWriter_fourcc(*"avc1")
                writer = cv2.VideoWriter(filename, fourcc, RECORDING_FPS, (w, h))

                if not writer.isOpened():
                    if shared_state:
                        shared_state.emitir_evento_dashboard('system_log', {
                            "type": "error", 
                            "message": f"ALMACENAMIENTO_ERROR: NO SE PUDO CREAR ARCHIVO DE VIDEO EN: {cam_upper}"
                        })
                    return

                if shared_state:
                    shared_state.emitir_evento_dashboard('system_log', {
                        "type": "warn", 
                        "message": f"ALMACENAMIENTO: ALERTA DETECTADA -> INICIANDO GRABACIÓN EN CANAL: {cam_upper}"
                    })

                state["recording"] = True
                state["writer"] = writer
                state["post_buffer_start_time"] = None
                state["current_file"] = filename
                state["last_frame_time"] = ahora

                # Volcado asíncrono del pre-búfer acumulado
                buffer_copy = list(state["frame_buffer"])
                state["frame_buffer"].clear()

                hilo_flush = threading.Thread(
                    target=_volcar_buffer_asincrono,
                    args=(writer, buffer_copy, state["lock"]),
                    daemon=True,
                    name=f"BufferFlush-{cam_upper}"
                )
                hilo_flush.start()

        # ESTADO 2: GRABACIÓN ACTIVA (Escribir en el archivo MP4)
        else:
            with state["lock"]:
                if state["writer"] is not None and state["writer"].isOpened():
                    for _ in range(cuadros_a_insertar):
                        state["writer"].write(frame_resized)
            state["last_frame_time"] = ahora

    # LÓGICA DE CONTROL DEL POST-BUFFER
    if state["recording"]:
        if amenaza_presente:
            state["post_buffer_start_time"] = None
        else:
            if state["post_buffer_start_time"] is None:
                state["post_buffer_start_time"] = ahora
            else:
                elapsed = ahora - state["post_buffer_start_time"]

                if elapsed >= post_buffer_seconds:
                    with state["lock"]:
                        if state["writer"] is not None:
                            state["writer"].release()
                            state["writer"] = None

                    state["recording"] = False
                    state["post_buffer_start_time"] = None

                    archivo_guardado = state["current_file"]

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
                            "message": f"ALMACENAMIENTO: GRABACIÓN COMPLETADA -> {cam_upper}"
                        })