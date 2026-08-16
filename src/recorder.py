"""
recorder.py

Módulo de captura y persistencia de evidencia digital con temporización
estricta por delta de tiempo para garantizar velocidad de reproducción 1:1 real.
"""

import os
import time
import cv2
import threading
import queue
import shutil
import subprocess
from collections import deque
from config import EVIDENCE_DIR, RECORDING_FPS
from telegram_bot import send_video_sync


_EVIDENCE_QUEUE = queue.Queue(maxsize=16)
_EVIDENCE_WORKERS_LOCK = threading.Lock()
_EVIDENCE_WORKERS_STARTED = False
_FFMPEG_ENCODER = None
_FFMPEG_ENCODER_LOCK = threading.Lock()


def _detectar_encoder_ffmpeg():
    """Obtiene codificadores H.264 disponibles, priorizando hardware."""
    global _FFMPEG_ENCODER

    with _FFMPEG_ENCODER_LOCK:
        if _FFMPEG_ENCODER is not None:
            return _FFMPEG_ENCODER

        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            _FFMPEG_ENCODER = False
            return None

        try:
            result = subprocess.run(
                [ffmpeg_path, "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            encoders = result.stdout
            candidates = tuple(
                candidate
                for candidate in ("h264_nvenc", "h264_qsv", "h264_amf")
                if candidate in encoders
            )
            _FFMPEG_ENCODER = candidates + ("libx264",)
            return _FFMPEG_ENCODER
        except Exception:
            pass

        _FFMPEG_ENCODER = ("libx264",)
        return _FFMPEG_ENCODER


def _comprimir_video(input_path, output_path, shared_state):
    """
    Transcodifica el video para reducir su tamaño manteniendo la tasa de tiempo real.
    """
    target_width, target_height = 640, 360
    fps_salida = max(1.0, RECORDING_FPS / 2.0)

    ffmpeg_path = shutil.which("ffmpeg")
    encoder_candidates = _detectar_encoder_ffmpeg()
    if ffmpeg_path and encoder_candidates:
        for encoder_name in encoder_candidates:
            command = [
                ffmpeg_path,
                "-y",
                "-hide_banner",
                "-loglevel", "error",
                "-i", input_path,
                "-vf", f"scale={target_width}:{target_height}:flags=fast_bilinear,fps={fps_salida}",
                "-an",
                "-c:v", encoder_name,
                "-b:v", "700k",
                "-maxrate", "900k",
                "-bufsize", "1400k",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                output_path,
            ]
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    timeout=300,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                if (
                    result.returncode == 0
                    and os.path.exists(output_path)
                    and os.path.getsize(output_path) > 0
                ):
                    # Recordar el encoder que realmente abrió el dispositivo;
                    # las siguientes evidencias no repetirán intentos fallidos.
                    global _FFMPEG_ENCODER
                    _FFMPEG_ENCODER = (encoder_name,)
                    return output_path
            except Exception:
                pass

            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass

    # Fallback compatible cuando FFmpeg no está instalado o el codificador de
    # hardware anunciado no puede abrir el dispositivo.
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return output_path

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


def _evidence_worker():
    while True:
        filepath, caption, shared_state = _EVIDENCE_QUEUE.get()
        try:
            _procesar_y_subir_evidencia(filepath, caption, shared_state)
        finally:
            _EVIDENCE_QUEUE.task_done()


def _iniciar_workers_evidencia():
    global _EVIDENCE_WORKERS_STARTED

    with _EVIDENCE_WORKERS_LOCK:
        if _EVIDENCE_WORKERS_STARTED:
            return

        for index in range(2):
            threading.Thread(
                target=_evidence_worker,
                daemon=True,
                name=f"EvidenceWorker-{index + 1}",
            ).start()
        _EVIDENCE_WORKERS_STARTED = True


def _encolar_evidencia(filepath, caption, shared_state):
    _iniciar_workers_evidencia()
    try:
        _EVIDENCE_QUEUE.put_nowait((filepath, caption, shared_state))
        return True
    except queue.Full:
        if shared_state:
            shared_state.emitir_evento_dashboard('system_log', {
                "type": "error",
                "message": "COLA_EVIDENCIA: CAPACIDAD AGOTADA; EL VIDEO QUEDÓ GUARDADO LOCALMENTE.",
            })
        return False


def initialize_recording_state(cameras, pre_buffer_seconds):
    """
    Inicializa los buffers y los registros de marcas de tiempo por cámara.

    El pre-búfer se limita por tiempo real y no por una cantidad estimada de
    fotogramas. Esto evita que su duración cambie cuando el ciclo de inferencia
    trabaja a una tasa distinta de RECORDING_FPS.
    """
    state = {}
    for cam_name in cameras:
        cam_upper = cam_name.upper()
        state[cam_upper] = {
            "recording": False,
            "writer": None,
            "frame_buffer": deque(),
            "pre_buffer_seconds": max(0.0, float(pre_buffer_seconds)),
            "next_prebuffer_time": None,
            "post_buffer_start_time": None,
            "next_frame_time": None,
            "last_recorded_frame": None,
            "current_file": None,
            "lock": threading.Lock()
        }
    return state


def _agregar_al_prebuffer(state, timestamp, frame, force=False):
    """
    Conserva frames con su instante de captura y elimina los que ya quedaron
    fuera de la ventana temporal configurada.
    """
    interval = 1.0 / RECORDING_FPS
    next_sample_time = state["next_prebuffer_time"]

    if next_sample_time is None:
        next_sample_time = timestamp

    if timestamp >= next_sample_time:
        state["frame_buffer"].append((timestamp, frame.copy()))
        while next_sample_time <= timestamp:
            next_sample_time += interval
        state["next_prebuffer_time"] = next_sample_time
    elif force:
        # El frame que dispara la alerta debe quedar incluido aunque haya
        # llegado entre dos muestras regulares del pre-búfer.
        state["frame_buffer"].append((timestamp, frame.copy()))

    cutoff = timestamp - state["pre_buffer_seconds"]

    while state["frame_buffer"] and state["frame_buffer"][0][0] < cutoff:
        state["frame_buffer"].popleft()


def _volcar_buffer_ordenado(writer, timed_frames, fps):
    """
    Convierte frames con timestamps variables a una secuencia CFR ordenada.

    Para cada instante de salida toma el frame más reciente disponible. Todo el
    pre-búfer se escribe antes de aceptar video nuevo, por lo que nunca puede
    intercalarse el pasado con el presente.
    """
    if not timed_frames:
        return None, None

    interval = 1.0 / fps
    next_frame_time = timed_frames[0][0]
    last_timestamp = timed_frames[-1][0]
    source_index = 0
    source_frame = timed_frames[0][1]

    while next_frame_time <= last_timestamp:
        while (
            source_index + 1 < len(timed_frames)
            and timed_frames[source_index + 1][0] <= next_frame_time
        ):
            source_index += 1
            source_frame = timed_frames[source_index][1]

        writer.write(source_frame)
        next_frame_time += interval

    return next_frame_time, timed_frames[-1][1]


def _escribir_hasta_timestamp(writer, state, timestamp):
    """Rellena la línea de tiempo CFR hasta el instante del frame actual."""
    next_frame_time = state["next_frame_time"]
    previous_frame = state["last_recorded_frame"]

    if next_frame_time is None or previous_frame is None:
        return

    interval = 1.0 / RECORDING_FPS

    # El frame actual aún no existía en estos instantes. Repetir el último frame
    # conocido conserva la duración real cuando la inferencia pierde fluidez.
    while next_frame_time <= timestamp:
        writer.write(previous_frame)
        next_frame_time += interval

    state["next_frame_time"] = next_frame_time


def handle_recording(cam_name, frame, camera_resolutions, recording_state, post_buffer_seconds, alert_triggered, amenaza_presente, shared_state=None, pre_buffer_seconds=None, frame_timestamp=None):
    """
    Máquina de estados para persistencia de video con una línea de tiempo CFR.

    La duración de salida se deriva de tiempo monotónico: si la IA procesa más
    lento se repite el último frame y, si procesa más rápido, se omiten los
    frames que caen entre dos instantes de salida. De este modo la reproducción
    conserva velocidad 1:1 aunque la tasa de entrada varíe.
    """
    cam_upper = cam_name.upper()
    state = recording_state[cam_upper]
    w, h = camera_resolutions[cam_upper]

    ahora = time.monotonic() if frame_timestamp is None else float(frame_timestamp)

    if pre_buffer_seconds is not None:
        state["pre_buffer_seconds"] = max(0.0, float(pre_buffer_seconds))

    # Escalado a la resolución destino solo si es necesario
    if frame.shape[1] != w or frame.shape[0] != h:
        frame_resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        frame_resized = frame

    # ESTADO 1: MONITOREO PASIVO (pre-búfer basado en segundos reales)
    if not state["recording"]:
        _agregar_al_prebuffer(
            state, ahora, frame_resized, force=bool(alert_triggered)
        )

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

            buffer_copy = list(state["frame_buffer"])
            state["frame_buffer"].clear()

            # Operación deliberadamente síncrona: garantiza que ningún frame
            # posterior a la alerta se inserte entre frames del pre-búfer.
            with state["lock"]:
                next_frame_time, last_frame = _volcar_buffer_ordenado(
                    writer, buffer_copy, RECORDING_FPS
                )

            state["recording"] = True
            state["writer"] = writer
            state["post_buffer_start_time"] = None
            state["current_file"] = filename
            state["next_frame_time"] = next_frame_time
            state["last_recorded_frame"] = last_frame

    # ESTADO 2: GRABACIÓN ACTIVA
    else:
        with state["lock"]:
            writer = state["writer"]
            if writer is not None and writer.isOpened():
                _escribir_hasta_timestamp(writer, state, ahora)
                state["last_recorded_frame"] = frame_resized.copy()

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
                    state["next_frame_time"] = None
                    state["last_recorded_frame"] = None
                    state["next_prebuffer_time"] = None

                    archivo_guardado = state["current_file"]

                    telemetria_caption = (
                        f"📹 VIDEO DE EVIDENCIA\n"
                        f"──────────────────────\n"
                        f"CÁMARA: {cam_upper}\n"
                        f"EVENTO: ALERTA DE SEGURIDAD\n"
                        f"ESTADO: ARCHIVADO EN EL SISTEMA"
                    )

                    _encolar_evidencia(
                        archivo_guardado, telemetria_caption, shared_state
                    )

                    if shared_state:
                        shared_state.emitir_evento_dashboard('system_log', {
                            "type": "success", 
                            "message": f"ALMACENAMIENTO: GRABACIÓN COMPLETADA -> {cam_upper}"
                        })
