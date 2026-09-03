"""
temporal_logic.py

Módulo que implementa la lógica de persistencia temporal para reducir
falsos positivos en la detección de objetos de riesgo.

Funciones principales:
- Crear ventanas deslizantes basadas en el tiempo real de transmisión.
- Actualizar la ventana con cada fotograma procesado.
- Aplicar histéresis temporal para evitar parpadeos e impulsos falsos.
"""

from collections import deque
import time


# =========================
# INICIALIZAR VENTANAS
# =========================

def initialize_windows(camera_fps, window_seconds):
    """
    Crea una ventana deslizante por cámara basada en su FPS real.

    Retorna:
        detection_windows (dict): Diccionario indexado por nombre de cámara en mayúsculas.
    """
    detection_windows = {}

    for cam_name, fps in camera_fps.items():
        cam_upper = cam_name.upper()
        nominal_fps = max(1.0, float(fps))
        detection_windows[cam_upper] = {
            "events": deque(),
            "window_seconds": max(0.1, float(window_seconds)),
            "nominal_fps": nominal_fps,
        }
        print(
            f"[INFO] Ventana {cam_upper}: {window_seconds:.2f} segundos "
            f"(referencia {nominal_fps:.1f} FPS)."
        )

    return detection_windows


# =========================
# ACTUALIZAR VENTANA E HISTÉRESIS
# =========================

def update_window(
    cam_name,
    event_detected,
    detection_windows,
    activation_threshold,
    alert_state,
    deactivation_threshold=None,
    timestamp=None,
):
    """
    Actualiza la ventana temporal y decide si se genera un impulso de inicio de alerta (flanco de subida).

    Retorna:
        True  -> Si se debe disparar una nueva alerta (Rising Edge).
        False -> Si la alerta ya estaba activa o no alcanza el umbral.
    """
    cam_upper = cam_name.upper()
    window_state = detection_windows[cam_upper]
    events = window_state["events"]
    now = time.monotonic() if timestamp is None else float(timestamp)
    window_seconds = window_state["window_seconds"]
    nominal_fps = window_state["nominal_fps"]

    # Una pausa mayor que la propia ventana indica discontinuidad de cámara;
    # no se debe considerar que la última detección persistió durante la caída.
    if events and now - events[-1][0] > window_seconds:
        events.clear()

    events.append((now, bool(event_detected)))
    cutoff = now - window_seconds

    # Conservar como máximo un evento anterior al corte para conocer el estado
    # vigente exactamente al inicio de la ventana.
    while len(events) > 1 and events[1][0] <= cutoff:
        events.popleft()

    positive_seconds = 0.0
    events_list = list(events)
    for index, (event_time, detected) in enumerate(events_list):
        segment_start = max(cutoff, event_time)
        segment_end = (
            events_list[index + 1][0]
            if index + 1 < len(events_list)
            else now
        )
        if detected and segment_end > segment_start:
            positive_seconds += segment_end - segment_start

    # Si no se especifica el umbral de desactivación, se define al 50% del de activación (Histéresis)
    if deactivation_threshold is None:
        deactivation_threshold = max(1, activation_threshold // 2)

    activation_seconds = float(activation_threshold) / nominal_fps
    deactivation_seconds = float(deactivation_threshold) / nominal_fps

    # 2. FLANCO DE SUBIDA: Transición de inactivado (False) a activado (True)
    if positive_seconds >= activation_seconds and not alert_state[cam_upper]:
        alert_state[cam_upper] = True
        return True  # Impulso confirmatorio para iniciar grabación / dispatch

    # 3. HISTÉRESIS: Desactivar únicamente cuando la densidad de detecciones baje del umbral inferior
    if positive_seconds < deactivation_seconds:
        alert_state[cam_upper] = False

    return False
