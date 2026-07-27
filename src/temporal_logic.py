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
        # Garantizar que la ventana contenga al menos 1 fotograma
        window_size = max(1, int(fps * window_seconds))
        detection_windows[cam_upper] = deque(maxlen=window_size)
        print(f"⏱️ Ventana {cam_upper}: {window_size} frames asignados.")

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
    deactivation_threshold=None
):
    """
    Actualiza la ventana temporal y decide si se genera un impulso de inicio de alerta (flanco de subida).

    Retorna:
        True  -> Si se debe disparar una nueva alerta (Rising Edge).
        False -> Si la alerta ya estaba activa o no alcanza el umbral.
    """
    cam_upper = cam_name.upper()
    window = detection_windows[cam_upper]
    
    # 1. Registrar evento binario en la ventana deslizante
    window.append(1 if event_detected else 0)

    detections_sum = sum(window)

    # Si no se especifica el umbral de desactivación, se define al 50% del de activación (Histéresis)
    if deactivation_threshold is None:
        deactivation_threshold = max(1, activation_threshold // 2)

    # 2. FLANCO DE SUBIDA: Transición de inactivado (False) a activado (True)
    if detections_sum >= activation_threshold and not alert_state[cam_upper]:
        alert_state[cam_upper] = True
        return True  # Impulso confirmatorio para iniciar grabación / dispatch

    # 3. HISTÉRESIS: Desactivar únicamente cuando la densidad de detecciones baje del umbral inferior
    if detections_sum < deactivation_threshold:
        alert_state[cam_upper] = False

    return False