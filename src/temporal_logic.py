"""
temporal_logic.py

Módulo que implementa la lógica de persistencia temporal para reducir
falsos positivos en la detección.

Funciones principales:
- Crear ventanas deslizantes basadas en tiempo real.
- Actualizar la ventana con cada frame procesado.
- Determinar cuándo una detección es suficientemente persistente
  para activar una alerta.

Este componente es clave para la robustez del sistema, ya que
filtra detecciones momentáneas e inestables.
"""

from collections import deque


# =========================
# INICIALIZAR VENTANAS
# =========================

def initialize_windows(camera_fps, window_seconds):
    """
    Crea una ventana deslizante por cámara basada en su FPS real.

    Retorna:
        detection_windows (dict)
    """

    detection_windows = {}

    for cam_name, fps in camera_fps.items():
        window_size = int(fps * window_seconds)
        detection_windows[cam_name] = deque(maxlen=window_size)
        print(f"⏱️ Ventana {cam_name}: {window_size} frames")

    return detection_windows


# =========================
# ACTUALIZAR VENTANA
# =========================

def update_window(
    cam_name,
    event_detected,      # <--- NUEVO: Nombre genérico
    detection_windows,
    activation_threshold,
    alert_state
):
    """
    Actualiza la ventana temporal y decide si se activa la alerta.

    Retorna:
        True  -> si se debe activar alerta
        False -> si no
    """

    window = detection_windows[cam_name]
    
    # Agregamos 1 si ocurrió el evento, 0 si no
    window.append(1 if event_detected else 0)

    detections_sum = sum(window)

    # Si supera umbral y antes no estaba activa → ACTIVAR
    if detections_sum >= activation_threshold and not alert_state[cam_name]:
        alert_state[cam_name] = True
        return True

    # Si baja demasiado → desactivar
    if detections_sum < activation_threshold:
        alert_state[cam_name] = False

    return False
