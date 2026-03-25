"""
main.py

Archivo principal del sistema de detección de comportamiento sospechoso.

Este módulo actúa como orquestador del sistema, coordinando los diferentes
componentes modulares:

- Inicialización de cámaras
- Carga del modelo de detección
- Ejecución del ciclo principal de procesamiento
- Aplicación de la lógica temporal
- Activación de grabación de evidencia
- Envío de alertas

No contiene lógica interna de detección, grabación o notificación,
sino que conecta los módulos especializados del sistema.
"""

import cv2
# Importa las nuevas variables (Quita RECORD_DURATION)
from config import (
    CONF_WEAPON,
    WINDOW_SECONDS,
    ACTIVATION_THRESHOLD,
    PRE_BUFFER_SECONDS,
    POST_BUFFER_SECONDS
)

from cameras import initialize_cameras, read_frames
from detection import load_weapon_model, detect_weapons
from temporal_logic import initialize_windows, update_window
from recorder import initialize_recording_state, handle_recording
from alerts import send_alert


def main():

    print("🔧 Inicializando sistema...")

    # =========================
    # Cargar modelo
    # =========================
    weapon_model = load_weapon_model()

    # =========================
    # Inicializar cámaras
    # =========================
    cameras, camera_resolutions, camera_fps = initialize_cameras()

    # =========================
    # Inicializar ventanas temporales
    # =========================
    detection_windows = initialize_windows(
        camera_fps, WINDOW_SECONDS
    )

    alert_state = {cam_name: False for cam_name in cameras}

    # =========================
    # Estado de grabación (Actualizado)
    # =========================
    # Ahora pasamos los fps de las cámaras y el tiempo de pre-buffer
    recording_state = initialize_recording_state(cameras, camera_fps, PRE_BUFFER_SECONDS)

    print("▶ Sistema iniciado. Presiona 'q' para salir.")

    # =========================
    # LOOP PRINCIPAL
    # =========================
    while True:

        frames = read_frames(cameras)

        for cam_name, frame in frames.items():

            # -------- Detección --------
            weapon_in_frame, frame = detect_weapons(
                weapon_model,
                frame,
                CONF_WEAPON
            )

            # -------- Lógica temporal --------
            alert_triggered = update_window(
                cam_name,
                weapon_in_frame,
                detection_windows,
                ACTIVATION_THRESHOLD,
                alert_state   
            )


            # -------- Activar alerta --------
            # Comprobamos si no estábamos ya grabando este mismo evento
            if alert_triggered and not recording_state[cam_name]["recording"]:
                send_alert(cam_name)

            # -------- Grabación (Actualizado) --------
            handle_recording(
                cam_name,
                frame,
                camera_resolutions,
                camera_fps,
                recording_state,
                POST_BUFFER_SECONDS, # Pasamos el tiempo de post-buffer
                alert_triggered,
                weapon_in_frame
            )

            cv2.imshow(f"Camera: {cam_name}", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # =========================
    # LIMPIEZA
    # =========================
    for cap in cameras.values():
        cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
