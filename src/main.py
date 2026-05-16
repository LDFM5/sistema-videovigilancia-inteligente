"""
main.py

Sistema optimizado:
- Thread independiente por cámara
- Streamer no bloqueante
- Locks para evitar race conditions
- Arquitectura preparada para Edge AI
"""

import cv2
import atexit
import time
import threading

from visualization import draw_performance_overlay
import config

from cameras import initialize_cameras
from detection import (
    load_weapon_model,
    detect_weapons,
    load_pose_model,
    detect_pose
)

from temporal_logic import (
    initialize_windows,
    update_window
)

from recorder import (
    initialize_recording_state,
    handle_recording
)

from alerts import send_alert
from streamer import RTSPStreamer


# ==========================================================
# WORKER POR CÁMARA
# ==========================================================
def procesar_camara(
    cam_name,
    cap,
    weapon_model,
    pose_model,

    windows_armas,
    windows_asalto,
    windows_golpe,

    alert_state_armas,
    alert_state_asalto,
    alert_state_golpe,

    alert_lock,

    recording_state,
    recording_lock,
    inference_lock,

    streamer,
    shared_state,

    camera_resolutions
):

    tiempo_anterior = 0

    while True:

        # ==========================================
        # CAPTURA
        # ==========================================
        ret, frame = cap.read()

        if not ret:
            time.sleep(0.05)
            continue

        # ==========================================
        # FLAGS
        # ==========================================
        weapon_in_frame = False
        asalto_detectado = False
        golpe_detectado = False
        caida_detectada = False

        # ==========================================
        # DETECCIÓN ARMAS
        # ==========================================
        if config.ACTIVAR_MODELO_ARMAS and weapon_model is not None:

            with inference_lock:

                weapon_in_frame, frame = detect_weapons(
                    weapon_model,
                    frame,
                    config.CONF_WEAPON
                )


        # ==========================================
        # DETECCIÓN COMPORTAMIENTO
        # ==========================================
        if config.ACTIVAR_MODELO_COMPORTAMIENTO and pose_model is not None:

            with inference_lock:

                asalto_detectado, golpe_detectado, caida_detectada, frame = detect_pose(
                    pose_model,
                    frame,
                    0.5
                )


        # ==========================================
        # LÓGICA TEMPORAL
        # ==========================================
        with alert_lock:

            alerta_arma = update_window(
                cam_name,
                weapon_in_frame,
                windows_armas,
                config.ACTIVATION_THRESHOLD,
                alert_state_armas
            )

            alerta_asalto = update_window(
                cam_name,
                asalto_detectado,
                windows_asalto,
                config.BEHAVIOR_ACTIVATION_THRESHOLD,
                alert_state_asalto
            )

            alerta_caida = update_window(
                cam_name,
                caida_detectada,
                windows_golpe,
                config.GOLPE_ACTIVATION_THRESHOLD,
                alert_state_golpe
            )

            alerta_golpe = update_window(
                cam_name,
                golpe_detectado,
                windows_golpe,
                config.GOLPE_ACTIVATION_THRESHOLD,
                alert_state_golpe
            )

        # ==========================================
        # ALERTAS
        # ==========================================
        alert_triggered = (
            alerta_arma or
            alerta_asalto or
            alerta_golpe or
            alerta_caida
        )

        amenaza_presente = (
            weapon_in_frame or
            asalto_detectado or
            golpe_detectado or
            caida_detectada
        )

        # ==========================================
        # FPS REAL
        # ==========================================
        tiempo_actual = time.time()

        fps_real = (
            1.0 / (tiempo_actual - tiempo_anterior)
            if tiempo_anterior > 0 else 0.0
        )

        tiempo_anterior = tiempo_actual

        # ==========================================
        # OVERLAY
        # ==========================================
        frame = draw_performance_overlay(
            frame,
            fps_real
        )

        # ==========================================
        # GRABACIÓN
        # ==========================================
        with recording_lock:

            handle_recording(
                cam_name,
                frame,
                camera_resolutions,
                recording_state,
                config.POST_BUFFER_SECONDS,
                alert_triggered,
                amenaza_presente
            )

        # ==========================================
        # STREAM RTSP
        # ==========================================
        streamer.enviar_frame(frame)

        # ==========================================
        # CONFIG DINÁMICA
        # ==========================================
        with shared_state.lock:

            config.UMBRAL_VELOCIDAD_GOLPE = (
                shared_state.config_ram["UMBRAL_VELOCIDAD_GOLPE"]
            )

            config.UMBRAL_VELOCIDAD_CAIDA = (
                shared_state.config_ram["UMBRAL_VELOCIDAD_CAIDA"]
            )


# ==========================================================
# MAIN
# ==========================================================
def ejecutar_sistema_principal(shared_state):

    print("🔧 Inicializando sistema...")

    # ======================================================
    # MODELOS
    # ======================================================
    weapon_model = None

    if config.ACTIVAR_MODELO_ARMAS:

        print(" -> Cargando modelo ARMAS...")
        weapon_model = load_weapon_model()

    pose_model = None

    if config.ACTIVAR_MODELO_COMPORTAMIENTO:

        print(" -> Cargando modelo COMPORTAMIENTO...")
        pose_model = load_pose_model()

    # ======================================================
    # CÁMARAS
    # ======================================================
    cameras, camera_resolutions, camera_fps = initialize_cameras()

    # ======================================================
    # VENTANAS TEMPORALES
    # ======================================================
    windows_armas = initialize_windows(
        camera_fps,
        config.WINDOW_SECONDS
    )

    windows_asalto = initialize_windows(
        camera_fps,
        config.BEHAVIOR_WINDOW_SECONDS
    )

    windows_golpe = initialize_windows(
        camera_fps,
        config.BEHAVIOR_WINDOW_SECONDS
    )

    # ======================================================
    # ESTADOS ALERTAS
    # ======================================================
    alert_state_armas = {
        cam_name: False for cam_name in cameras
    }

    alert_state_asalto = {
        cam_name: False for cam_name in cameras
    }

    alert_state_golpe = {
        cam_name: False for cam_name in cameras
    }

    # ======================================================
    # LOCKS
    # ======================================================
    alert_lock = threading.Lock()
    recording_lock = threading.Lock()
    inference_lock = threading.Lock()


    # ======================================================
    # GRABACIÓN
    # ======================================================
    recording_state = initialize_recording_state(
        cameras,
        config.PRE_BUFFER_SECONDS
    )

    # ======================================================
    # STREAMERS RTSP
    # ======================================================
    streamers = {}

    for cam_name in cameras:

        w, h = camera_resolutions[cam_name]

        target_w = 800
        target_h = int((target_w / w) * h)

        if target_h % 2 != 0:
            target_h += 1

        streamers[cam_name] = RTSPStreamer(
            cam_name,
            width=target_w,
            height=target_h,
            fps=15
        )

    # ======================================================
    # LIMPIEZA SEGURA
    # ======================================================
    def limpieza_segura():

        print("\n🧹 Cerrando sistema...")

        for cap in cameras.values():
            cap.release()

        for streamer in streamers.values():
            streamer.cerrar()

        cv2.destroyAllWindows()

        print("✅ Sistema apagado correctamente.")

    atexit.register(limpieza_segura)

    # ======================================================
    # THREADS POR CÁMARA
    # ======================================================
    workers = []

    for cam_name, cap in cameras.items():

        worker = threading.Thread(
            target=procesar_camara,
            args=(

                cam_name,
                cap,

                weapon_model,
                pose_model,

                windows_armas,
                windows_asalto,
                windows_golpe,

                alert_state_armas,
                alert_state_asalto,
                alert_state_golpe,

                alert_lock,

                recording_state,
                recording_lock,
                inference_lock,

                streamers[cam_name],
                shared_state,

                camera_resolutions
            ),
            daemon=True
        )

        worker.start()

        workers.append(worker)

        print(f"✅ Worker iniciado -> {cam_name}")

    # ======================================================
    # LOOP PRINCIPAL VACÍO
    # ======================================================
    try: 
        while True: time.sleep(1) 
    except KeyboardInterrupt: 
        print("\n🛑 CTRL+C detectado...")


# ==========================================================
# ENTRY POINT
# ==========================================================
if __name__ == "__main__":

    class DummyState:
        def __init__(self):

            self.lock = threading.Lock()

            self.config_ram = {
                "UMBRAL_VELOCIDAD_GOLPE": 15.0,
                "UMBRAL_VELOCIDAD_CAIDA": 20.0
            }

    ejecutar_sistema_principal(DummyState())