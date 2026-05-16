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
import atexit  
import time

import time
from visualization import draw_performance_overlay  # Importamos el UI
import config

from cameras import initialize_cameras, read_frames
from detection import load_weapon_model, detect_weapons, load_pose_model, detect_pose
from temporal_logic import initialize_windows, update_window
from recorder import initialize_recording_state, handle_recording
from alerts import send_alert
from streamer import RTSPStreamer


def ejecutar_sistema_principal(shared_state):

    print("🔧 Inicializando sistema...")

    # =================================
    # 1. CARGA MODULAR DE MODELOS 
    # =================================
    weapon_model = None
    if config.ACTIVAR_MODELO_ARMAS:
        print(" -> Cargando modelo de detección de ARMAS...")
        weapon_model = load_weapon_model()

    pose_model = None
    if config.ACTIVAR_MODELO_COMPORTAMIENTO:
        print(" -> Cargando modelo de COMPORTAMIENTO (Postura)...")
        pose_model = load_pose_model()

    # =========================
    # Inicializar cámaras
    # =========================
    cameras, camera_resolutions, camera_fps = initialize_cameras()

    # =========================
    # Inicializar ventanas temporales (SEPARADAS)
    # =========================
    windows_armas = initialize_windows(camera_fps, config.WINDOW_SECONDS)
    windows_asalto = initialize_windows(camera_fps, config.BEHAVIOR_WINDOW_SECONDS)
    windows_golpe = initialize_windows(camera_fps, config.BEHAVIOR_WINDOW_SECONDS) # Usa la misma duración que asalto o crea una nueva

    alert_state_armas = {cam_name: False for cam_name in cameras}
    alert_state_asalto = {cam_name: False for cam_name in cameras}
    alert_state_golpe = {cam_name: False for cam_name in cameras}

    alertas_enviadas_evento = {cam_name: set() for cam_name in cameras}

    # =========================
    # Variables de rendimiento
    # =========================
    tiempo_anterior = 0

    # =========================
    # Estado de grabación (Actualizado)
    # =========================
    # Ahora pasamos los fps de las cámaras y el tiempo de pre-buffer
    recording_state = initialize_recording_state(cameras, config.PRE_BUFFER_SECONDS)

    # ==========================================
    # SEGURO DE LIMPIEZA (Se ejecuta al dar Ctrl+C)
    # ==========================================
    def limpieza_segura():
        print("\n🧹 Apagando sistema: Liberando cámaras y memoria...")
        for cap in cameras.values(): # O cap.release() si solo usas una variable
            cap.release()
        cv2.destroyAllWindows()
        print("✅ Sistema apagado correctamente.")

    # Registramos la función para que Python la ejecute justo antes de cerrarse
    atexit.register(limpieza_segura)

    # ==================================================
    # INICIALIZAR STREAMERS RTSP
    # ==================================================
    streamers = {}
    for cam_name in cameras:
        # 1. Obtener la resolución real de esta cámara
        w, h = camera_resolutions[cam_name]
        
        # 2. Definir un ancho estándar optimizado para la web
        target_w = 800 
        
        # 3. Calcular la altura para mantener la proporción exacta (ej. 16:9)
        target_h = int((target_w / w) * h)
        
        # 4. FFmpeg exige que las dimensiones sean números pares, si no, crashea.
        if target_h % 2 != 0:
            target_h += 1
            
        # Mandamos 15 FPS para que no pida más de lo que la IA puede darle (reduce el lag)
        streamers[cam_name] = RTSPStreamer(cam_name, width=target_w, height=target_h, fps=15)
    
    # =========================
    # LOOP PRINCIPAL
    # =========================
    while True:
        # 1. Leemos los frames de todas las cámaras activas
        # 'frames' es un diccionario: {"Camara 1": frame, "Camara 2": frame...}
        frames = read_frames(cameras)

        if not frames:
            break

        for cam_name, frame in frames.items():
            
            # CRÍTICO: Reiniciamos las alertas DENTRO del for para que
            # lo que detecte la Cámara 1 no se le "pegue" a la Cámara 2.
            weapon_in_frame = False
            asalto_detectado = False
            golpe_detectado = False
            caida_detectada = False

            # --- Módulo de Armas ---
            if config.ACTIVAR_MODELO_ARMAS and weapon_model is not None:
                weapon_in_frame, frame = detect_weapons(
                    weapon_model,
                    frame,
                    config.CONF_WEAPON
                )

            # --- Módulo de Comportamiento ---
            if config.ACTIVAR_MODELO_COMPORTAMIENTO and pose_model is not None:
                asalto_detectado, golpe_detectado, caida_detectada, frame = detect_pose(
                    pose_model, 
                    frame, 
                    0.5
                ) 

            # ==================================================
            # LÓGICA TEMPORAL
            # ==================================================
            alerta_arma = update_window(cam_name, weapon_in_frame, windows_armas, config.ACTIVATION_THRESHOLD, alert_state_armas)
            alerta_asalto = update_window(cam_name, asalto_detectado, windows_asalto, config.BEHAVIOR_ACTIVATION_THRESHOLD, alert_state_asalto)
            alerta_caida = update_window(cam_name, caida_detectada, windows_golpe, config.GOLPE_ACTIVATION_THRESHOLD, alert_state_golpe)
            alerta_golpe = update_window(cam_name, golpe_detectado, windows_golpe, config.GOLPE_ACTIVATION_THRESHOLD, alert_state_golpe)

            alert_triggered = alerta_arma or alerta_asalto or alerta_golpe or alerta_caida
            amenaza_presente = weapon_in_frame or asalto_detectado or golpe_detectado or caida_detectada

            # --- GESTIÓN DE ALERTAS (Telegram) ---
            if not recording_state[cam_name]["recording"]:
                alertas_enviadas_evento[cam_name].clear()
            
            if alerta_arma and "arma" not in alertas_enviadas_evento[cam_name]:
                send_alert(cam_name, "🔫 Arma detectada")
                alertas_enviadas_evento[cam_name].add("arma")
            # ... (repetir para asalto, golpe, caida) ...

            # ==========================================
            # MONITOREO DE RENDIMIENTO (Visualización)
            # ==========================================
            tiempo_actual = time.time()
            fps_real = 1.0 / (tiempo_actual - tiempo_anterior) if tiempo_anterior > 0 else 0.0
            tiempo_anterior = tiempo_actual

            # Dibujamos el overlay sobre el frame de esta cámara específica
            frame = draw_performance_overlay(frame, fps_real)

            # -------- Grabación --------
            handle_recording(
                cam_name, frame, camera_resolutions, recording_state,
                config.POST_BUFFER_SECONDS, alert_triggered, amenaza_presente
            )

            # ENVIAR AL SERVIDOR RTSP
            streamers[cam_name].enviar_frame(frame)

        # Guardamos la configuración
        with shared_state.lock:
            config.UMBRAL_VELOCIDAD_GOLPE = shared_state.config_ram["UMBRAL_VELOCIDAD_GOLPE"]
            config.UMBRAL_VELOCIDAD_CAIDA = shared_state.config_ram["UMBRAL_VELOCIDAD_CAIDA"]

# Esto evita que el código corra solo al ser importado por Flask
if __name__ == "__main__":
    # Si ejecutas main.py directamente, creamos un diccionario vacío y corre normal
    ejecutar_sistema_principal({'frame': None})