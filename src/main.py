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
    
    # =========================
    # LOOP PRINCIPAL
    # =========================
    while True:

        frames = read_frames(cameras)

        if not frames:
            break

        # Inicializamos variables de alerta en False por defecto
        weapon_in_frame = False
        asalto_detectado = False
        golpe_detectado = False
        caida_detectada = False

        for cam_name, frame in frames.items():

            # --- Módulo de Armas ---
            if config.ACTIVAR_MODELO_ARMAS and weapon_model is not None:
                weapon_in_frame, frame = detect_weapons(
                    weapon_model,
                    frame,
                    config.CONF_WEAPON
                )

            frame = frame.copy()

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
            
            # Usamos el umbral casi inmediato para los golpes
            alerta_golpe = update_window(
                cam_name, 
                golpe_detectado, 
                windows_golpe, 
                config.GOLPE_ACTIVATION_THRESHOLD, # <--- ¡Cambiamos esto!
                alert_state_golpe
            )

            alert_triggered = alerta_arma or alerta_asalto or alerta_golpe or alerta_caida
            amenaza_presente = weapon_in_frame or asalto_detectado or golpe_detectado or caida_detectada

            # ==================================================
            # GESTIÓN INTELIGENTE DE ALERTAS
            # ==================================================
            # 1. Si NO estamos grabando, significa que no hay evento activo. Limpiamos la memoria de alertas.
            if not recording_state[cam_name]["recording"]:
                alertas_enviadas_evento[cam_name].clear()

            # 2. Evaluamos cada amenaza de forma independiente
            # Si hay alerta, Y esa alerta NO se ha enviado en esta grabación, la enviamos y la registramos.
            
            if alerta_arma and "arma" not in alertas_enviadas_evento[cam_name]:
                send_alert(cam_name, "🔫 Arma detectada en cámara")
                alertas_enviadas_evento[cam_name].add("arma")

            if alerta_asalto and "asalto" not in alertas_enviadas_evento[cam_name]:
                send_alert(cam_name, "✋ Posible Asalto (Manos arriba prolongado)")
                alertas_enviadas_evento[cam_name].add("asalto")

            if alerta_golpe and "golpe" not in alertas_enviadas_evento[cam_name]:
                send_alert(cam_name, "🥊 Agresión física / Movimiento brusco detectado")
                alertas_enviadas_evento[cam_name].add("golpe")

            if alerta_caida and "caida" not in alertas_enviadas_evento[cam_name]:
                send_alert(cam_name, "⚠️ Hombre caído / Desplome detectado")
                alertas_enviadas_evento[cam_name].add("caida")

            # ==========================================
            # MONITOREO DE RENDIMIENTO
            # ==========================================
            # Calcular FPS reales
            tiempo_actual = time.time()
            fps_real = 1.0 / (tiempo_actual - tiempo_anterior) if tiempo_anterior > 0 else 0.0
            tiempo_anterior = tiempo_actual

            # Dibujar el rectángulo gris con los datos en el frame
            frame = draw_performance_overlay(frame, fps_real)

            # -------- Grabación --------
            handle_recording(
                cam_name,
                frame,
                camera_resolutions,
                recording_state,
                config.POST_BUFFER_SECONDS, # Pasamos el tiempo de post-buffer
                alert_triggered,
                amenaza_presente
            )


            # cv2.imshow(f"Camera: {cam_name}", frame)

        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #    break

        # ==========================================
        # GUARDAR EN LA RAM PARA FLASK (CORREGIDO)
        # ==========================================
        with shared_state.lock:
            # 1. Leemos los valores en vivo de la RAM
            config.UMBRAL_VELOCIDAD_GOLPE = shared_state.config_ram["UMBRAL_VELOCIDAD_GOLPE"]
            config.UMBRAL_VELOCIDAD_CAIDA = shared_state.config_ram["UMBRAL_VELOCIDAD_CAIDA"]
            
            # 2. Guardamos el frame y el estado de grabación usando "puntos"
            shared_state.frame = frame.copy() 
            # Si necesitas saber si está grabando en el futuro, puedes agregarlo así:
            # shared_state.grabando = recording_state[cam_name]["recording"]

# Esto evita que el código corra solo al ser importado por Flask
if __name__ == "__main__":
    # Si ejecutas main.py directamente, creamos un diccionario vacío y corre normal
    ejecutar_sistema_principal({'frame': None})