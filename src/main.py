"""
main.py

Motor Principal de Inferencias (Edge AI Ready)
- Recolecta frames de los hilos de CameraStream.
- Ejecuta Inferencia por Lotes (Batching) sin locks.
- Centraliza la grabación, alertas y streaming RTSP/WebRTC.
"""

import cv2
import time
import atexit
from ultralytics import YOLO

# Importaciones de módulos
import config
from cameras import initialize_cameras
from detection import batch_detect_weapons, batch_detect_pose
from temporal_logic import initialize_windows, update_window
from recorder import initialize_recording_state, handle_recording
from alerts import send_alert
from streamer import RTSPStreamer
from visualization import draw_performance_overlay

from behavior import load_behavior_model, predict_behavior
from collections import deque


def ejecutar_sistema_principal(shared_state=None):
    print("🔧 Inicializando sistema Edge AI...")

    # ======================================================
    # 1. CARGA DE MODELOS (Se hace aquí, una sola vez)
    # ======================================================
    weapon_model = None
    if config.ACTIVAR_MODELO_ARMAS:
        print(" -> Cargando modelo de ARMAS...")
        weapon_model = YOLO(config.WEAPON_MODEL_PATH)
        print(" ✅ Modelo de ARMAS listo.")

    pose_model = None
    if config.ACTIVAR_MODELO_COMPORTAMIENTO:
        print(" -> Cargando modelo de POSE (YOLO11)...")
        pose_model = YOLO(config.POSE_MODEL_PATH)
        print(" ✅ Modelo de POSE listo.")
        print(" -> Cargando modelo de COMPORTAMIENTO (GRU)...")
        # Ajusta los parámetros según tu entrenamiento (ej. 15 frames de secuencia)
        SEQ_LENGTH = 16 
        CLASES_COMPORTAMIENTO = ["Normal", "Fighting"]
        #CLASES_COMPORTAMIENTO = ["Abuse", "Assault", "Burglary", "Fighting", "Normal", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"]
        gru_model, device = load_behavior_model(config.BEHAVIOR_MODEL_PATH)

    # ======================================================
    # 2. INICIALIZAR CÁMARAS (CameraStreamers en hilos)
    # ======================================================
    cameras, camera_resolutions, camera_fps = initialize_cameras()

    # ======================================================
    # 2.5 INICIALIZAR MEMORIA DE PERSONAS (Ahora sí conocemos las cámaras)
    # ======================================================
    if config.ACTIVAR_MODELO_COMPORTAMIENTO:
        # Estructura: { cam_name: { track_id: deque(maxlen=SEQ_LENGTH) } }
        historial_personas = {cam_name: {} for cam_name in cameras}

    # ======================================================
    # 3. ESTADOS Y VENTANAS TEMPORALES
    # ======================================================
    windows_armas = initialize_windows(camera_fps, config.WINDOW_SECONDS)
    
    alert_state_armas = {cam_name: False for cam_name in cameras}
    
    # Memoria para no spamear Telegram durante un mismo evento
    alertas_enviadas_evento = {cam_name: set() for cam_name in cameras}

    recording_state = initialize_recording_state(cameras, config.PRE_BUFFER_SECONDS)

    # ======================================================
    # 4. STREAMERS RTSP (MediaMTX)
    # ======================================================
    streamers = {}
    for cam_name in cameras:
        w, h = camera_resolutions[cam_name]
        target_w = 800
        target_h = int((target_w / w) * h)
        if target_h % 2 != 0: 
            target_h += 1

        streamers[cam_name] = RTSPStreamer(
            cam_name, width=target_w, height=target_h, fps=15
        )

    # ======================================================
    # 5. LIMPIEZA SEGURA AL CERRAR
    # ======================================================
    def limpieza_segura():
        print("\n🧹 Cerrando sistema y liberando recursos...")
        for cap in cameras.values():
            if hasattr(cap, 'release'): cap.release()
            elif hasattr(cap, 'stop'): cap.stop()
            
        for streamer in streamers.values():
            streamer.cerrar()
            
        cv2.destroyAllWindows()
        print("✅ Sistema apagado correctamente.")

    atexit.register(limpieza_segura)

    print("🚀 BUCLE PRINCIPAL INICIADO (Procesamiento por Lotes)")
    tiempo_anterior = time.time()

    # ======================================================
    # 6. BUCLE PRINCIPAL DE INFERENCIA
    # ======================================================
    while True:
        frames_list = []
        cam_names_list = []

        # A. Recolectar el último frame de cada cámara simultáneamente
        for cam_name, cap in cameras.items():
            # Compatible con CameraStream o cv2 estándar
            res = cap.read()
            frame = res[1] if isinstance(res, tuple) else res
            
            if frame is not None:
                frames_list.append(frame.copy())
                cam_names_list.append(cam_name)

        if not frames_list:
            time.sleep(0.01)
            continue

        # B. Cálculo de FPS del ciclo principal
        tiempo_actual = time.time()
        fps_real = 1.0 / (tiempo_actual - tiempo_anterior) if tiempo_anterior > 0 else 0.0
        tiempo_anterior = tiempo_actual

        # ==================================================
        # C. INFERENCIA POR LOTES (Sin Locks, Máxima Velocidad)
        # ==================================================
        weapon_results = []
        alertas_armas_batch = [] # NUEVA MEMORIA PARA LAS ALERTAS
        pose_results = []
        skeletons_data = []

        if config.ACTIVAR_MODELO_ARMAS and weapon_model:
            # Ahora recibimos dos variables de la función
            weapon_results, alertas_armas_batch = batch_detect_weapons(
                weapon_model, frames_list, conf=config.CONF_WEAPON, 
                clases_alerta=config.CLASES_ARMAS_ALERTA, modo_debug=config.MODO_DEBUG
            )

        if config.ACTIVAR_MODELO_COMPORTAMIENTO and pose_model:
            pose_results, skeletons_data = batch_detect_pose(
                pose_model, frames_list, conf=0.5
            )

        # ==================================================
        # D. PROCESAR RESULTADOS POR CÁMARA
        # ==================================================
        for i, cam_name in enumerate(cam_names_list):
            frame = frames_list[i]
            w_res = weapon_results[i] if i < len(weapon_results) else None
            p_res = pose_results[i] if i < len(pose_results) else None
            esqueletos_camara = skeletons_data[i] if i < len(skeletons_data) else None

            weapon_in_frame = False
            comportamiento_anomalo = False
            nombre_comportamiento = ""

            # --- DIBUJO Y LÓGICA DE ARMAS ---
            # Leemos si la función detectó un arma real en este frame específico
            weapon_in_frame = alertas_armas_batch[i] if i < len(alertas_armas_batch) else False

            if w_res and len(w_res.boxes) > 0:
                frame = w_res.plot(img=frame)

            # --- DIBUJO DE POSE Y ANÁLISIS DE COMPORTAMIENTO ---
            if p_res:
                frame = p_res.plot(img=frame)
                
                # Verificamos si YOLO Pose detectó personas y les asignó un ID
                if esqueletos_camara is not None and p_res.boxes.id is not None:
                    ids_personas = p_res.boxes.id.int().cpu().tolist()
                    
                    for i_persona, track_id in enumerate(ids_personas):
                        # 1. Extraer los datos crudos de esta persona
                        kp_persona = esqueletos_camara[i_persona] 
                        
                        # 2. Aplanar los datos: GRU espera un vector 1D por frame 
                        # (ej. 17 puntos * 2 coordenadas (x,y) = 34 valores)
                        # Solo tomamos x, y (ignoramos la confianza en la posición 2)
                        kp_aplanado = kp_persona[:, :2].flatten()
                        
                        # 3. Guardar en el historial de esta persona
                        if track_id not in historial_personas[cam_name]:
                            historial_personas[cam_name][track_id] = deque(maxlen=SEQ_LENGTH)
                        
                        historial_personas[cam_name][track_id].append(kp_aplanado)
                        
                        # 4. Inferencia: Solo predecimos si ya tenemos la secuencia completa (ej. 15 frames)
                        if len(historial_personas[cam_name][track_id]) == SEQ_LENGTH:
                            comportamiento = predict_behavior(
                                gru_model, device, 
                                historial_personas[cam_name][track_id], 
                                CLASES_COMPORTAMIENTO
                            )
                            
                            # 5. Si detecta algo anómalo, pintar en pantalla
                            if comportamiento != "Normal":
                                comportamiento_anomalo = True
                                nombre_comportamiento = comportamiento
                                cv2.putText(frame, f"ALERTA: {comportamiento}", (50, 100), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                                
                                
                    # ==========================================
                    # 🧹 LIMPIEZA DE MEMORIA (NUEVO)
                    # Borramos de la RAM a las personas que ya salieron de la cámara
                    # ==========================================
                    ids_a_borrar = [tid for tid in historial_personas[cam_name].keys() if tid not in ids_personas]
                    for tid in ids_a_borrar:
                        del historial_personas[cam_name][tid]

            # --- EVALUACIÓN TEMPORAL (Evitar falsos positivos) ---
            alerta_arma = update_window(
                cam_name, weapon_in_frame, windows_armas, 
                config.ACTIVATION_THRESHOLD, alert_state_armas
            )

            # --- ALERTAS DE TELEGRAM ---
            # Si no estamos grabando, limpiamos la memoria para que el próximo evento sí envíe mensaje
            if not recording_state[cam_name]["recording"]:
                alertas_enviadas_evento[cam_name].clear()
                
            # 1. Enviar mensaje si hay un arma
            if alerta_arma and "arma" not in alertas_enviadas_evento[cam_name]:
                send_alert(cam_name, "🔫 Arma detectada en cámara")
                alertas_enviadas_evento[cam_name].add("arma")

            # 2. Enviar mensaje si hay un comportamiento extraño (GRU)
            if comportamiento_anomalo and nombre_comportamiento not in alertas_enviadas_evento[cam_name]:
                # Dependiendo de tu clase, puedes ponerle emojis
                emoji = "🚨" if nombre_comportamiento == "Asalto" else "⚠️"
                send_alert(cam_name, f"{emoji} Comportamiento detectado: {nombre_comportamiento}")
                
                # Lo agregamos a la memoria para no enviarte 30 mensajes por segundo
                alertas_enviadas_evento[cam_name].add(nombre_comportamiento)

            alert_triggered = alerta_arma or comportamiento_anomalo
            amenaza_presente = weapon_in_frame or comportamiento_anomalo

            # --- UI WEB Y GRABACIÓN ---
            frame = draw_performance_overlay(frame, fps_real)

            handle_recording(
                cam_name, frame, camera_resolutions, recording_state,
                config.POST_BUFFER_SECONDS, alert_triggered, amenaza_presente
            )

            # --- ENVIAR A SERVIDOR DE VIDEO (MediaMTX) ---
            streamers[cam_name].enviar_frame(frame)


# ==========================================================
# PUNTO DE ENTRADA (Si se ejecuta directamente para pruebas)
# ==========================================================
if __name__ == "__main__":
    ejecutar_sistema_principal()