"""
main.py

Motor Principal de Inferencias (Edge AI Core)
- Recolecta frames de los hilos asincronos de CameraStream.
- Ejecuta Inferencia por Lotes (Batching) a nivel de hardware.
- Centraliza la grabacion automatizada, alertas y streaming RTSP/WebRTC.
"""

import cv2
import time
import atexit
from ultralytics import YOLO

# Importaciones de modulos del ecosistema core
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
    print("STATUS: Inicializando modulo core de inferencias Edge AI...")

    # ======================================================
    # 1. CARGA DE MODELOS EN ACELERADORES DE HARDWARE
    # ======================================================
    weapon_model = None
    if config.ACTIVAR_MODELO_ARMAS:
        print("INFO: Cargando red neuronal de deteccion de armamento...")
        weapon_model = YOLO(config.WEAPON_MODEL_PATH)
        print("SUCCESS: Pipeline de deteccion de armamento operativo.")

    pose_model = None
    if config.ACTIVAR_MODELO_COMPORTAMIENTO:
        print("INFO: Cargando red neuronal de estimacion de pose estructural...")
        pose_model = YOLO(config.POSE_MODEL_PATH)
        print("SUCCESS: Pipeline de estimacion de pose operativo.")
        
        print("INFO: Cargando clasificador secuencial de comportamiento (GRU)...")
        SEQ_LENGTH = 16 
        CLASES_COMPORTAMIENTO = ["Normal", "Fighting"]
        gru_model, device = load_behavior_model(config.BEHAVIOR_MODEL_PATH)
        print("SUCCESS: Clasificador secuencial GRU asignado a dispositivo core.")

    # ======================================================
    # 2. INTERCONEXIÓN DE DISPOSITIVOS DE CAPTURA
    # ======================================================
    cameras, camera_resolutions, camera_fps = initialize_cameras()

    # ======================================================
    # 2.5 ASIGNACIÓN DE REGISTROS DE MEMORIA VOLÁTIL
    # ======================================================
    if config.ACTIVAR_MODELO_COMPORTAMIENTO:
        # Estructura: { cam_name: { track_id: deque(maxlen=SEQ_LENGTH) } }
        historial_personas = {cam_name: {} for cam_name in cameras}

    # ======================================================
    # 3. FILTROS DE MITIGACIÓN Y VENTANAS TEMPORALES
    # ======================================================
    windows_armas = initialize_windows(camera_fps, config.WINDOW_SECONDS)
    alert_state_armas = {cam_name: False for cam_name in cameras}
    
    # Registro de supresion de redundancia para notificaciones externas
    alertas_enviadas_evento = {cam_name: set() for cam_name in cameras}

    recording_state = initialize_recording_state(cameras, config.PRE_BUFFER_SECONDS)

    # ======================================================
    # 4. INSTANCIACIÓN DE CANALES RTSP (MediaMTX OUTBOUND)
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
    # 5. PROTOCOLO DE DESCONEXIÓN SEGURA (CLEANUP)
    # ======================================================
    def limpieza_segura():
        print("\nINFO: Solicitando detencion general de hilos y liberacion de hardware...")
        for cap in cameras.values():
            if hasattr(cap, 'release'): cap.release()
            elif hasattr(cap, 'stop'): cap.stop()
            
        for streamer in streamers.values():
            streamer.cerrar()
            
        cv2.destroyAllWindows()
        print("SUCCESS: Todos los recursos de hardware han sido liberados de forma segura.")

    atexit.register(limpieza_segura)

    print("INFO: Bucle de ejecucion continua inicializado (Modo Batch Activo)")
    tiempo_anterior = time.time()

    # ======================================================
    # 6. BUCLE PRINCIPAL DE INFERENCIA
    # ======================================================
    while True:
        frames_list = []
        cam_names_list = []

        # A. Captura paralela de frames desde buffers de memoria
        for cam_name, cap in cameras.items():
            res = cap.read()
            frame = res[1] if isinstance(res, tuple) else res
            
            if frame is not None:
                frames_list.append(frame.copy())
                cam_names_list.append(cam_name)

        if not frames_list:
            time.sleep(0.01)
            continue

        # B. Metricas de rendimiento del ciclo core
        tiempo_actual = time.time()
        fps_real = 1.0 / (tiempo_actual - tiempo_anterior) if tiempo_anterior > 0 else 0.0
        tiempo_anterior = tiempo_actual

        # ==================================================
        # C. PROCESAMIENTO MATRICIAL EN LOTE (BATCH INFERENCE)
        # ==================================================
        weapon_results = []
        alertas_armas_batch = [] 
        pose_results = []
        skeletons_data = []

        if config.ACTIVAR_MODELO_ARMAS and weapon_model:
            weapon_results, alertas_armas_batch = batch_detect_weapons(
                weapon_model, frames_list, conf=config.CONF_WEAPON, 
                clases_alerta=config.CLASES_ARMAS_ALERTA, modo_debug=config.MODO_DEBUG
            )

        if config.ACTIVAR_MODELO_COMPORTAMIENTO and pose_model:
            pose_results, skeletons_data = batch_detect_pose(
                pose_model, frames_list, conf=0.5, modo_debug=config.MODO_DEBUG
            )

        # ==================================================
        # D. ANALÍTICA DE DATOS COOPERATIVA POR CANAL
        # ==================================================
        for i, cam_name in enumerate(cam_names_list):
            frame = frames_list[i]
            w_res = weapon_results[i] if i < len(weapon_results) else None
            p_res = pose_results[i] if i < len(pose_results) else None
            esqueletos_camara = skeletons_data[i] if i < len(skeletons_data) else None

            weapon_in_frame = False
            comportamiento_anomalo = False
            nombre_comportamiento = ""

            # --- FILTRADO DE SEGURIDAD: DETECCIÓN DE ARMAS ---
            weapon_in_frame = alertas_armas_batch[i] if i < len(alertas_armas_batch) else False

            if w_res and len(w_res.boxes) > 0:
                frame = w_res.plot(img=frame)

            # --- PROCESAMIENTO TEMPORAL: ANÁLISIS DE POSE Y GRU ---
            if p_res:
                # CORRECCIÓN DE RUIDO VISUAL: Solo dibuja el esqueleto si MODO_DEBUG esta activo.
                # En modo produccion, los puntos se extraen en silencio para alimentar al clasificador.
                if config.MODO_DEBUG:
                    frame = p_res.plot(img=frame)
                
                if esqueletos_camara is not None and p_res.boxes.id is not None:
                    ids_personas = p_res.boxes.id.int().cpu().tolist()
                    
                    for i_persona, track_id in enumerate(ids_personas):
                        kp_persona = esqueletos_camara[i_persona] 
                        
                        # Extraccion geometrica de coordenadas (X, Y) normalizadas
                        kp_aplanado = kp_persona[:, :2].flatten()
                        
                        if track_id not in historial_personas[cam_name]:
                            historial_personas[cam_name][track_id] = deque(maxlen=SEQ_LENGTH)
                        
                        historial_personas[cam_name][track_id].append(kp_aplanado)
                        
                        # Evaluacion secuencial temporal
                        if len(historial_personas[cam_name][track_id]) == SEQ_LENGTH:
                            comportamiento = predict_behavior(
                                gru_model, device, 
                                historial_personas[cam_name][track_id], 
                                CLASES_COMPORTAMIENTO
                            )
                            
                            if comportamiento != "Normal":
                                comportamiento_anomalo = True
                                nombre_comportamiento = comportamiento
                                # Alerta en pantalla en formato rigido
                                cv2.putText(frame, f"CRITICAL: {comportamiento.upper()} DETECTED", (50, 100), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                                
                    # Liberacion dinamica de registros de entidades desvinculadas (RAM cleanup)
                    ids_a_borrar = [tid for tid in historial_personas[cam_name].keys() if tid not in ids_personas]
                    for tid in ids_a_borrar:
                        del historial_personas[cam_name][tid]

            # --- ANÁLISIS LOGÍCO TEMPORAL (Supresion de falsos positivos) ---
            alerta_arma = update_window(
                cam_name, weapon_in_frame, windows_armas, 
                config.ACTIVATION_THRESHOLD, alert_state_armas
            )

            # --- DESPACHO DE ALERTAS CRÍTICAS ---
            if not recording_state[cam_name]["recording"]:
                alertas_enviadas_evento[cam_name].clear()
                
            if alerta_arma and "arma" not in alertas_enviadas_evento[cam_name]:
                send_alert(cam_name, f"WARNING: Deteccion de armamento confirmada en canal: {cam_name}")
                alertas_enviadas_evento[cam_name].add("arma")

            if comportamiento_anomalo and nombre_comportamiento not in alertas_enviadas_evento[cam_name]:
                send_alert(cam_name, f"CRITICAL: Comportamiento hostil confirmado en canal {cam_name}: {nombre_comportamiento.upper()}")
                alertas_enviadas_evento[cam_name].add(nombre_comportamiento)

            alert_triggered = alerta_arma or comportamiento_anomalo
            amenaza_presente = weapon_in_frame or comportamiento_anomalo

            # --- CONTROL MULTIMEDIA Y SALIDA RTSP ---
            frame = draw_performance_overlay(frame, fps_real)

            handle_recording(
                cam_name, frame, camera_resolutions, recording_state,
                config.POST_BUFFER_SECONDS, alert_triggered, amenaza_presente
            )

            streamers[cam_name].enviar_frame(frame)


if __name__ == "__main__":
    ejecutar_sistema_principal()