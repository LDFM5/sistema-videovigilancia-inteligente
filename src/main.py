"""
main.py

Motor Principal de Inferencias (Edge AI Core)
- Recolecta frames de los hilos asíncronos de CameraStream.
- Ejecuta Inferencia por Lotes (Batching) a nivel de hardware.
- Centraliza la grabación automatizada, alertas y streaming RTSP/WebRTC.
"""

import cv2
import time
import atexit
from ultralytics import YOLO

# Importaciones de módulos del ecosistema core
import config
from cameras import initialize_cameras
from detection import batch_detect_weapons, batch_detect_pose
from temporal_logic import initialize_windows, update_window
from recorder import initialize_recording_state, handle_recording
from streamer import RTSPStreamer
from visualization import draw_performance_overlay

from behavior import load_behavior_model, predict_behavior
from collections import deque


def ejecutar_sistema_principal(shared_state):
    # LAZY IMPORT ATÓMICO: Importaciones encapsuladas localmente para mitigar
    # bloqueos mutuos (Deadlocks) e importaciones circulares en frío con app.py
    from alerts import dispatch_security_alert

    # Si por algún motivo shared_state no viene, evitamos un colapso en frío
    if shared_state is None:
        print("SYS_CORE_ERROR: NO SE PROVEYÓ EL PUNTERO DE MEMORIA COMPARTIDO SHARED_STATE.")
        return

    print("SYS_CORE: INICIALIZANDO NÚCLEO INFALIBLE DE INFERENCIAS EDGE AI...")

    # ======================================================
    # 1. CARGA DE MODELOS EN ACELERADORES DE HARDWARE
    # ======================================================
    weapon_model = None
    if config.ACTIVAR_MODELO_ARMAS:
        shared_state.emitir_evento_dashboard('system_log', {"type": "info", "message": "NÚCLEO_IA: CARGANDO RED NEURONAL DE DETECCIÓN DE ARMAMIENTO (YOLOV8)..."})
        weapon_model = YOLO(config.WEAPON_MODEL_PATH)
        shared_state.emitir_evento_dashboard('system_log', {"type": "success", "message": "NÚCLEO_IA: PIPELINE DE DETECCIÓN DE ARMAMIENTO TOTALMENTE OPERATIVO."})

    pose_model = None
    if config.ACTIVAR_MODELO_COMPORTAMIENTO:
        shared_state.emitir_evento_dashboard('system_log', {"type": "info", "message": "NÚCLEO_IA: CARGANDO RED NEURONAL DE ESTIMACIÓN DE POSE ESTRUCTURAL..."})
        pose_model = YOLO(config.POSE_MODEL_PATH)
        shared_state.emitir_evento_dashboard('system_log', {"type": "success", "message": "NÚCLEO_IA: PIPELINE DE ESTIMACIÓN DE POSE OPERATIVO (STATUS_OK)."})
        
        shared_state.emitir_evento_dashboard('system_log', {"type": "info", "message": "NÚCLEO_IA: CARGANDO CLASIFICADOR SECUENCIAL DE COMPORTAMIENTO (GRU)..."})
        SEQ_LENGTH = 16 
        CLASES_COMPORTAMIENTO = ["Normal", "Fighting"]
        gru_model, device = load_behavior_model(config.BEHAVIOR_MODEL_PATH)
        shared_state.emitir_evento_dashboard('system_log', {"type": "success", "message": f"NÚCLEO_IA: CLASIFICADOR SECUENCIAL GRU ASIGNADO DISPOSITIVO: {str(device).upper()}"})

    # ======================================================
    # 2. INTERCONEXIÓN DE DISPOSITIVOS DE CAPTURA
    # ======================================================
    cameras, camera_resolutions_raw, camera_fps = initialize_cameras()
    
    # NORMALIZACIÓN INDUSTRIAL: Forzamos las llaves de resolución a mayúsculas
    # para evitar errores de colisión (KeyError) entre hilos concurrentes
    camera_resolutions = {k.upper(): v for k, v in camera_resolutions_raw.items()}

    # ======================================================
    # 2.5 ASIGNACIÓN DE REGISTROS DE MEMORIA VOLÁTIL
    # ======================================================
    if config.ACTIVAR_MODELO_COMPORTAMIENTO:
        historial_personas = {cam_name.upper(): {} for cam_name in cameras}

    # ======================================================
    # 3. FILTROS DE MITIGACIÓN Y VENTANAS TEMPORALES
    # ======================================================
    windows_armas = initialize_windows(camera_fps, config.WINDOW_SECONDS)
    alert_state_armas = {cam_name.upper(): False for cam_name in cameras}
    
    # Registro de supresión de redundancia para notificaciones externas
    alertas_enviadas_evento = {cam_name.upper(): set() for cam_name in cameras}

    recording_state = initialize_recording_state(cameras, config.PRE_BUFFER_SECONDS)

    # ======================================================
    # 4. INSTANCIACIÓN DE CANALES RTSP (MediaMTX OUTBOUND)
    # ======================================================
    streamers = {}
    for cam_name in cameras:
        cam_upper = cam_name.upper()
        w, h = camera_resolutions[cam_upper] 
        target_w = 800
        target_h = int((target_w / w) * h)
        if target_h % 2 != 0: 
            target_h += 1

        # Pasamos shared_state para que el streamer emita logs en la misma memoria
        streamers[cam_upper] = RTSPStreamer(
            cam_name, width=target_w, height=target_h, fps=15, shared_state=shared_state
        )

    # ======================================================
    # 5. PROTOCOLO DE DESCONEXIÓN SEGURA (CLEANUP)
    # ======================================================
    def limpieza_segura():
        print("\nSYS_CORE: SIGNAL_DE_TERMINACIÓN_RECIBIDA. LIBERANDO RECURSOS DE HARDWARE...")
        for cap in cameras.values():
            if hasattr(cap, 'release'): cap.release()
            elif hasattr(cap, 'stop'): cap.stop()
            
        for streamer in streamers.values():
            streamer.cerrar()
            
        cv2.destroyAllWindows()
        print("SYS_CORE: PROTOCOLO_DE_LIMPIEZA_COMPLETADO. RECURSOS LIBERADOS EN REGLA.")

    atexit.register(limpieza_segura)

    # INICIALIZACIÓN UNIFICADA DEL DASHBOARD EN FRÍO
    for cam_name in cameras:
        shared_state.emitir_evento_dashboard('camera_status', {
            "camera": cam_name.lower(), 
            "status": "analyzing"
        })

    shared_state.emitir_evento_dashboard('system_log', {"type": "success", "message": "NÚCLEO_IA: BUCLE_DE_INFERENCIA_CONTINUO_DESPLEGADO (MODO_BATCH_ESTABLE)"})
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
                cam_names_list.append(cam_name.upper())

        if not frames_list:
            time.sleep(0.01)
            continue

        # B. Métricas de rendimiento del ciclo core
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
                pose_model, frames_list, conf=0.5
            )

        # ==================================================
        # D. ANALÍTICA DE DATOS COOPERATIVA POR CANAL
        # ==================================================
        for i, cam_upper in enumerate(cam_names_list):
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
                if config.MODO_DEBUG:
                    frame = p_res.plot(img=frame)
                
                if esqueletos_camara is not None and p_res.boxes.id is not None:
                    ids_personas = p_res.boxes.id.int().cpu().tolist()
                    
                    for i_persona, track_id in enumerate(ids_personas):
                        kp_persona = esqueletos_camara[i_persona] 
                        kp_aplanado = kp_persona[:, :2].flatten()
                        
                        if track_id not in historial_personas[cam_upper]:
                            historial_personas[cam_upper][track_id] = deque(maxlen=SEQ_LENGTH)
                        
                        historial_personas[cam_upper][track_id].append(kp_aplanado)
                        
                        if len(historial_personas[cam_upper][track_id]) == SEQ_LENGTH:
                            comportamiento = predict_behavior(
                                gru_model, device, 
                                historial_personas[cam_upper][track_id], 
                                CLASES_COMPORTAMIENTO
                            )
                            
                            if comportamiento != "Normal":
                                comportamiento_anomalo = True
                                nombre_comportamiento = comportamiento.upper()
                                cv2.putText(frame, f"CRITICAL: {nombre_comportamiento} DETECTED", (50, 100), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                                
                    ids_a_borrar = [tid for tid in historial_personas[cam_upper].keys() if tid not in ids_personas]
                    for tid in ids_a_borrar:
                        del historial_personas[cam_upper][tid]

            # --- ANÁLISIS LÓGICO TEMPORAL (Supresión de falsos positivos) ---
            alerta_arma = update_window(
                cam_upper.lower(), weapon_in_frame, windows_armas, 
                config.ACTIVATION_THRESHOLD, alert_state_armas
            )

            # RESOLUCIÓN DE VARIABLES DE CONTROL INTERNO
            alert_triggered = alerta_arma or comportamiento_anomalo
            amenaza_presente = weapon_in_frame or comportamiento_anomalo

            # --- OBTENCIÓN DEL ESTADO ANTES DE LA EVALUACIÓN MULTIMEDIA ---
            estaba_grabando = recording_state[cam_upper]["recording"]

            # --- PIPELINE DE PERSISTENCIA MULTIMEDIA (MÁQUINA DE ESTADOS) ---
            handle_recording(
                cam_upper, frame, camera_resolutions, recording_state,
                config.POST_BUFFER_SECONDS, alert_triggered, amenaza_presente, shared_state=shared_state
            )

            # --- OBTENCIÓN DEL ESTADO POST-EVALUACIÓN MULTIMEDIA ---
            esta_grabando_actualmente = recording_state[cam_upper]["recording"]

            # =========================================================================
            # SINCRONIZACIÓN SOBERANA DEL DASHBOARD CON EL ESTADO REAL DE GRABACIÓN
            # =========================================================================
            
            # CASO 1: Transición de Apagado -> Encendido (Comienza a grabar un evento confirmado)
            if esta_grabando_actualmente and not estaba_grabando:
                shared_state.emitir_evento_dashboard('camera_status', {
                    "camera": cam_upper.lower(), 
                    "status": "detecting"
                })
                
                if alerta_arma and "ARMA" not in alertas_enviadas_evento[cam_upper]:
                    dispatch_security_alert(
                        cam_name=cam_upper, 
                        log_type="Arma detectada", 
                        message_payload="Presencia de objeto peligroso confirmada", 
                        shared_state=shared_state
                    )
                    alertas_enviadas_evento[cam_upper].add("ARMA")

                if comportamiento_anomalo and nombre_comportamiento not in alertas_enviadas_evento[cam_upper]:
                    dispatch_security_alert(
                        cam_name=cam_upper, 
                        log_type="Comportamiento hostil", 
                        message_payload=f"Actividad sospechosa detectada como {nombre_comportamiento.lower()}", 
                        shared_state=shared_state
                    )
                    alertas_enviadas_evento[cam_upper].add(nombre_comportamiento)

            # CASO 2: Transición de Encendido -> Apagado (Finalizó el evento y se salvó el clip)
            elif not esta_grabando_actualmente and estaba_grabando:
                alertas_enviadas_evento[cam_upper].clear()
                
                shared_state.emitir_evento_dashboard('camera_status', {
                    "camera": cam_upper.lower(), 
                    "status": "analyzing"
                })

            # --- CONTROL MULTIMEDIA Y SALIDA RTSP ---
            frame = draw_performance_overlay(frame, fps_real)
            streamers[cam_upper].enviar_frame(frame)