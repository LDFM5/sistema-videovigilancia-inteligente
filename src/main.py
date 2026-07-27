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
import torch
import numpy as np
from collections import deque
from ultralytics import YOLO

# Importaciones de módulos del ecosistema core
import config
from cameras import initialize_cameras
from detection import batch_detect_weapons
from temporal_logic import initialize_windows, update_window
from recorder import initialize_recording_state, handle_recording
from streamer import RTSPStreamer
from visualization import draw_performance_overlay
from behavior_cnn import cargar_modelo_violencia, evaluar_secuencia_violencia

# Registro global dinámico para permitir la lectura externa de estados en caliente
_registro_estados_global = {}


def ejecutar_sistema_principal(shared_state):
    # Importación encapsulada para evitar dependencias circulares con app.py
    from alerts import dispatch_security_alert

    if shared_state is None:
        print("SYS_CORE_ERROR: NO SE PROVEYÓ EL PUNTERO DE MEMORIA COMPARTIDO SHARED_STATE.")
        return

    print("SYS_CORE: INICIALIZANDO NÚCLEO INFALIBLE DE INFERENCIAS EDGE AI...")

    # ======================================================
    # 1. INICIALIZACIÓN DE PUNTEROS NEURONALES Y HARDWARE
    # ======================================================
    weapon_model = None
    behavior_model = None  
    dispositivo_ia = "cuda" if torch.cuda.is_available() else "cpu"

    # ======================================================
    # 2. INTERCONEXIÓN DE DISPOSITIVOS DE CAPTURA
    # ======================================================
    cameras, camera_resolutions_raw, camera_fps = initialize_cameras()
    camera_resolutions = {k.upper(): v for k, v in camera_resolutions_raw.items()}

    # Enlazar referencia de estado compartido en cada cámara
    for cap_obj in cameras.values():
        cap_obj.shared_state = shared_state

    # ======================================================
    # 2.5 REGISTROS DE MEMORIA VOLÁTIL POR CANAL
    # ======================================================
    # Búfer circular para secuencias (imágenes reducidas a 224x224)
    historial_secuencias = {cam_name.upper(): deque(maxlen=90) for cam_name in cameras}

    # Búfer de estabilización de predicciones temporales
    historial_predicciones = {cam_name.upper(): deque(maxlen=12) for cam_name in cameras}

    # ======================================================
    # 3. FILTROS DE MITIGACIÓN Y MAQUINAS DE ESTADO
    # ======================================================
    windows_armas = initialize_windows(camera_fps, config.WINDOW_SECONDS)
    alert_state_armas = {cam_name.upper(): False for cam_name in cameras}
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

        streamers[cam_upper] = RTSPStreamer(
            cam_name, width=target_w, height=target_h, fps=15, shared_state=shared_state
        )

    # ======================================================
    # 5. PROTOCOLO DE DESCONEXIÓN SEGURA (CLEANUP)
    # ======================================================
    def limpieza_segura():
        print("\nSYS_CORE: SIGNAL DE TERMINACIÓN RECIBIDA. LIBERANDO RECURSOS DE HARDWARE...")
        for cap in cameras.values():
            if hasattr(cap, 'release'): cap.release()
            elif hasattr(cap, 'stop'): cap.stop()
            
        for streamer in streamers.values():
            streamer.cerrar()
            
        cv2.destroyAllWindows()
        print("SYS_CORE: PROTOCOLO DE LIMPIEZA COMPLETADO. RECURSOS LIBERADOS EN REGLA.")

    atexit.register(limpieza_segura)

    # Inicialización del estado en el dashboard
    for cam_name in cameras:
        shared_state.emitir_evento_dashboard('camera_status', {
            "camera": cam_name.lower(), 
            "status": "analyzing"
        })

    shared_state.emitir_evento_dashboard('system_log', {
        "type": "success", 
        "message": "NÚCLEO_IA: BUCLE_DE_INFERENCIA_CONTINUO_DESPLEGADO (MODO_BATCH_ESTABLE)"
    })
    
    tiempo_anterior = time.time()

    # ======================================================
    # 6. BUCLE PRINCIPAL DE INFERENCIA CONTINUA
    # ======================================================
    while True:
        frames_list = []
        cam_names_list = []

        for cam_name, cap in cameras.items():
            res = cap.read()
            frame = res[1] if isinstance(res, tuple) else res
            if frame is not None:
                frames_list.append(frame.copy())
                cam_names_list.append(cam_name.upper())

        if not frames_list:
            time.sleep(0.01)
            continue

        tiempo_actual = time.time()
        diferencia_tiempo = tiempo_actual - tiempo_anterior
        if diferencia_tiempo <= 0:
            diferencia_tiempo = 0.001
            
        fps_real = 1.0 / diferencia_tiempo
        tiempo_anterior = tiempo_actual

        # =========================================================================
        # GESTIÓN EN CALIENTE DE HARDWARE (CARGA/DESCARGA DINÁMICA DE MODELOS)
        # =========================================================================
        
        # --- CONTROL MODELO DE ARMAS ---
        if shared_state.config_ram.get("cfg_armas", True):
            if weapon_model is None:
                try:
                    shared_state.emitir_evento_dashboard('system_log', {
                        "type": "info", 
                        "message": "NÚCLEO_IA: DETECTADO COMANDO DE ACTIVACIÓN. CARGANDO MODELO DE ARMAS..."
                    })
                    weapon_model = YOLO(config.WEAPON_MODEL_PATH)
                    shared_state.emitir_evento_dashboard('system_log', {
                        "type": "success", 
                        "message": "NÚCLEO_IA: MODELO DE ARMAS TOTALMENTE DESPLEGADO EN MEMORIA."
                    })
                except Exception as e:
                    shared_state.emitir_evento_dashboard('system_log', {
                        "type": "error", 
                        "message": f"NÚCLEO_IA_ERROR: FALLO AL CARGAR MODELO DE ARMAS -> {str(e).upper()}"
                    })
        else:
            if weapon_model is not None:
                shared_state.emitir_evento_dashboard('system_log', {
                    "type": "warn", 
                    "message": "NÚCLEO_IA: LIBERANDO MEMORIA DEL MODELO DE ARMAS..."
                })
                weapon_model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # --- CONTROL MODELO DE COMPORTAMIENTO (CNN + GRU VIOLENCIA) ---
        if shared_state.config_ram.get("cfg_comportamiento", False):
            if behavior_model is None:
                try:
                    shared_state.emitir_evento_dashboard('system_log', {
                        "type": "info", 
                        "message": "NÚCLEO_IA: CARGANDO CLASIFICADOR DE VIOLENCIA SECUENCIAL (CNN + GRU)..."
                    })
                    behavior_model = cargar_modelo_violencia(config.BEHAVIOR_MODEL_PATH, dispositivo_ia)
                    if behavior_model is not None:
                        shared_state.emitir_evento_dashboard('system_log', {
                            "type": "success", 
                            "message": "NÚCLEO_IA: PIPELINE DE COMPORTAMIENTO DESPLEGADO CORRECTAMENTE."
                        })
                except Exception as e:
                    shared_state.emitir_evento_dashboard('system_log', {
                        "type": "error", 
                        "message": f"NÚCLEO_IA_ERROR: FALLO AL CARGAR MODELO DE COMPORTAMIENTO -> {str(e).upper()}"
                    })
        else:
            if behavior_model is not None:
                shared_state.emitir_evento_dashboard('system_log', {
                    "type": "warn", 
                    "message": "NÚCLEO_IA: LIBERANDO PIPELINE DE COMPORTAMIENTO..."
                })
                behavior_model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # ==================================================
        # C. PROCESAMIENTO MATRICIAL EN LOTE (BATCH INFERENCE)
        # ==================================================
        weapon_results = []
        alertas_armas_batch = [] 

        if weapon_model is not None:
            weapon_results, alertas_armas_batch = batch_detect_weapons(
                weapon_model, frames_list, 
                conf=shared_state.config_ram.get("cfg_confianza_armas", 0.50), 
                clases_alerta=config.CLASES_ARMAS_ALERTA, 
                modo_debug=shared_state.config_ram.get("cfg_debug", False)
            )

        # ==================================================
        # D. ANALÍTICA DE DATOS COOPERATIVA POR CANAL
        # ==================================================
        for i, cam_upper in enumerate(cam_names_list):
            frame = frames_list[i]
            w_res = weapon_results[i] if i < len(weapon_results) else None

            weapon_in_frame = False
            comportamiento_anomalo = False
            nombre_comportamiento = ""

            # 1. Almacenar imagen reducida en el historial para mitigar uso de RAM
            frame_small = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
            historial_secuencias[cam_upper].append(frame_small)

            # --- DETECCIÓN DE ARMAS ---
            if weapon_model is not None:
                weapon_in_frame = alertas_armas_batch[i] if i < len(alertas_armas_batch) else False
                if w_res and len(w_res.boxes) > 0:
                    frame = w_res.plot(img=frame)

            # --- ANÁLISIS DE COMPORTAMIENTO HOSTIL ---
            if behavior_model is not None:
                modo_debug = shared_state.config_ram.get("cfg_debug", False)
                umbral_actual = shared_state.config_ram.get("cfg_confianza_comportamiento", 0.50)

                # Submuestreo: Inferencia cada 3 cuadros para liberar CPU/GPU
                if len(historial_secuencias[cam_upper]) % 3 == 0:
                    clase_comportamiento, score = evaluar_secuencia_violencia(
                        behavior_model, 
                        dispositivo_ia, 
                        historial_secuencias[cam_upper], 
                        umbral_confianza=umbral_actual
                    )
                    es_violencia_frame = (clase_comportamiento == "VIOLENCE")
                    
                    if modo_debug:
                        print(f"[DEBUG - {cam_upper}] Estado: {clase_comportamiento} | Score: {score:.4f} | Umbral: {umbral_actual}")
                else:
                    es_violencia_frame = historial_predicciones[cam_upper][-1] if len(historial_predicciones[cam_upper]) > 0 else False

                if modo_debug:
                    if es_violencia_frame:
                        comportamiento_anomalo = True
                        nombre_comportamiento = "VIOLENCIA"
                else:
                    historial_predicciones[cam_upper].append(es_violencia_frame)
                    alertas_activas = sum(historial_predicciones[cam_upper])
                    total_evaluaciones = len(historial_predicciones[cam_upper])
                    
                    if total_evaluaciones >= 4 and (alertas_activas / total_evaluaciones) >= 0.50:
                        comportamiento_anomalo = True
                        nombre_comportamiento = "VIOLENCIA"

                # Renderizado optimizado del banner superior
                if comportamiento_anomalo:
                    h_img, w_img = frame.shape[:2]
                    roi = frame[0:45, 0:w_img]
                    # Aplicar atenuación del 75% usando multiplicación de escala atómica en C++
                    frame[0:45, 0:w_img] = cv2.convertScaleAbs(roi, alpha=0.25)

                    cv2.putText(
                        frame, 
                        "ALERTA: INCIDENTE DE CONDUCTA HOSTIL CONFIRMADO", 
                        (20, 28), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.55, 
                        (68, 23, 255),  # BGR: Rojo
                        2, 
                        cv2.LINE_AA
                    )

            # --- ANÁLISIS LÓGICO TEMPORAL DE ARMAS ---
            alerta_arma = update_window(
                cam_upper, weapon_in_frame, windows_armas, 
                config.ACTIVATION_THRESHOLD, alert_state_armas
            )

            # Resolución de disparadores de alerta
            alert_triggered = (alerta_arma if weapon_model is not None else False) or comportamiento_anomalo
            amenaza_presente = (weapon_in_frame if weapon_model is not None else False) or comportamiento_anomalo

            estaba_grabando = recording_state[cam_upper]["recording"]

            # Pipeline de persistencia y volcado de video
            handle_recording(
                cam_upper, frame, camera_resolutions, recording_state,
                shared_state.config_ram.get("cfg_postbuffer", 15), 
                alert_triggered, amenaza_presente, shared_state=shared_state
            )

            esta_grabando_actualmente = recording_state[cam_upper]["recording"]

            # --- SINCRONIZACIÓN CON EL DASHBOARD WEB Y TELEGRAM ---
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

            elif not esta_grabando_actualmente and estaba_grabando:
                alertas_enviadas_evento[cam_upper].clear()
                
                shared_state.emitir_evento_dashboard('camera_status', {
                    "camera": cam_upper.lower(), 
                    "status": "analyzing"
                })

            # Overlay de rendimiento en modo Debug
            if shared_state.config_ram.get("cfg_debug", False):
                frame = draw_performance_overlay(frame, fps_real)

            # Estampado estilo CCTV (Metadatos)
            cadena_tiempo_actual = time.strftime("%d/%m/%Y  ──  %H:%M:%S")
            texto_metadatos = f"CAM: {cam_upper}  |  {cadena_tiempo_actual}"
            
            h_img, w_img = frame.shape[:2]
            cv2.rectangle(frame, (10, h_img - 30), (460, h_img - 5), (10, 11, 13), -1)
            cv2.putText(
                frame, 
                texto_metadatos, 
                (20, h_img - 12), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.45, 
                (227, 230, 235), 
                1, 
                cv2.LINE_AA
            )

            # Transmisión RTSP
            streamers[cam_upper].enviar_frame(frame)
            
            # Actualización corregida de la fotografía global de estado
            global _registro_estados_global
            cam_key_lower = cam_upper.lower()
            
            if hasattr(cameras.get(cam_key_lower), 'estado_error_enviado') and cameras[cam_key_lower].estado_error_enviado:
                _registro_estados_global[cam_key_lower] = "error"
            elif esta_grabando_actualmente:
                _registro_estados_global[cam_key_lower] = "detecting"
            else:
                _registro_estados_global[cam_key_lower] = "analyzing"


def obtener_mapa_estados_actual():
    """
    Devuelve una copia segura para hilos del estado en tiempo real de las cámaras.
    """
    global _registro_estados_global
    return dict(_registro_estados_global)