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
from ultralytics import YOLO

# Importaciones de módulos del ecosistema core
import config
from cameras import initialize_cameras
from detection import batch_detect_weapons
from temporal_logic import initialize_windows, update_window
from recorder import initialize_recording_state, handle_recording
from streamer import RTSPStreamer
from visualization import draw_performance_overlay

# Nuevo pipeline integrado CNN + GRU autónomo
from behavior_cnn import cargar_modelo_violencia, evaluar_secuencia_violencia
from collections import deque

# Registro global dinámico para permitir la lectura externa de estados en caliente
_registro_estados_global = {}


def ejecutar_sistema_principal(shared_state):
    # LAZY IMPORT ATÓMICO: Importaciones encapsuladas localmente para mitigar
    # bloqueos mutuos (Deadlocks) e importaciones circulares en frío con app.py
    from alerts import dispatch_security_alert

    if shared_state is None:
        print("SYS_CORE_ERROR: NO SE PROVEYÓ EL PUNTERO DE MEMORIA COMPARTIDO SHARED_STATE.")
        return

    print("SYS_CORE: INICIALIZANDO NÚCLEO INFALIBLE DE INFERENCIAS EDGE AI...")

    # ======================================================
    # 1. INICIALIZACIÓN DE PUNTEROS NEURONALES (DIFERIDOS)
    # ======================================================
    weapon_model = None
    behavior_model = None  # Puntero único para la arquitectura ViolenceNet
    
    # Determinar el acelerador de hardware óptimo para las operaciones de tensores
    dispositivo_ia = "cuda" if torch.cuda.is_available() else "cpu"

    # ======================================================
    # 2. INTERCONEXIÓN DE DISPOSITIVOS DE CAPTURA
    # ======================================================
    cameras, camera_resolutions_raw, camera_fps = initialize_cameras()
    camera_resolutions = {k.upper(): v for k, v in camera_resolutions_raw.items()}

    # ENLACE DE MEMORIA COMPARTIDA: Inyectar la referencia unificada a cada cámara
    for cap_obj in cameras.values():
        cap_obj.shared_state = shared_state

    # ======================================================
    # 2.5 ASIGNACIÓN DE REGISTROS DE MEMORIA VOLÁTIL
    # ======================================================
    # Almacena una cola circular con los últimos frames físicos (imágenes a color)
    # para poder realizar el muestreo equidistante de 16 cuadros. El tamaño 90 
    # cubre aproximadamente de 3 a 5 segundos de memoria temporal según los FPS.
    historial_secuencias = {cam_name.upper(): deque(maxlen=90) for cam_name in cameras}

    # BLINDAJE ANTI-PARPADEO: Almacena las últimas 12 predicciones booleanas (True/False)
    # de violencia para estabilizar el estado mediante un filtro de promedio móvil.
    historial_predicciones = {cam_name.upper(): deque(maxlen=12) for cam_name in cameras}

    # ======================================================
    # 3. FILTROS DE MITIGACIÓN Y VENTANAS TEMPORALES
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
        if shared_state.config_ram["cfg_armas"]:
            if weapon_model is None:
                shared_state.emitir_evento_dashboard('system_log', {"type": "info", "message": "NÚCLEO_IA: DETECTADO COMANDO DE ACTIVACIÓN. CARGANDO MODELO DE ARMAS (YOLOV8)..."})
                weapon_model = YOLO(config.WEAPON_MODEL_PATH)
                shared_state.emitir_evento_dashboard('system_log', {"type": "success", "message": "NÚCLEO_IA: MODELO DE ARMAS TOTALMENTE DESPLEGADO EN MEMORIA."})
        else:
            if weapon_model is not None:
                shared_state.emitir_evento_dashboard('system_log', {"type": "warn", "message": "NÚCLEO_IA: DETECTADO COMANDO DE DESACTIVACIÓN. LIBERANDO MEMORIA DEL MODELO DE ARMAS..."})
                weapon_model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                shared_state.emitir_evento_dashboard('system_log', {"type": "info", "message": "NÚCLEO_IA: RECURSOS DEL MODELO DE ARMAS LIBERADOS CORRECTAMENTE."})

        # --- CONTROL MODELO DE COMPORTAMIENTO (CNN + GRU VIOLENCIA) ---
        if shared_state.config_ram["cfg_comportamiento"]:
            if behavior_model is None:
                shared_state.emitir_evento_dashboard('system_log', {"type": "info", "message": "NÚCLEO_IA: DETECTADO COMANDO DE ACTIVACIÓN. CARGANDO CLASIFICADOR DE VIOLENCIA SECUENCIAL (CNN + GRU)..."})
                behavior_model = cargar_modelo_violencia(config.BEHAVIOR_MODEL_PATH, dispositivo_ia)
                if behavior_model is not None:
                    shared_state.emitir_evento_dashboard('system_log', {"type": "success", "message": "NÚCLEO_IA: PIPELINE DE COMPORTAMIENTO DESPLEGADO CORRECTAMENTE."})
        else:
            if behavior_model is not None:
                shared_state.emitir_evento_dashboard('system_log', {"type": "warn", "message": "NÚCLEO_IA: DETECTADO COMANDO DE DESACTIVACIÓN. LIBERANDO PIPELINE DE COMPORTAMIENTO..."})
                behavior_model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                shared_state.emitir_evento_dashboard('system_log', {"type": "info", "message": "NÚCLEO_IA: RECURSOS DE COMPORTAMIENTO CNN + GRU LIBERADOS SEGUROS."})

        # ==================================================
        # C. PROCESAMIENTO MATRICIAL EN LOTE (BATCH INFERENCE)
        # ==================================================
        weapon_results = []
        alertas_armas_batch = [] 

        if weapon_model is not None:
            weapon_results, alertas_armas_batch = batch_detect_weapons(
                weapon_model, frames_list, 
                conf=shared_state.config_ram["cfg_confianza"], 
                clases_alerta=config.CLASES_ARMAS_ALERTA, 
                modo_debug=shared_state.config_ram["cfg_debug"]
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

            # Alimentar continuamente la cola temporal de imágenes a color
            historial_secuencias[cam_upper].append(frame.copy())

            # --- FILTRADO DE SEGURIDAD: DETECCIÓN DE ARMAS ---
            if weapon_model is not None:
                weapon_in_frame = alertas_armas_batch[i] if i < len(alertas_armas_batch) else False
                if w_res and len(w_res.boxes) > 0:
                    frame = w_res.plot(img=frame)

            # --- PROCESAMIENTO TEMPORAL: ANÁLISIS DE COMPORTAMIENTO CNN + GRU ---
            if behavior_model is not None:
                clase_comportamiento, score = evaluar_secuencia_violencia(
                    behavior_model, dispositivo_ia, 
                    historial_secuencias[cam_upper], 
                    umbral_confianza=0.55  # Sutil incremento para mitigar falsos positivos directos
                )
                
                # Registrar el resultado binario en el búfer de suavizado temporal
                es_violencia_frame = (clase_comportamiento == "VIOLENCE")
                historial_predicciones[cam_upper].append(es_violencia_frame)
                
                # EVALUACIÓN DE HISTÉRESIS: Calcular la densidad de alertas en la ventana temporal
                alertas_activas = sum(historial_predicciones[cam_upper])
                total_evaluaciones = len(historial_predicciones[cam_upper])
                
                # Factor de activación industrial: Al menos el 65% de los últimos cuadros deben ser positivos
                if total_evaluaciones >= 6 and (alertas_activas / total_evaluaciones) >= 0.65:
                    comportamiento_anomalo = True
                    nombre_comportamiento = "VIOLENCIA"
                    
                    # 1. Dibujar rectángulo de fondo semi-transparente oscuro en la parte superior del frame
                    h_img, w_img = frame.shape[:2]
                    overlay_banner = frame.copy()
                    cv2.rectangle(overlay_banner, (0, 0), (w_img, 45), (15, 16, 18), -1)
                    # Aplicar transparencia (Alfa = 0.75) para no tapar el video por completo
                    cv2.addWeighted(overlay_banner, 0.75, frame, 0.25, 0, frame)
                    
                    # 2. Inyectar tipografía ejecutiva limpia (Escala baja, grosor moderado)
                    cv2.putText(
                        frame, 
                        f"ALERTA PERIMETRAL: INCIDENTE DE CONDUCTA HOSTIL CONFIRMADO", 
                        (20, 28), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.55, 
                        (68, 23, 255),  # Color rojo/alerta industrial puro (BGR: 255, 23, 68)
                        2, 
                        cv2.LINE_AA
                    )

            # --- ANÁLISIS LÓGICO TEMPORAL (Supresión de falsos positivos) ---
            alerta_arma = update_window(
                cam_upper, weapon_in_frame, windows_armas, 
                config.ACTIVATION_THRESHOLD, alert_state_armas
            )

            # RESOLUCIÓN DE VARIABLES DE CONTROL INTERNO
            alert_triggered = (alerta_arma if weapon_model is not None else False) or comportamiento_anomalo
            amenaza_presente = (weapon_in_frame if weapon_model is not None else False) or comportamiento_anomalo

            # --- OBTENCIÓN DEL ESTADO ANTES DE LA EVALUACIÓN MULTIMEDIA ---
            estaba_grabando = recording_state[cam_upper]["recording"]

            # --- PIPELINE DE PERSISTENCIA MULTIMEDIA (MÁQUINA DE ESTADOS) ---
            handle_recording(
                cam_upper, frame, camera_resolutions, recording_state,
                shared_state.config_ram["cfg_postbuffer"], alert_triggered,  amenaza_presente, shared_state=shared_state
            )

            # --- OBTENCIÓN DEL ESTADO POST-EVALUACIÓN MULTIMEDIA ---
            esta_grabando_actualmente = recording_state[cam_upper]["recording"]

            # =========================================================================
            # SINCRONIZACIÓN SOBERANA DEL DASHBOARD CON EL ESTADO REAL DE GRABACIÓN
            # =========================================================================
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

            # --- CONTROL MULTIMEDIA Y SALIDA RTSP ---
            # --- CONTROL DE VISUALIZACIÓN DE TELEMETRÍA (MODO DEBUG) ---
            if shared_state.config_ram["cfg_debug"]:
                frame = draw_performance_overlay(frame, fps_real)

            # INYECCIÓN PERMANENTE DE METADATOS (Nombre Cámara, Fecha y Hora)
            # Formato resultante: CAM: WEBCAM  |  18/07/2026  ──  09:47:00
            cadena_tiempo_actual = time.strftime("%d/%m/%Y  ──  %H:%M:%S")
            texto_metadatos = f"CAM: {cam_upper}  |  {cadena_tiempo_actual}"
            
            h_img, w_img = frame.shape[:2]
            
            # Dibujar el fondo oscuro adaptativo (un poco más ancho para albergar el nombre)
            cv2.rectangle(frame, (10, h_img - 30), (460, h_img - 5), (10, 11, 13), -1)
            
            # Estampar la cadena unificada estilo CCTV de alta gama
            cv2.putText(
                frame, 
                texto_metadatos, 
                (20, h_img - 12), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.45, 
                (227, 230, 235),  # Color var(--text-main) en BGR
                1, 
                cv2.LINE_AA
            )

            # Enviar el frame procesado al canal RTSP correspondiente
            streamers[cam_upper].enviar_frame(frame)
            
            # ACTUALIZACIÓN DE FOTOGRAFÍA GLOBAL: Guarda el estado exacto de este frame
            global _registro_estados_global
            # Comprobación de error prioritaria mapeada desde cameras.py
            if hasattr(cameras[cam_name.lower()], 'estado_error_enviado') and cameras[cam_name.lower()].estado_error_enviado:
                _registro_estados_global[cam_name.lower()] = "error"
            elif esta_grabando_actualmente:
                _registro_estados_global[cam_name.lower()] = "detecting"
            else:
                _registro_estados_global[cam_name.lower()] = "analyzing"


def obtener_mapa_estados_actual():
    """
    Punto de acceso seguro para que Flask consulte el estado en tiempo real 
    de las cámaras sin intervenir en el bucle principal de inferencia.
    """
    global _registro_estados_global
    return _registro_estados_global