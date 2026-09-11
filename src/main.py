"""
main.py

Motor principal de inferencia de DAOCS.
- Sincronización de ciclo a 30 FPS reales de cámara (Pacing).
- Despacho continuo y desacoplado de alertas (Armas y Comportamiento).
- Normalización de las claves de cámara para las ventanas temporales.
- Captura explícita de excepciones en hilos de alertas.
"""

import os
import cv2
import numpy as np
import time
import atexit
import torch
import traceback
import threading
import queue
from collections import deque
from ultralytics import YOLO

# Módulos del sistema.
import config
from cameras import CameraStream, initialize_cameras
from detection import batch_detect_weapons
from temporal_logic import initialize_windows, update_window
from recorder import initialize_recording_state, handle_recording, inicializar_camara_grabacion
from streamer import RTSPStreamer
from visualization import draw_performance_overlay, dibujar_osd_cuadricula, dibujar_osd_proporcional
from behavior_cnn import cargar_modelo_violencia, evaluar_secuencias_violencia_batch

_registro_estados_global = {}


def _ejecutar_alerta_segura(cam_name, log_type, message_payload, shared_state):
    """Ejecuta dispatch_security_alert capturando cualquier error para evitar fallos silenciosos."""
    from alerts import dispatch_security_alert
    try:
        dispatch_security_alert(
            cam_name=cam_name,
            log_type=log_type,
            message_payload=message_payload,
            shared_state=shared_state
        )
    except Exception as e:
        print(f"\n[ERROR] No se pudo enviar la alerta {log_type} de {cam_name}: {e}")
        traceback.print_exc()


def ejecutar_sistema_principal(shared_state):
    if shared_state is None:
        print("[ERROR] No se recibió el estado compartido del sistema.")
        return

    print("[INFO] Iniciando el motor de inferencia.")

    # ======================================================
    # 1. ESTADOS DE MODELOS Y HARDWARE
    # ======================================================
    weapon_model = None
    behavior_model = None  
    dispositivo_ia = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    estado_previo_armas = None
    estado_previo_comportamiento = None
    modelo_previo_armas = None
    modelo_previo_comportamiento = None
    cola_carga_armas = queue.Queue()
    cola_carga_comportamiento = queue.Queue()
    generacion_armas = 0
    generacion_comportamiento = 0

    def cargar_armas_en_segundo_plano(generacion, ruta_modelo):
        try:
            modelo = YOLO(ruta_modelo)
            cola_carga_armas.put((generacion, modelo, None, os.path.basename(ruta_modelo)))
        except Exception as error:
            cola_carga_armas.put((generacion, None, error, os.path.basename(ruta_modelo)))

    def cargar_comportamiento_en_segundo_plano(generacion, ruta_modelo):
        try:
            modelo = cargar_modelo_violencia(
                ruta_modelo, dispositivo_ia
            )
            error = None if modelo is not None else RuntimeError(
                "EL MODELO DE COMPORTAMIENTO NO PUDO CARGARSE"
            )
            cola_carga_comportamiento.put((generacion, modelo, error, os.path.basename(ruta_modelo)))
        except Exception as error:
            cola_carga_comportamiento.put((generacion, None, error, os.path.basename(ruta_modelo)))

    # ======================================================
    # 2. DISPOSITIVOS Y BÚFERES POR CANAL (CLAVES NORMALIZADAS)
    # ======================================================
    cameras, camera_resolutions_raw, camera_fps_raw = initialize_cameras(shared_state=shared_state)
    
    # Normalizar las claves de cámara en mayúsculas para mantener la misma convención.
    camera_resolutions = {k.upper(): v for k, v in camera_resolutions_raw.items()}
    camera_fps = {k.upper(): v for k, v in camera_fps_raw.items()}

    for cap_obj in cameras.values():
        cap_obj.shared_state = shared_state

    historial_secuencias = {cam_name.upper(): deque() for cam_name in cameras}
    historial_predicciones = {cam_name.upper(): deque(maxlen=8) for cam_name in cameras}
    ultimo_score_comportamiento = {cam_name.upper(): 0.0 for cam_name in cameras}
    ultimo_frame_procesado = {cam_name.upper(): None for cam_name in cameras}
    cola_inferencia_comportamiento = queue.Queue(maxsize=max(8, len(cameras) * 2))
    cola_resultados_comportamiento = queue.Queue()
    inferencia_comportamiento_en_vuelo = {
        cam_name.upper(): False for cam_name in cameras
    }
    resultados_comportamiento_pendientes = {}
    ultimo_error_inferencia_comportamiento = {
        cam_name.upper(): 0.0 for cam_name in cameras
    }

    # ======================================================
    # WORKER DE INFERENCIA EN LOTE PARA COMPORTAMIENTO (TSM)
    # ======================================================
    def worker_inferencia_comportamiento():
        while True:
            batch_tasks = []
            try:
                primero = cola_inferencia_comportamiento.get()
                batch_tasks.append(primero)
                # Drenar rápidamente cualquier otra cámara que haya entrado a la cola
                while not cola_inferencia_comportamiento.empty():
                    try:
                        batch_tasks.append(cola_inferencia_comportamiento.get_nowait())
                    except queue.Empty:
                        break
            except Exception:
                continue

            if not batch_tasks:
                continue

            # Extraer modelo y empaquetar para evaluación en lote
            modelo_actual = batch_tasks[0][2]
            items_eval = [(item[0], item[3], item[4]) for item in batch_tasks]

            try:
                resultados = evaluar_secuencias_violencia_batch(
                    modelo_actual, dispositivo_ia, items_eval
                )
                for idx, (cam_k, clase, score) in enumerate(resultados):
                    gen = batch_tasks[idx][1]
                    cola_resultados_comportamiento.put(
                        (cam_k, gen, clase, score, None)
                    )
            except Exception as error:
                for item in batch_tasks:
                    cola_resultados_comportamiento.put(
                        (item[0], item[1], None, 0.0, error)
                    )
            finally:
                for _ in batch_tasks:
                    cola_inferencia_comportamiento.task_done()

    threading.Thread(
        target=worker_inferencia_comportamiento,
        daemon=True,
        name="BehaviorInferenceWorker",
    ).start()

    # Temporizador para inferencia TSM (cada 200 ms) y longitud del buffer temporal
    HISTORIAL_COMPORTAMIENTO_SEG = 1.5
    ultimo_tiempo_inferencia = {cam_name.upper(): 0.0 for cam_name in cameras}
    INTERVALO_INFERENCIA_SEG = 0.20

    # Retención de alerta (Hysteresis de 2.5s)
    ultimo_tiempo_alerta_violencia = {cam_name.upper(): 0.0 for cam_name in cameras}
    TIEMPO_RETENCION_ALERTA_SEG = 2.5

    # Ventanas temporales inicializadas con claves en MAYÚSCULAS
    windows_armas = initialize_windows(camera_fps, config.WINDOW_SECONDS)
    alert_state_armas = {cam_name.upper(): False for cam_name in cameras}
    alertas_enviadas_evento = {cam_name.upper(): set() for cam_name in cameras}
    recording_state = initialize_recording_state(cameras, config.PRE_BUFFER_SECONDS)

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

    def limpieza_segura():
        print("\n[INFO] Liberando los recursos del sistema.")
        for cap in cameras.values():
            if hasattr(cap, 'release'): cap.release()
            elif hasattr(cap, 'stop'): cap.stop()
            
        for streamer in streamers.values():
            streamer.cerrar()
            
        cv2.destroyAllWindows()

    atexit.register(limpieza_segura)

    for cam_name in cameras:
        shared_state.emitir_evento_dashboard('camera_status', {
            "camera": cam_name.lower(), 
            "status": "analyzing"
        })

    TARGET_FPS = 30.0
    TARGET_FRAME_TIME = 1.0 / TARGET_FPS
    tiempo_anterior = time.monotonic()
    fps_mostrar = 0.0
    ultimo_sync_camaras = 0.0

    # ======================================================
    # 3. BUCLE PRINCIPAL DE INFERENCIA CONTINUA
    # ======================================================
    while True:
        inicio_ciclo = time.monotonic()

        # Sincronización dinámica de cámaras en caliente cada 2.5 segundos
        if inicio_ciclo - ultimo_sync_camaras >= 2.5:
            ultimo_sync_camaras = inicio_ciclo
            try:
                config_cams = config.obtener_camaras_configuradas()
                
                # A) Agregar nuevas cámaras registradas en la web
                for c_name, c_source in config_cams.items():
                    c_upper = c_name.upper()
                    if c_name not in cameras:
                        print(f"[INFO] Agregando nueva cámara al sistema en caliente: {c_name} -> {c_source}")
                        cap = CameraStream(c_name, c_source, shared_state=shared_state)
                        cameras[c_name] = cap
                        camera_resolutions[c_upper] = (cap.width, cap.height)
                        camera_fps[c_upper] = cap.fps
                        
                        target_w = 800
                        target_h = int((target_w / max(1, cap.width)) * cap.height)
                        if target_h % 2 != 0: target_h += 1
                        
                        streamers[c_upper] = RTSPStreamer(
                            c_name, width=target_w, height=target_h, fps=15, shared_state=shared_state
                        )
                        
                        historial_secuencias[c_upper] = deque()
                        historial_predicciones[c_upper] = deque(maxlen=8)
                        ultimo_score_comportamiento[c_upper] = 0.0
                        ultimo_frame_procesado[c_upper] = None
                        inferencia_comportamiento_en_vuelo[c_upper] = False
                        ultimo_error_inferencia_comportamiento[c_upper] = 0.0
                        ultimo_tiempo_inferencia[c_upper] = 0.0
                        ultimo_tiempo_alerta_violencia[c_upper] = 0.0
                        windows_armas[c_upper] = {
                            "events": deque(),
                            "window_seconds": max(0.1, float(config.WINDOW_SECONDS)),
                            "nominal_fps": max(1.0, float(cap.fps)),
                        }
                        alert_state_armas[c_upper] = False
                        alertas_enviadas_evento[c_upper] = set()
                        recording_state[c_upper] = inicializar_camara_grabacion(c_upper, config.PRE_BUFFER_SECONDS)
                        
                        shared_state.emitir_evento_dashboard('camera_status', {
                            "camera": c_name.lower(), 
                            "status": "analyzing"
                        })
                        shared_state.emitir_evento_dashboard('system_log', {
                            "type": "success",
                            "message": f"Cámara '{c_name}' activada y transmitiendo en vivo."
                        })

                # B) Remover cámaras eliminadas
                cams_a_eliminar = [c_name for c_name in list(cameras.keys()) if c_name not in config_cams]
                for c_name in cams_a_eliminar:
                    c_upper = c_name.upper()
                    print(f"[INFO] Removiendo cámara del sistema: {c_name}")
                    try: cameras[c_name].release()
                    except Exception: pass
                    try: streamers[c_upper].cerrar()
                    except Exception: pass
                    cameras.pop(c_name, None)
                    streamers.pop(c_upper, None)
                    camera_resolutions.pop(c_upper, None)
                    camera_fps.pop(c_upper, None)
                    historial_secuencias.pop(c_upper, None)
                    historial_predicciones.pop(c_upper, None)
                    ultimo_score_comportamiento.pop(c_upper, None)
                    ultimo_frame_procesado.pop(c_upper, None)
                    inferencia_comportamiento_en_vuelo.pop(c_upper, None)
                    ultimo_error_inferencia_comportamiento.pop(c_upper, None)
                    ultimo_tiempo_inferencia.pop(c_upper, None)
                    ultimo_tiempo_alerta_violencia.pop(c_upper, None)
                    windows_armas.pop(c_upper, None)
                    alert_state_armas.pop(c_upper, None)
                    alertas_enviadas_evento.pop(c_upper, None)
                    recording_state.pop(c_upper, None)
            except Exception as e:
                print(f"[WARN] Error sincronizando cámaras: {e}")

        frames_list = []
        cam_names_list = []
        frame_timestamps_list = []
        comportamiento_habilitado = shared_state.config_ram.get(
            "cfg_comportamiento", False
        )

        if not comportamiento_habilitado:
            for history in historial_secuencias.values():
                history.clear()

        for cam_name, cap in cameras.items():
            res = cap.read()
            frame = res[1] if isinstance(res, tuple) else res
            frame_id = res[2] if isinstance(res, tuple) and len(res) >= 3 else None
            frame_timestamp = (
                res[3]
                if isinstance(res, tuple) and len(res) >= 4
                else time.monotonic()
            )
            
            if frame is not None:
                cam_upper = cam_name.upper()

                # CameraStream conserva el último frame para lecturas rápidas.
                # No volver a ejecutar IA si aún no llegó una captura nueva.
                if (
                    frame_id is not None
                    and frame_id == ultimo_frame_procesado[cam_upper]
                ):
                    continue
                ultimo_frame_procesado[cam_upper] = frame_id

                frames_list.append(frame)
                cam_names_list.append(cam_upper)
                frame_timestamps_list.append(frame_timestamp)
                camera_resolutions[cam_upper] = (frame.shape[1], frame.shape[0])

                if comportamiento_habilitado:
                    frame_small = cv2.resize(
                        frame, (224, 224), interpolation=cv2.INTER_AREA
                    )
                    history = historial_secuencias[cam_upper]
                    history.append((frame_timestamp, frame_small))
                    cutoff = frame_timestamp - HISTORIAL_COMPORTAMIENTO_SEG
                    while history and history[0][0] < cutoff:
                        history.popleft()

        tiempo_actual = time.monotonic()
        dt_ciclo = tiempo_actual - tiempo_anterior
        tiempo_anterior = tiempo_actual
        dt_ciclo = tiempo_actual - tiempo_anterior
        tiempo_anterior = tiempo_actual
        
        if dt_ciclo > 0:
            fps_inst = 1.0 / dt_ciclo
            fps_mostrar = (0.9 * fps_mostrar) + (0.1 * fps_inst)

        # ======================================================
        # GESTIÓN CONTROLADA DE MODELOS (HOT-SWAP DINÁMICO)
        # ======================================================
        cfg_armas_actual = shared_state.config_ram.get("cfg_armas", False)
        modelo_armas_actual = shared_state.config_ram.get("cfg_modelo_armas", config.WEAPON_DEFAULT_MODEL)

        cambio_armas = (cfg_armas_actual != estado_previo_armas) or (cfg_armas_actual and modelo_armas_actual != modelo_previo_armas)
        if cambio_armas:
            estado_previo_armas = cfg_armas_actual
            modelo_previo_armas = modelo_armas_actual
            generacion_armas += 1
            if cfg_armas_actual:
                ruta_armas = config.resolver_ruta_modelo(modelo_armas_actual, "armas")
                threading.Thread(
                    target=cargar_armas_en_segundo_plano,
                    args=(generacion_armas, ruta_armas),
                    daemon=True,
                    name="ModelLoader-Weapons",
                ).start()
                shared_state.emitir_evento_dashboard('system_log', {
                    "type": "info", "message": f"Cargando modelo de armas: {os.path.basename(ruta_armas)}..."
                })
            else:
                weapon_model = None
                for win in windows_armas.values():
                    win["events"].clear()
                for cam_u in alert_state_armas:
                    alert_state_armas[cam_u] = False
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        cfg_comp_actual = shared_state.config_ram.get("cfg_comportamiento", False)
        modelo_comp_actual = shared_state.config_ram.get("cfg_modelo_comportamiento", config.BEHAVIOR_DEFAULT_MODEL)

        cambio_comp = (cfg_comp_actual != estado_previo_comportamiento) or (cfg_comp_actual and modelo_comp_actual != modelo_previo_comportamiento)
        if cambio_comp:
            estado_previo_comportamiento = cfg_comp_actual
            modelo_previo_comportamiento = modelo_comp_actual
            generacion_comportamiento += 1
            if cfg_comp_actual:
                ruta_comp = config.resolver_ruta_modelo(modelo_comp_actual, "comportamiento")
                threading.Thread(
                    target=cargar_comportamiento_en_segundo_plano,
                    args=(generacion_comportamiento, ruta_comp),
                    daemon=True,
                    name="ModelLoader-Behavior",
                ).start()
                shared_state.emitir_evento_dashboard('system_log', {
                    "type": "info", "message": f"Cargando modelo de comportamiento: {os.path.basename(ruta_comp)}..."
                })
            else:
                behavior_model = None
                for history in historial_secuencias.values():
                    history.clear()
                for predictions in historial_predicciones.values():
                    predictions.clear()
                resultados_comportamiento_pendientes.clear()
                for cam_upper in inferencia_comportamiento_en_vuelo:
                    inferencia_comportamiento_en_vuelo[cam_upper] = False
                for cam_upper in ultimo_score_comportamiento:
                    ultimo_score_comportamiento[cam_upper] = 0.0
                for cam_upper in ultimo_tiempo_alerta_violencia:
                    ultimo_tiempo_alerta_violencia[cam_upper] = 0.0
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        while True:
            try:
                generacion, modelo_cargado, error, nombre_arch = cola_carga_armas.get_nowait()
            except queue.Empty:
                break
            if generacion != generacion_armas or not cfg_armas_actual:
                continue
            if error is None:
                weapon_model = modelo_cargado
                shared_state.emitir_evento_dashboard('system_log', {
                    "type": "success", "message": f"Modelo de detección de armas '{nombre_arch}' activado."
                })
            else:
                weapon_model = None
                shared_state.emitir_evento_dashboard('system_log', {
                    "type": "error", "message": f"No se pudo cargar el modelo de armas '{nombre_arch}': {error}"
                })

        while True:
            try:
                generacion, modelo_cargado, error, nombre_arch = cola_carga_comportamiento.get_nowait()
            except queue.Empty:
                break
            if generacion != generacion_comportamiento or not cfg_comp_actual:
                continue
            if error is None:
                behavior_model = modelo_cargado
                shared_state.emitir_evento_dashboard('system_log', {
                    "type": "success", "message": f"Modelo de comportamiento '{nombre_arch}' activado."
                })
            else:
                behavior_model = None
                shared_state.emitir_evento_dashboard('system_log', {
                    "type": "error", "message": f"No se pudo cargar el modelo de comportamiento '{nombre_arch}': {error}"
                })

        while True:
            try:
                (
                    cam_resultado,
                    generacion_resultado,
                    clase_resultado,
                    score_resultado,
                    error_resultado,
                ) = cola_resultados_comportamiento.get_nowait()
            except queue.Empty:
                break

            if (
                generacion_resultado != generacion_comportamiento
                or not cfg_comp_actual
            ):
                continue

            inferencia_comportamiento_en_vuelo[cam_resultado] = False
            resultados_comportamiento_pendientes[cam_resultado] = (
                clase_resultado,
                score_resultado,
                error_resultado,
            )

        if not frames_list:
            time.sleep(0.005)
            continue

        # ======================================================
        # INFERENCIA DE ARMAS EN LOTE (Todas las cámaras continuas)
        # ======================================================
        modo_debug = shared_state.config_ram.get("cfg_debug", False)
        weapon_results = [None] * len(frames_list)
        alertas_armas_batch = [False] * len(frames_list)

        if cfg_armas_actual and weapon_model is not None and frames_list:
            raw_w_results, raw_alertas = batch_detect_weapons(
                weapon_model, frames_list, 
                conf=shared_state.config_ram.get("cfg_confianza_armas", 0.50), 
                clases_alerta=config.CLASES_ARMAS_ALERTA, 
                modo_debug=modo_debug
            )
            weapon_results = raw_w_results
            alertas_armas_batch = raw_alertas

        # ======================================================
        # PROCESAMIENTO Y ANALÍTICA POR CÁMARA
        # ======================================================
        for i, cam_upper in enumerate(cam_names_list):
            frame = frames_list[i]
            frame_timestamp = frame_timestamps_list[i]
            w_res = weapon_results[i]

            weapon_in_frame = False
            comportamiento_anomalo = False
            nombre_comportamiento = ""

            if cfg_armas_actual and weapon_model is not None:
                weapon_in_frame = alertas_armas_batch[i]

            # EVALUACIÓN DE VIOLENCIA TEMPORAL (TSM)
            if cfg_comp_actual and behavior_model is not None:
                umbral_actual = shared_state.config_ram.get("cfg_confianza_comportamiento", 0.50)

                resultado_nuevo = resultados_comportamiento_pendientes.pop(
                    cam_upper, None
                )
                if resultado_nuevo is not None:
                    clase_comportamiento, score, error_inferencia = resultado_nuevo
                    if error_inferencia is None:
                        es_violencia_frame = 1 if (
                            clase_comportamiento == "VIOLENCE"
                        ) else 0
                        historial_predicciones[cam_upper].append(
                            es_violencia_frame
                        )
                        ultimo_score_comportamiento[cam_upper] = float(score)

                        # Actualizar la alerta únicamente ante una nueva confirmación positiva real
                        alertas_activas = sum(historial_predicciones[cam_upper])
                        total_evaluaciones = len(historial_predicciones[cam_upper])
                        if es_violencia_frame == 1 and total_evaluaciones >= 2 and alertas_activas >= 2 and (alertas_activas / total_evaluaciones) >= 0.30:
                            ultimo_tiempo_alerta_violencia[cam_upper] = tiempo_actual

                    elif (
                        tiempo_actual
                        - ultimo_error_inferencia_comportamiento.get(cam_upper, 0.0)
                        >= 10.0
                    ):
                        ultimo_error_inferencia_comportamiento[cam_upper] = tiempo_actual
                        shared_state.emitir_evento_dashboard('system_log', {
                            "type": "error",
                            "message": f"Falló el análisis de comportamiento: {error_inferencia}",
                        })

                # Evaluar TSM según el intervalo temporal configurado
                if (
                    (tiempo_actual - ultimo_tiempo_inferencia.get(cam_upper, 0.0) >= INTERVALO_INFERENCIA_SEG)
                    and not inferencia_comportamiento_en_vuelo.get(cam_upper, False)
                ):
                    try:
                        cola_inferencia_comportamiento.put_nowait((
                            cam_upper,
                            generacion_comportamiento,
                            behavior_model,
                            list(historial_secuencias[cam_upper]),
                            umbral_actual,
                        ))
                        inferencia_comportamiento_en_vuelo[cam_upper] = True
                        ultimo_tiempo_inferencia[cam_upper] = tiempo_actual
                    except queue.Full:
                        pass

                if (tiempo_actual - ultimo_tiempo_alerta_violencia.get(cam_upper, 0.0)) < TIEMPO_RETENCION_ALERTA_SEG:
                    comportamiento_anomalo = True
                    nombre_comportamiento = "VIOLENCIA"

            # Ventana temporal para armas
            alerta_arma = update_window(
                cam_upper, weapon_in_frame, windows_armas, 
                config.ACTIVATION_THRESHOLD, alert_state_armas,
                timestamp=frame_timestamp,
            ) if (cfg_armas_actual and weapon_model is not None) else False

            alert_triggered = (alerta_arma if (cfg_armas_actual and weapon_model is not None) else False) or (comportamiento_anomalo if cfg_comp_actual else False)
            amenaza_presente = (weapon_in_frame if (cfg_armas_actual and weapon_model is not None) else False) or (comportamiento_anomalo if cfg_comp_actual else False)

            # OVERLAYS HUD
            if cfg_armas_actual and weapon_model is not None and w_res and len(w_res.boxes) > 0:
                frame = w_res.plot(img=frame)

            score_comp_actual = ultimo_score_comportamiento.get(cam_upper, 0.0) if cfg_comp_actual else 0.0

            # PERSISTENCIA (Grabación de evidencia con OSD proporcional al fotograma nativo)
            estaba_grabando = recording_state[cam_upper]["recording"]

            frame_grabacion = dibujar_osd_proporcional(
                frame.copy(), cam_upper, 
                alerta_comportamiento=(comportamiento_anomalo if cfg_comp_actual else False), 
                score_comp=score_comp_actual
            )

            handle_recording(
                cam_upper, frame_grabacion, camera_resolutions, recording_state,
                shared_state.config_ram.get("cfg_postbuffer", 15), 
                alert_triggered, amenaza_presente, shared_state=shared_state,
                pre_buffer_seconds=shared_state.config_ram.get("cfg_prebuffer", 10),
                frame_timestamp=frame_timestamp,
            )

            esta_grabando_actualmente = recording_state[cam_upper]["recording"]

            # SINCRONIZACIÓN DE ESTADO EN DASHBOARD
            if esta_grabando_actualmente and not estaba_grabando:
                shared_state.emitir_evento_dashboard('camera_status', {
                    "camera": cam_upper.lower(), "status": "detecting"
                })

            elif not esta_grabando_actualmente and estaba_grabando:
                alertas_enviadas_evento[cam_upper].clear()
                shared_state.emitir_evento_dashboard('camera_status', {
                    "camera": cam_upper.lower(), "status": "analyzing"
                })

            # EVALUACIÓN CONTINUA Y DESPACHO DE ALERTAS
            if esta_grabando_actualmente:
                # Alerta por Confirmación de Arma de Fuego
                if cfg_armas_actual and alerta_arma and "ARMA" not in alertas_enviadas_evento[cam_upper]:
                    print(f"\n[ALERTA] Presencia de arma confirmada en {cam_upper}.")
                    shared_state.emitir_evento_dashboard('system_log', {
                        "type": "error",
                        "message": f"Alerta de seguridad: arma de fuego confirmada en {cam_upper}."
                    })
                    threading.Thread(
                        target=_ejecutar_alerta_segura,
                        args=(cam_upper, "Arma detectada", "Presencia de objeto peligroso confirmada", shared_state),
                        daemon=True
                    ).start()
                    alertas_enviadas_evento[cam_upper].add("ARMA")

                # Alerta por Comportamiento Hostil
                if cfg_comp_actual and comportamiento_anomalo and nombre_comportamiento not in alertas_enviadas_evento[cam_upper]:
                    print(f"\n[ALERTA] Conducta hostil confirmada en {cam_upper}.")
                    shared_state.emitir_evento_dashboard('system_log', {
                        "type": "error",
                        "message": f"Alerta de seguridad: comportamiento hostil confirmado en {cam_upper}."
                    })
                    threading.Thread(
                        target=_ejecutar_alerta_segura,
                        args=(cam_upper, "Comportamiento hostil", f"Actividad sospechosa detectada como {nombre_comportamiento.lower()}", shared_state),
                        daemon=True
                    ).start()
                    alertas_enviadas_evento[cam_upper].add(nombre_comportamiento)
            else:
                # Si la cámara ya no está grabando, limpiar registro de eventos enviados
                alertas_enviadas_evento[cam_upper].clear()

            # TRANSMISIÓN RTSP (Estandarizado a la cuadrícula: 800 x target_h con OSD nítido)
            if cam_upper in streamers:
                target_w = streamers[cam_upper].width
                target_h = streamers[cam_upper].height
                frame_stream = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                
                # Dibujar OSD nítido con dimensiones uniformes sobre la resolución de salida
                frame_stream = dibujar_osd_cuadricula(
                    frame_stream, cam_upper, 
                    alerta_comportamiento=(comportamiento_anomalo if cfg_comp_actual else False), 
                    score_comp=score_comp_actual
                )
                
                if shared_state.config_ram.get("cfg_debug", False):
                    frame_stream = draw_performance_overlay(frame_stream, fps_mostrar)
                    
                streamers[cam_upper].enviar_frame(frame_stream)
            
            # ESTADO GLOBAL
            cam_key_lower = cam_upper.lower()
            cam_obj = cameras.get(cam_key_lower)
            if cam_obj and getattr(cam_obj, 'estado_error_enviado', False):
                _registro_estados_global[cam_key_lower] = "error"
            elif esta_grabando_actualmente:
                _registro_estados_global[cam_key_lower] = "detecting"
            else:
                _registro_estados_global[cam_key_lower] = "analyzing"

        # Regulador de ritmo
        tiempo_transcurrido = time.monotonic() - inicio_ciclo
        tiempo_espera = TARGET_FRAME_TIME - tiempo_transcurrido
        if tiempo_espera > 0:
            time.sleep(tiempo_espera)


def obtener_mapa_estados_actual():
    global _registro_estados_global
    return dict(_registro_estados_global)
