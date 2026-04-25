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
import os       
import psutil   
import time

# Importa las nuevas variables
from config import (
    CONF_WEAPON,
    WINDOW_SECONDS,
    ACTIVATION_THRESHOLD,
    BEHAVIOR_WINDOW_SECONDS,
    BEHAVIOR_ACTIVATION_THRESHOLD,
    GOLPE_ACTIVATION_THRESHOLD,
    PRE_BUFFER_SECONDS,
    POST_BUFFER_SECONDS
)

from cameras import initialize_cameras, read_frames
from detection import load_weapon_model, detect_weapons, load_pose_model, detect_pose
from temporal_logic import initialize_windows, update_window
from recorder import initialize_recording_state, handle_recording
from alerts import send_alert


def main():

    print("🔧 Inicializando sistema...")

    # =========================
    # Cargar modelo
    # =========================
    weapon_model = load_weapon_model()
    pose_model = load_pose_model()

    # =========================
    # Inicializar cámaras
    # =========================
    cameras, camera_resolutions, camera_fps = initialize_cameras()

    # =========================
    # Inicializar ventanas temporales (SEPARADAS)
    # =========================
    windows_armas = initialize_windows(camera_fps, WINDOW_SECONDS)
    windows_asalto = initialize_windows(camera_fps, BEHAVIOR_WINDOW_SECONDS)
    windows_golpe = initialize_windows(camera_fps, BEHAVIOR_WINDOW_SECONDS) # Usa la misma duración que asalto o crea una nueva

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
    recording_state = initialize_recording_state(cameras, PRE_BUFFER_SECONDS)

    print("▶ Sistema iniciado. Presiona 'q' para salir.")

    # =========================
    # LOOP PRINCIPAL
    # =========================
    while True:

        frames = read_frames(cameras)

        for cam_name, frame in frames.items():

            # -------- Detección de Armas --------
            weapon_in_frame, frame = detect_weapons(
                weapon_model,
                frame,
                CONF_WEAPON
            )

            # -------- Detección de Postura --------
            # Recibimos las variables separadas
            asalto_detectado, golpe_detectado, caida_detectada, frame = detect_pose(pose_model, frame, 0.5)

            # ==================================================
            # LÓGICA TEMPORAL 
            # ==================================================
            alerta_arma = update_window(cam_name, weapon_in_frame, windows_armas, ACTIVATION_THRESHOLD, alert_state_armas)
            alerta_asalto = update_window(cam_name, asalto_detectado, windows_asalto, BEHAVIOR_ACTIVATION_THRESHOLD, alert_state_asalto)
            alerta_caida = update_window(cam_name, caida_detectada, windows_golpe, GOLPE_ACTIVATION_THRESHOLD, alert_state_golpe)
            
            # Usamos el umbral casi inmediato para los golpes
            alerta_golpe = update_window(
                cam_name, 
                golpe_detectado, 
                windows_golpe, 
                GOLPE_ACTIVATION_THRESHOLD, # <--- ¡Cambiamos esto!
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
            # 1. Calcular FPS reales
            tiempo_actual = time.time()
            # Evitamos división por cero en el primer frame
            if tiempo_anterior > 0:
                fps_real = 1.0 / (tiempo_actual - tiempo_anterior)
            else:
                fps_real = 0.0
            tiempo_anterior = tiempo_actual

            # 2. Obtener datos de RAM y CPU
            proceso = psutil.Process(os.getpid())
            ram_mb = proceso.memory_info().rss / (1024 * 1024)
            ram_app_pct = proceso.memory_percent()
            ram_pc_pct = psutil.virtual_memory().percent
            cpu_percent = psutil.cpu_percent()

            # 3. Crear el texto a mostrar (Ahora incluyendo los FPS)
            texto_rendimiento = f"FPS: {int(fps_real)} | RAM App: {ram_mb:.1f}MB ({ram_app_pct:.1f}%) | RAM PC: {ram_pc_pct}% | CPU: {cpu_percent}%"

            # 2. Configuración visual del texto
            fuente = cv2.FONT_HERSHEY_SIMPLEX
            escala_fuente = 0.5         # Letra más pequeña (antes 0.6)
            grosor_fuente = 1           # Más delgada
            color_texto = (200, 200, 200) # Gris claro en formato (B, G, R)
            
            # 3. Calcular el tamaño exacto del texto para hacer el rectángulo a medida
            (ancho_texto, alto_texto), baseline = cv2.getTextSize(texto_rendimiento, fuente, escala_fuente, grosor_fuente)
            
            # Coordenadas donde empezará el texto
            x, y = 5, 5 + alto_texto
            
            # Coordenadas del rectángulo de fondo (con un poco de margen)
            rect_x1, rect_y1 = x - 5, y - alto_texto - 5
            rect_x2, rect_y2 = x + ancho_texto + 5, y + baseline + 5

            # 4. Crear el fondo negro semitransparente
            overlay = frame.copy()
            cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1) # -1 rellena el rectángulo de negro
            
            # Mezclar el overlay con el frame original (0.5 = 50% de opacidad)
            opacidad = 0.5 
            cv2.addWeighted(overlay, opacidad, frame, 1 - opacidad, 0, frame)

            # 5. Dibujar el texto gris encima del fondo ya mezclado
            cv2.putText(
                frame, 
                texto_rendimiento, 
                (x, y), 
                fuente, 
                escala_fuente, 
                color_texto, 
                grosor_fuente,
                cv2.LINE_AA # Suaviza los bordes de las letras
            )
            # ==========================================

            # -------- Grabación --------
            handle_recording(
                cam_name,
                frame,
                camera_resolutions,
                recording_state,
                POST_BUFFER_SECONDS, # Pasamos el tiempo de post-buffer
                alert_triggered,
                amenaza_presente
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