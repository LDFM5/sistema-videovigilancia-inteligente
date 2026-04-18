"""
detection.py

Módulo encargado de la detección de armas utilizando un modelo YOLO entrenado.

Funciones principales:
- Cargar el modelo de detección.
- Ejecutar inferencia sobre cada frame.
- Dibujar bounding boxes y etiquetas.
- Indicar si se detectó un arma en el frame actual.

Este módulo no toma decisiones de alerta ni maneja grabación;
su responsabilidad es exclusivamente la detección.
"""
import cv2
from ultralytics import YOLO
from config import MODEL_PATH, COLOR_GUN, COLOR_KNIFE, POSE_MODEL_PATH, UMBRAL_VELOCIDAD_GOLPE
import math
import time

# Diccionario de posiciones
estado_postura = {}

# Memoria para mantener el texto en pantalla
tiempo_ultimo_golpe = 0

# =========================
# FUNCIONES AUXILIARES
# =========================
def obtener_longitud_torso(kp):
    """Calcula la distancia entre el centro de los hombros y el centro de las caderas."""
    # Hombros: 5 (izq), 6 (der) | Caderas: 11 (izq), 12 (der)
    if len(kp) >= 13 and kp[5][1] > 0 and kp[6][1] > 0 and kp[11][1] > 0 and kp[12][1] > 0:
        centro_hombros_x = (kp[5][0] + kp[6][0]) / 2
        centro_hombros_y = (kp[5][1] + kp[6][1]) / 2
        centro_caderas_x = (kp[11][0] + kp[12][0]) / 2
        centro_caderas_y = (kp[11][1] + kp[12][1]) / 2
        
        return math.hypot(centro_hombros_x - centro_caderas_x, centro_hombros_y - centro_caderas_y)
    return None


# =========================
# CARGAR MODELO
# =========================

def load_weapon_model():
    """
    Carga el modelo de armas desde la ruta definida en config.
    """
    model = YOLO(MODEL_PATH)
    print("✅ Modelo de armas cargado correctamente.")
    return model

def load_pose_model():
    """
    Carga el modelo de estimación de postura.
    """
    model = YOLO(POSE_MODEL_PATH)
    print("✅ Modelo de Pose cargado correctamente.")
    return model

# =========================
# DETECCIÓN DE ARMA
# =========================

def detect_weapons(model, frame, conf_threshold):
    """
    Ejecuta detección de armas sobre un frame.

    Retorna:
        weapon_detected (bool)
        frame (con cajas dibujadas)
    """

    results = model(
        frame,
        conf=conf_threshold,
        verbose=False
    )

    weapon_detected = False

    for r in results:
        for box in r.boxes:
            weapon_detected = True

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # Selección de color según tipo
            color = COLOR_GUN if label == "gun" else COLOR_KNIFE

            # Dibujar caja
            frame[x1:x1]  # Dummy access to avoid lint warnings (no effect)

            import cv2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                frame,
                f"{label.upper()} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

    return weapon_detected, frame

# =========================
# DETECCIÓN DE POSE Y COMPORTAMIENTO
# =========================
# (Mantén tus importaciones y la función obtener_longitud_torso igual)

def detect_pose(model, frame, conf_threshold=0.5):
    global tiempo_ultimo_golpe 
    
    results = model.track(frame, conf=conf_threshold, persist=True, tracker="bytetrack.yaml", verbose=False)

    asalto_detectado = False
    golpe_detectado = False

    if len(results[0].boxes) > 0 and results[0].boxes.id is not None:
        frame = results[0].plot() 
        keypoints_personas = results[0].keypoints.xy 
        ids_personas = results[0].boxes.id.int().cpu().tolist() 
        
        ids_activos_este_frame = []

        for persona_kp, track_id in zip(keypoints_personas, ids_personas):
            ids_activos_este_frame.append(track_id)
            
            largo_torso = obtener_longitud_torso(persona_kp)
            
            if largo_torso is None or largo_torso < 10:
                if persona_kp[5][1] > 0 and persona_kp[6][1] > 0:
                    largo_torso = math.hypot(persona_kp[5][0] - persona_kp[6][0], persona_kp[5][1] - persona_kp[6][1]) * 1.5
                else:
                    largo_torso = 100 

            if track_id not in estado_postura:
                estado_postura[track_id] = {
                    "muneca_der_ant": None, 
                    "muneca_izq_ant": None,
                    "tiempo_ant": time.time() 
                }

            if len(persona_kp) >= 11:
                x_muneca_izq = float(persona_kp[9][0])
                y_muneca_izq = float(persona_kp[9][1])
                x_muneca_der = float(persona_kp[10][0])
                y_muneca_der = float(persona_kp[10][1])
                y_ojo_izq = float(persona_kp[1][1])
                y_ojo_der = float(persona_kp[2][1])

                # ==========================================
                # 1. VALIDACIÓN INDEPENDIENTE DE VISIBILIDAD
                # ==========================================
                # YOLO envía (0,0) si no ve la parte del cuerpo
                izq_visible = (x_muneca_izq > 0 and y_muneca_izq > 0)
                der_visible = (x_muneca_der > 0 and y_muneca_der > 0)

                # ==========================================
                # REGLA 1: MANOS ARRIBA (Requiere ambas visibles por seguridad)
                # ==========================================
                if izq_visible and der_visible and y_ojo_izq > 0 and y_ojo_der > 0:
                    if y_muneca_izq < y_ojo_izq and y_muneca_der < y_ojo_der:
                        asalto_detectado = True

                # ==========================================
                # REGLA 2: GOLPE / MOVIMIENTO BRUSCO
                # ==========================================
                memoria = estado_postura[track_id]
                tiempo_actual = time.time()
                delta_t = tiempo_actual - memoria["tiempo_ant"]

                # ESTABILIZADOR: Ignorar si el ciclo fue anormalmente rápido (menor a 20ms)
                if delta_t > 0.02: 
                    
                    # FILTRO ANTI-RUIDO: Ignorar vibraciones menores al 10% del torso
                    distancia_minima_ruido = largo_torso * 0.10

                    # --- EVALUAR MUÑECA DERECHA ---
                    if der_visible and memoria["muneca_der_ant"] is not None:
                        d_der = math.hypot(x_muneca_der - memoria["muneca_der_ant"][0], y_muneca_der - memoria["muneca_der_ant"][1])
                        
                        if d_der > distancia_minima_ruido: # Si superó el ruido, calculamos velocidad
                            vel_der = (d_der / delta_t) / largo_torso
                            if vel_der > UMBRAL_VELOCIDAD_GOLPE:
                                golpe_detectado = True
                                tiempo_ultimo_golpe = tiempo_actual

                    # --- EVALUAR MUÑECA IZQUIERDA ---
                    if izq_visible and memoria["muneca_izq_ant"] is not None:
                        d_izq = math.hypot(x_muneca_izq - memoria["muneca_izq_ant"][0], y_muneca_izq - memoria["muneca_izq_ant"][1])
                        
                        if d_izq > distancia_minima_ruido: # Si superó el ruido, calculamos velocidad
                            vel_izq = (d_izq / delta_t) / largo_torso
                            if vel_izq > UMBRAL_VELOCIDAD_GOLPE:
                                golpe_detectado = True
                                tiempo_ultimo_golpe = tiempo_actual

                # ==========================================
                # ACTUALIZACIÓN SEGURA DE MEMORIA
                # ==========================================
                # Si la mano desaparece, borramos su memoria. 
                # Así evitamos que al reaparecer se marque como un movimiento a la velocidad de la luz.
                if der_visible:
                    estado_postura[track_id]["muneca_der_ant"] = (x_muneca_der, y_muneca_der)
                else:
                    estado_postura[track_id]["muneca_der_ant"] = None

                if izq_visible:
                    estado_postura[track_id]["muneca_izq_ant"] = (x_muneca_izq, y_muneca_izq)
                else:
                    estado_postura[track_id]["muneca_izq_ant"] = None
                    
                estado_postura[track_id]["tiempo_ant"] = tiempo_actual

        # Limpieza
        ids_a_borrar = [tid for tid in estado_postura if tid not in ids_activos_este_frame]
        for tid in ids_a_borrar:
            del estado_postura[tid]

    # Renderizado
    if asalto_detectado:
        cv2.putText(frame, "ALERTA: MANOS ARRIBA", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    if time.time() - tiempo_ultimo_golpe < 1.5:
        cv2.putText(frame, "ALERTA: GOLPE / EMPUJON", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3) 

    return asalto_detectado, golpe_detectado, frame