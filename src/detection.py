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
from config import MODEL_PATH, COLOR_GUN, COLOR_KNIFE, POSE_MODEL_PATH, UMBRAL_PUNETAZO
import math

# =========================
# MEMORIA DE MOVIMIENTO (Multi-Persona)
# =========================
# Ahora guardamos un diccionario por cada ID detectado.
# Ejemplo: { 1: {"muneca_der_ant": (x,y), "muneca_izq_ant": (x,y)}, 2: {...} }
estado_postura = {}


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
def detect_pose(model, frame, conf_threshold=0.5):
    
    # CAMBIO 1: Usamos .track() en lugar de la inferencia normal.
    # persist=True le dice al modelo que conecte los frames temporalmente.
    # tracker="bytetrack.yaml" usa el algoritmo optimizado para vigilancia.
    results = model.track(
        frame, 
        conf=conf_threshold, 
        persist=True, 
        tracker="bytetrack.yaml", 
        verbose=False
    )

    asalto_detectado = False
    golpe_detectado = False

    if len(results[0].boxes) > 0 and results[0].boxes.id is not None:
        frame = results[0].plot() 
        keypoints_personas = results[0].keypoints.xy 
        ids_personas = results[0].boxes.id.int().cpu().tolist() 
        ids_activos_este_frame = []

        for persona_kp, track_id in zip(keypoints_personas, ids_personas):
            ids_activos_este_frame.append(track_id)
            
            if track_id not in estado_postura:
                estado_postura[track_id] = {"muneca_der_ant": None, "muneca_izq_ant": None}

            if len(persona_kp) >= 11:
                x_muneca_izq = float(persona_kp[9][0])
                y_muneca_izq = float(persona_kp[9][1])
                x_muneca_der = float(persona_kp[10][0])
                y_muneca_der = float(persona_kp[10][1])
                y_ojo_izq = float(persona_kp[1][1])
                y_ojo_der = float(persona_kp[2][1])

                if y_muneca_izq > 0 and y_muneca_der > 0:
                    
                    # REGLA 1: MANOS ARRIBA
                    if y_ojo_izq > 0 and y_ojo_der > 0:
                        if y_muneca_izq < y_ojo_izq and y_muneca_der < y_ojo_der:
                            asalto_detectado = True
                            cv2.putText(frame, "ALERTA: MANOS ARRIBA", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                    # REGLA 2: MOVIMIENTO BRUSCO
                    memoria_persona = estado_postura[track_id]
                    if memoria_persona["muneca_der_ant"] is not None:
                        dist_der = math.hypot(x_muneca_der - memoria_persona["muneca_der_ant"][0], y_muneca_der - memoria_persona["muneca_der_ant"][1])
                        dist_izq = math.hypot(x_muneca_izq - memoria_persona["muneca_izq_ant"][0], y_muneca_izq - memoria_persona["muneca_izq_ant"][1])

                        # Usamos el umbral importado de config.py
                        if dist_der > UMBRAL_PUNETAZO or dist_izq > UMBRAL_PUNETAZO:
                            golpe_detectado = True
                            cv2.putText(frame, "ALERTA: GOLPE", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                    estado_postura[track_id]["muneca_der_ant"] = (x_muneca_der, y_muneca_der)
                    estado_postura[track_id]["muneca_izq_ant"] = (x_muneca_izq, y_muneca_izq)

        # Limpieza
        ids_a_borrar = [tid for tid in estado_postura if tid not in ids_activos_este_frame]
        for tid in ids_a_borrar:
            del estado_postura[tid]

    # Devolvemos ambas variables por separado
    return asalto_detectado, golpe_detectado, frame