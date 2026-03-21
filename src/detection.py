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

from ultralytics import YOLO
from config import MODEL_PATH, COLOR_GUN, COLOR_KNIFE


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


# =========================
# DETECCIÓN
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
