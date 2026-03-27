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
from config import MODEL_PATH, COLOR_GUN, COLOR_KNIFE, POSE_MODEL_PATH


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
    """
    Ejecuta detección de postura, dibuja esqueletos y evalúa comportamientos.
    Retorna: (comportamiento_detectado, frame)
    """
    results = model(frame, conf=conf_threshold, verbose=False)

    comportamiento_sospechoso = False

    if len(results[0].boxes) > 0:
        # 1. Dibujar el esqueleto automáticamente
        frame = results[0].plot()

        # 2. Extraer los puntos clave (keypoints) de todas las personas detectadas
        # results[0].keypoints.xy contiene las coordenadas (X, Y)
        keypoints_personas = results[0].keypoints.xy 

        for persona_kp in keypoints_personas:
            # persona_kp tiene 17 puntos. Verificamos que tenga suficientes datos.
            if len(persona_kp) >= 11:
                
                # Extraer coordenada Y (índice 1) de hombros y muñecas
                # Nota: El índice 0 sería la coordenada X
                y_hombro_izq = float(persona_kp[5][1])
                y_hombro_der = float(persona_kp[6][1])
                y_muneca_izq = float(persona_kp[9][1])
                y_muneca_der = float(persona_kp[10][1])

                # Validación: Si YOLO no ve una muñeca, su valor Y suele ser 0.0
                if y_muneca_izq > 0 and y_muneca_der > 0 and y_hombro_izq > 0 and y_hombro_der > 0:
                    
                    # REGLA: ¿Ambas muñecas están más altas (menor Y) que los hombros?
                    if y_muneca_izq < y_hombro_izq and y_muneca_der < y_hombro_der:
                        comportamiento_sospechoso = True
                        
                        # Dibujar una advertencia visual muy clara en pantalla
                        cv2.putText(
                            frame, 
                            "ALERTA: MANOS ARRIBA", 
                            (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1.5, 
                            (0, 0, 255), # Rojo
                            4
                        )

    return comportamiento_sospechoso, frame