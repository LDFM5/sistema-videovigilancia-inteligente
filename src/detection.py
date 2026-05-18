"""
detection.py

Módulo optimizado para inferencia en tiempo real y arquitectura Edge AI.

Responsabilidades:
- Ejecutar inferencia por lotes (batching) usando modelos YOLO proporcionados.
- Filtrar clases de interés (armas).
- Extraer keypoints y dibujar anotaciones.

Nota Arquitectónica: 
Este módulo ya NO calcula lógicas biométricas espaciales. Únicamente retorna 
los tensores puros de pose (skeletons_data) para ser inyectados en un clasificador 
secundario de Machine Learning (MLP/GCN).
"""
import cv2

# =========================
# DETECCIÓN DE ARMAS
# =========================

def batch_detect_weapons(model, frames_list, conf=0.5, clases_alerta=None, modo_debug=False):
    """
    Procesa múltiples frames al mismo tiempo (Batching) para detección de armas.
    
    Args:
        model: Instancia del modelo YOLO cargada en memoria.
        frames_list (list): Lista de frames crudos de OpenCV.
        conf (float): Umbral de confianza.
        clases_alerta (list): Nombres de las clases a detectar.
        modo_debug (bool): Si es True, ignora el filtro y dibuja todo.
        
    Returns:
        list: Resultados de la inferencia por cada frame.
    """
    if not frames_list:
        return []
        
    # 1. Obtener los IDs numéricos de las armas reales (firearm, melee_weapon)
    ids_armas = []
    if clases_alerta:
        ids_armas = [id_clase for id_clase, nombre in model.names.items() if nombre in clases_alerta]

    # 2. Si MODO_DEBUG es True, YOLO detecta todo (None). Si es False, filtra solo armas.
    filtro_clases = None if modo_debug else (ids_armas if ids_armas else None)

    results = model(frames_list, conf=conf, classes=filtro_clases, verbose=False)
    
    # 3. MAGIA DEL FILTRO DE ALERTAS
    # Revisamos frame por frame si entre todo lo que encontró, hay un arma real
    alertas_por_frame = []
    for r in results:
        tiene_arma_real = False
        if r.boxes is not None:
            for cls_id in r.boxes.cls:
                if int(cls_id) in ids_armas:
                    tiene_arma_real = True
                    break  # Con un arma es suficiente para la alerta
        alertas_por_frame.append(tiene_arma_real)
            
    return results, alertas_por_frame


# =========================
# EXTRACCIÓN DE POSE 
# =========================

def batch_detect_pose(model, frames_list, conf=0.5):
    """
    Procesa esqueletos en lote. Retorna los resultados visuales y los datos crudos (Keypoints)
    preparados para inyectarse en un futuro modelo clasificador (MLP/ST-GCN).
    
    Args:
        model: Instancia del modelo YOLO Pose cargada en memoria.
        frames_list (list): Lista de frames crudos de OpenCV.
        conf (float): Umbral de confianza.
        
    Returns:
        tuple: (Resultados visuales, Lista de tensores de skeletons_data)
    """
    if not frames_list:
        return [], []
        
    results = model.track(frames_list, conf=conf, persist=True, tracker="bytetrack.yaml", verbose=False)
    
    skeletons_data = []
    
    for r in results:
        # Extraemos la matriz matemática del esqueleto (X, Y, Confianza)
        # Nos aseguramos de que keypoints exista y tenga datos (personas detectadas)
        if hasattr(r, 'keypoints') and r.keypoints is not None and r.keypoints.data.shape[1] > 0:
            # .cpu().numpy() convierte de tensor de PyTorch a array de Numpy
            # listo para ser alimentado a un clasificador scikit-learn/TensorFlow
            skeletons_data.append(r.keypoints.data.cpu().numpy())
        else:
            skeletons_data.append(None)
            
    return results, skeletons_data