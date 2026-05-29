"""
detection.py

Módulo optimizado para inferencia en tiempo real y arquitectura Edge AI por lotes.
Responsabilidades:
- Ejecutar inferencia matricial síncrona por lotes (Batching) en hardware acelerador.
- Filtrar descriptores y taxonomías de objetos de interés (Armas/Riesgos).
- Extraer keypoints geométricos bidimensionales mediante tracking persistente.
"""

# =========================================================================
# SUBSISTEMA: DETECCIÓN DE OBJETOS PELIGROSOS EN LOTE (WEAPON BATCH INFERENCE)
# =========================================================================

def batch_detect_weapons(model, frames_list, conf=0.5, clases_alerta=None, modo_debug=False):
    """
    Procesa de forma simultánea múltiples flujos matriciales de video para la detección de armamento.
    Mitiga el overhead de transferencia hacia la memoria de la GPU.
    """
    if not frames_list:
        return []
        
    # Extracción de índices enteros de las clases bajo protocolo de alerta (firearm, melee_weapon)
    ids_armas = []
    if clases_alerta:
        ids_armas = [id_clase for id_clase, nombre in model.names.items() if nombre in clases_alerta]

    # Gestión de modo depuración o aplicación del filtro restrictivo estricto de clases
    filtro_clases = None if modo_debug else (ids_armas if ids_armas else None)

    results = model(frames_list, conf=conf, classes=filtro_clases, verbose=False)
    
    # Evaluación secuencial de la carga de inferencia por cada frame del lote
    alertas_por_frame = []
    for r in results:
        tiene_arma_real = False
        if r.boxes is not None:
            for cls_id in r.boxes.cls:
                if int(cls_id) in ids_armas:
                    tiene_arma_real = True
                    break # Interrupción por hallazgo confirmatorio mínimo
        alertas_por_frame.append(tiene_arma_real)
            
    return results, alertas_por_frame


# =========================================================================
# SUBSISTEMA: ESTIMACIÓN DE POSE E INFRAESTRUCTURA DE SEGUIMIENTO (POSE TRACKING)
# =========================================================================

def batch_detect_pose(model, frames_list, conf=0.5):
    """
    Ejecuta la extracción espacial de esqueleto en lote utilizando persistencia tracking (ByteTrack).
    Retorna los tensores puros de keypoints para su inyección en clasificadores secuenciales.
    """
    if not frames_list:
        return [], []
        
    # Invocación del pipeline de tracking persistente multi-objeto
    results = model.track(frames_list, conf=conf, persist=True, tracker="bytetrack.yaml", verbose=False)
    
    skeletons_data = []
    
    for r in results:
        # Validación de la existencia de la estructura matemática del esqueleto (Matriz X, Y, Conf)
        if hasattr(r, 'keypoints') and r.keypoints is not None and r.keypoints.data.shape[1] > 0:
            # Desacoplamiento del tensor de GPU a la memoria compartida del sistema en formato NumPy array
            skeletons_data.append(r.keypoints.data.cpu().numpy())
        else:
            skeletons_data.append(None)
            
    return results, skeletons_data