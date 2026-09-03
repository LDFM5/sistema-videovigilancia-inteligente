"""
detection.py

Inferencia por lotes para el modelo de detección de objetos peligrosos.
Responsabilidades:
- Ejecutar la inferencia de varios fotogramas en una sola operación.
- Filtrar las clases de objetos configuradas.
- Extraer regiones de interés y etiquetas por cámara.
"""

import torch

# =========================================================================
# SUBSISTEMA: DETECCIÓN DE OBJETOS PELIGROSOS EN LOTE (WEAPON BATCH INFERENCE)
# =========================================================================

def batch_detect_weapons(model, frames_list, conf=0.5, clases_alerta=None, modo_debug=False):
    """
    Procesa de forma simultánea múltiples flujos matriciales de video para la detección de armamento.
    Mitiga el overhead de transferencia hacia la memoria de la GPU.
    """
    # Mantener un tipo de retorno uniforme cuando no hay fotogramas.
    if not frames_list:
        return [], []

    # Extracción de índices enteros de las clases bajo protocolo de alerta (ej. firearm, melee_weapon)
    cache_key = tuple(sorted(clases_alerta or ()))
    cache = getattr(model, "_daocs_class_filter_cache", {})
    ids_armas = cache.get(cache_key)

    if ids_armas is None:
        ids_armas = []
        if clases_alerta and hasattr(model, 'names'):
            names_items = (
                model.names.items()
                if hasattr(model.names, "items")
                else enumerate(model.names)
            )
            ids_armas = [
                id_clase
                for id_clase, nombre in names_items
                if nombre in clases_alerta
            ]
        cache[cache_key] = ids_armas
        try:
            model._daocs_class_filter_cache = cache
        except Exception:
            pass


    # Convertir a conjunto para realizar búsquedas O(1).
    set_ids_armas = set(ids_armas)

    # Aplicar el filtro de clases cuando el modo de depuración está desactivado.
    filtro_clases = None if modo_debug else (ids_armas if ids_armas else None)

    # Detección automática de precisión: Si el modelo corre en GPU (CUDA), se activa half (FP16)
    use_half = False
    try:
        if hasattr(model, 'device') and model.device.type == 'cuda':
            use_half = True
        elif hasattr(model, 'model') and next(model.model.parameters()).device.type == 'cuda':
            use_half = True
        elif torch.cuda.is_available():
            use_half = True
    except Exception:
        use_half = False

    # Inferencia en lote con contexto de gradiente desactivado para optimizar VRAM
    with torch.inference_mode():
        results = model(frames_list, conf=conf, classes=filtro_clases, half=use_half, verbose=False)
    
    alertas_por_frame = []

    # Evaluación secuencial de la carga de inferencia por cada frame del lote
    for r in results:
        tiene_arma_real = False
        if r.boxes is not None and len(r.boxes) > 0:
            if set_ids_armas:
                # Una sola reducción en GPU y una sola sincronización escalar,
                # en vez de int(cls_id) por cada caja detectada.
                class_ids = r.boxes.cls
                target_ids = torch.as_tensor(
                    tuple(set_ids_armas),
                    device=class_ids.device,
                    dtype=class_ids.dtype,
                )
                tiene_arma_real = bool(
                    (class_ids[:, None] == target_ids[None, :]).any().item()
                )
            else:
                # Si no se definieron clases_alerta, cualquier detección sobre el umbral es válida
                tiene_arma_real = True

        alertas_por_frame.append(tiene_arma_real)

    # Devolver los resultados del lote una vez procesados todos los fotogramas.
    return results, alertas_por_frame
