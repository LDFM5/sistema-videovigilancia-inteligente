"""
detection.py

Módulo optimizado para inferencia en tiempo real y arquitectura Edge AI por lotes.
Responsabilidades:
- Ejecutar inferencia matricial síncrona por lotes (Batching) en hardware acelerador.
- Filtrar descriptores y taxonomías de objetos de interés (Armas/Riesgos).
- Extraer regiones de interés y etiquetas confirmatorias por canal.
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
    # 🚨 SOLUCIÓN BUG 2: Garantizar retorno de tupla homogénea ante listas vacías
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

    # Convertir a conjunto hash para búsqueda ultrarrápida O(1)
    set_ids_armas = set(ids_armas)

    # Gestión de modo depuración o aplicación del filtro restrictivo estricto de clases en el modelo
    filtro_clases = None if modo_debug else (ids_armas if ids_armas else None)

    # Inferencia en lote con contexto de gradiente desactivado para optimizar VRAM
    with torch.inference_mode():
        results = model(frames_list, conf=conf, classes=filtro_clases, verbose=False)
    
    alertas_por_frame = []

    # Evaluación secuencial de la carga de inferencia por cada frame del lote
    for r in results:
        tiene_arma_real = False
        if r.boxes is not None and len(r.boxes) > 0:
            for cls_id in r.boxes.cls:
                # Si hay filtro de clases activo, validar contra el conjunto de interés
                if set_ids_armas:
                    if int(cls_id) in set_ids_armas:
                        tiene_arma_real = True
                        break  # Interrupción por hallazgo confirmatorio mínimo
                else:
                    # Si no se definieron clases_alerta, cualquier detección sobre el umbral es válida
                    tiene_arma_real = True
                    break

        alertas_por_frame.append(tiene_arma_real)

    # 🚨 SOLUCIÓN BUG 1: Retorno unificado fuera del bucle FOR
    return results, alertas_por_frame
