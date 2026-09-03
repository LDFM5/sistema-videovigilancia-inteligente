"""
visualization.py

Muestra métricas de hardware (RAM, CPU y FPS) sobre el video.
Utiliza una caché y limita el dibujo al área ocupada por el texto.
"""

import cv2
import psutil
import os
import time

# Variables globales de caché para no saturar al sistema operativo con llamadas a psutil
_ULTIMA_ACTUALIZACION_METRICAS = 0.0
_TEXTO_RENDIMIENTO_CACHE = ""


def draw_performance_overlay(frame, fps_real):
    """
    Recibe un fotograma crudo, calcula el consumo de hardware (actualizado cada 0.5s)
    y muestra las métricas en el área correspondiente del fotograma.
    """
    global _ULTIMA_ACTUALIZACION_METRICAS, _TEXTO_RENDIMIENTO_CACHE

    ahora = time.time()

    # Actualizar las estadísticas de CPU y memoria cada 0.5 segundos.
    if ahora - _ULTIMA_ACTUALIZACION_METRICAS >= 0.5 or not _TEXTO_RENDIMIENTO_CACHE:
        try:
            proceso = psutil.Process(os.getpid())
            ram_mb = proceso.memory_info().rss / (1024 * 1024)
            ram_app_pct = proceso.memory_percent()
            ram_pc_pct = psutil.virtual_memory().percent
            cpu_percent = psutil.cpu_percent()

            _TEXTO_RENDIMIENTO_CACHE = (
                f"FPS: {int(fps_real)} | "
                f"RAM App: {ram_mb:.1f}MB ({ram_app_pct:.1f}%) | "
                f"RAM PC: {ram_pc_pct}% | "
                f"CPU: {cpu_percent}%"
            )
        except Exception:
            _TEXTO_RENDIMIENTO_CACHE = f"FPS: {int(fps_real)} | METRICAS_NO_DISPONIBLES"
            
        _ULTIMA_ACTUALIZACION_METRICAS = ahora
    else:
        # En fotogramas intermedios, solo actualizar el valor de FPS
        partes = _TEXTO_RENDIMIENTO_CACHE.split(" | ", 1)
        if len(partes) > 1:
            _TEXTO_RENDIMIENTO_CACHE = f"FPS: {int(fps_real)} | {partes[1]}"

    # Configuración de tipografía
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    escala_fuente = 0.45
    grosor_fuente = 1
    color_texto = (220, 220, 220)  # BGR: Gris claro

    # Geometría del texto
    (ancho_texto, alto_texto), baseline = cv2.getTextSize(
        _TEXTO_RENDIMIENTO_CACHE, fuente, escala_fuente, grosor_fuente
    )

    h_frame, w_frame = frame.shape[:2]

    # Delimitar coordenadas con protección contra bordes (Out of Bounds)
    x, y = 8, 8 + alto_texto
    rect_x1, rect_y1 = max(0, x - 5), max(0, y - alto_texto - 5)
    rect_x2, rect_y2 = min(w_frame, x + ancho_texto + 5), min(h_frame, y + baseline + 5)

    # Aplicar transparencia únicamente al área ocupada por el texto.
    if rect_x2 > rect_x1 and rect_y2 > rect_y1:
        roi = frame[rect_y1:rect_y2, rect_x1:rect_x2]
        frame[rect_y1:rect_y2, rect_x1:rect_x2] = cv2.convertScaleAbs(roi, alpha=0.50)

    # Dibujar el texto de las métricas.
    cv2.putText(
        frame, 
        _TEXTO_RENDIMIENTO_CACHE, 
        (x, y), 
        fuente, 
        escala_fuente, 
        color_texto, 
        grosor_fuente, 
        cv2.LINE_AA
    )

    return frame
