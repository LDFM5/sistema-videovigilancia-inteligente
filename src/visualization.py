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


def dibujar_osd_cuadricula(frame, cam_name, alerta_comportamiento=False, score_comp=0.0):
    """
    Renderiza el OSD sobre el fotograma con resolución estandarizada de transmisión (ej. 800x450).
    Garantiza tamaños, fuentes y márgenes uniformes y nítidos en todas las cámaras del mosaico web.
    """
    h_frame, w_frame = frame.shape[:2]
    cam_upper = str(cam_name).upper()

    # 1. Alerta de Comportamiento Hostil (Banner Superior)
    if alerta_comportamiento:
        if score_comp and score_comp > 0:
            texto_alerta = f"ALERTA [{cam_upper}]: CONDUCTA HOSTIL ({int(score_comp * 100)}%)"
        else:
            texto_alerta = f"ALERTA [{cam_upper}]: CONDUCTA HOSTIL DETECTADA"
            
        fuente_a = cv2.FONT_HERSHEY_SIMPLEX
        escala_a = 0.46
        grosor_a = 1
        (w_txt, h_txt), _ = cv2.getTextSize(texto_alerta, fuente_a, escala_a, grosor_a)
        
        box_w = max(420, w_txt + 55)
        cv2.rectangle(frame, (12, 10), (min(w_frame - 12, 12 + box_w), 42), (15, 15, 20), -1)
        cv2.rectangle(frame, (12, 10), (min(w_frame - 12, 12 + box_w), 42), (50, 50, 240), 2)
        cv2.circle(frame, (28, 26), 6, (0, 0, 255), -1)
        cv2.putText(
            frame, texto_alerta, (44, 31), 
            fuente_a, escala_a, (255, 255, 255), grosor_a, cv2.LINE_AA
        )

    # 2. Metadatos de Canal: Cámara y Fecha/Hora (Banner Inferior Izquierdo)
    cadena_tiempo = time.strftime("%d/%m/%Y  ──  %H:%M:%S")
    texto_meta = f"CAM: {cam_upper}  |  {cadena_tiempo}"
    fuente_m = cv2.FONT_HERSHEY_SIMPLEX
    escala_m = 0.44
    grosor_m = 1
    (wm_txt, hm_txt), _ = cv2.getTextSize(texto_meta, fuente_m, escala_m, grosor_m)
    
    x1_m = 10
    y2_m = h_frame - 10
    y1_m = max(0, y2_m - hm_txt - 16)
    x2_m = min(w_frame - 10, x1_m + wm_txt + 24)
    
    cv2.rectangle(frame, (x1_m, y1_m), (x2_m, y2_m), (10, 11, 13), -1)
    cv2.rectangle(frame, (x1_m, y1_m), (x2_m, y2_m), (35, 40, 48), 1)
    cv2.putText(
        frame, texto_meta, (x1_m + 12, y2_m - 8), 
        fuente_m, escala_m, (227, 230, 235), grosor_m, cv2.LINE_AA
    )

    return frame


def dibujar_osd_proporcional(frame, cam_name, alerta_comportamiento=False, score_comp=0.0):
    """
    Renderiza el OSD proporcional para grabaciones de evidencia en resoluciones nativas arbitrarias.
    Ajusta dinámicamente la escala y asegura que los elementos nunca desborden el fotograma.
    """
    h_frame, w_frame = frame.shape[:2]
    cam_upper = str(cam_name).upper()
    
    # Factor de escala referenciado a 800px de ancho base
    factor = max(0.35, min(w_frame / 800.0, 2.0))
    
    # 1. Alerta de Comportamiento
    if alerta_comportamiento:
        if score_comp and score_comp > 0:
            texto_alerta = f"ALERTA [{cam_upper}]: CONDUCTA HOSTIL ({int(score_comp * 100)}%)"
        else:
            texto_alerta = f"ALERTA [{cam_upper}]: CONDUCTA HOSTIL DETECTADA"
            
        fuente_a = cv2.FONT_HERSHEY_SIMPLEX
        escala_a = max(0.32, 0.46 * factor)
        grosor_a = max(1, int(1 * factor))
        (w_txt, h_txt), _ = cv2.getTextSize(texto_alerta, fuente_a, escala_a, grosor_a)
        
        pad_x = int(14 * factor)
        pad_y = int(8 * factor)
        r_circulo = max(3, int(6 * factor))
        
        y1_a = int(8 * factor)
        y2_a = min(h_frame - 5, y1_a + h_txt + pad_y * 2)
        x1_a = int(10 * factor)
        x2_a = min(w_frame - 10, x1_a + w_txt + r_circulo * 2 + pad_x * 2)
        
        cv2.rectangle(frame, (x1_a, y1_a), (x2_a, y2_a), (15, 15, 20), -1)
        cv2.rectangle(frame, (x1_a, y1_a), (x2_a, y2_a), (50, 50, 240), max(1, int(2 * factor)))
        
        centro_c = (x1_a + pad_x + r_circulo, (y1_a + y2_a) // 2)
        cv2.circle(frame, centro_c, r_circulo, (0, 0, 255), -1)
        
        cv2.putText(
            frame, texto_alerta, (centro_c[0] + r_circulo + int(8 * factor), y2_a - pad_y), 
            fuente_a, escala_a, (255, 255, 255), grosor_a, cv2.LINE_AA
        )

    # 2. Metadatos de Canal
    cadena_tiempo = time.strftime("%d/%m/%Y  ──  %H:%M:%S")
    texto_meta = f"CAM: {cam_upper}  |  {cadena_tiempo}"
    fuente_m = cv2.FONT_HERSHEY_SIMPLEX
    escala_m = max(0.30, 0.44 * factor)
    grosor_m = max(1, int(1 * factor))
    (wm_txt, hm_txt), _ = cv2.getTextSize(texto_meta, fuente_m, escala_m, grosor_m)
    
    pad_xm = int(12 * factor)
    pad_ym = int(6 * factor)
    
    x1_m = int(10 * factor)
    y2_m = max(hm_txt + 10, h_frame - int(10 * factor))
    y1_m = max(0, y2_m - hm_txt - pad_ym * 2)
    x2_m = min(w_frame - 5, x1_m + wm_txt + pad_xm * 2)
    
    cv2.rectangle(frame, (x1_m, y1_m), (x2_m, y2_m), (10, 11, 13), -1)
    cv2.rectangle(frame, (x1_m, y1_m), (x2_m, y2_m), (35, 40, 48), 1)
    cv2.putText(
        frame, texto_meta, (x1_m + pad_xm, y2_m - pad_ym), 
        fuente_m, escala_m, (227, 230, 235), grosor_m, cv2.LINE_AA
    )

    return frame
