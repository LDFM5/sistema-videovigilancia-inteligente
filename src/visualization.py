"""
visualization.py

Módulo responsable de la interfaz gráfica superpuesta (Overlay) en el video.
Calcula el rendimiento (RAM/CPU) y lo dibuja sobre el frame.
"""
import cv2
import psutil
import os

def draw_performance_overlay(frame, fps_real):
    """
    Toma un frame crudo, calcula el consumo de hardware actual,
    dibuja el panel de rendimiento y retorna el frame modificado.
    """
    # 1. Obtener datos de RAM y CPU
    proceso = psutil.Process(os.getpid())
    ram_mb = proceso.memory_info().rss / (1024 * 1024)
    ram_app_pct = proceso.memory_percent()
    ram_pc_pct = psutil.virtual_memory().percent
    cpu_percent = psutil.cpu_percent()

    # 2. Crear el texto a mostrar
    texto_rendimiento = f"FPS: {int(fps_real)} | RAM App: {ram_mb:.1f}MB ({ram_app_pct:.1f}%) | RAM PC: {ram_pc_pct}% | CPU: {cpu_percent}%"

    # 3. Configuración visual
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    escala_fuente = 0.5         
    grosor_fuente = 1           
    color_texto = (200, 200, 200) 
    
    # 4. Calcular geometría del rectángulo
    (ancho_texto, alto_texto), baseline = cv2.getTextSize(texto_rendimiento, fuente, escala_fuente, grosor_fuente)
    x, y = 5, 5 + alto_texto
    rect_x1, rect_y1 = x - 5, y - alto_texto - 5
    rect_x2, rect_y2 = x + ancho_texto + 5, y + baseline + 5

    # 5. Dibujar el fondo semitransparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1) 
    opacidad = 0.5 
    cv2.addWeighted(overlay, opacidad, frame, 1 - opacidad, 0, frame)

    # 6. Dibujar el texto
    cv2.putText(frame, texto_rendimiento, (x, y), fuente, escala_fuente, color_texto, grosor_fuente, cv2.LINE_AA)
    
    return frame