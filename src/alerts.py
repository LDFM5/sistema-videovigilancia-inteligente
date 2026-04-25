"""
alerts.py

Módulo responsable del envío de notificaciones ante eventos detectados.

Funciones principales:
- Enviar mensajes a plataformas externas (ej. Telegram).
- Gestionar el contenido de la alerta (cámara, hora, tipo de evento).
- Permitir futuras integraciones con otros servicios (WhatsApp, email, SMS).

Este módulo desacopla la lógica de notificación del resto del sistema,
facilitando su expansión o modificación.
"""

"""
Supongamos que envías 100 alertas al mes por WhatsApp con tu sistema y tu BSP te cobra:

Meta: ~$0.02 por mensaje entregado

BSP: $20 USD/mes

Entonces:

Meta: 100 × $0.02 = $2

BSP: $20
👉 Total aproximado: ~$22/mes

Este es solo un ejemplo ilustrativo; el coste real depende de:

tu BSP

país receptor

tipo de mensajes
"""
import requests
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
import threading
import os
import cv2


# =========================
# ENVIAR ALERTA TELEGRAM
# =========================

def _send(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    try:
        requests.post(url, data=payload, timeout=5)
    except:
        pass


def send_alert(cam_name, mensaje_evento="Evento sospechoso detectado"):
    """
    Envía una alerta a Telegram con el nombre de la cámara y el tipo de evento.
    """
    message = (
        "🚨 ALERTA DE SEGURIDAD\n\n"
        f"Cámara: {cam_name}\n"
        f"Evento: {mensaje_evento}"
    )

    thread = threading.Thread(target=_send, args=(message,))
    thread.daemon = True
    thread.start()

def _upload_video_thread(filepath, caption):
    temp_path = filepath.replace(".mp4", "_lite.mp4")
    
    try:
        print("⚙️ Generando versión ligera para Telegram...")
        comprimir_para_telegram(filepath, temp_path)
        
        tamaño_mb = os.path.getsize(temp_path) / (1024 * 1024)
        print(f"📊 Video comprimido: {tamaño_mb:.2f} MB")

        # Ahora enviamos el archivo temporal
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendVideo"
        with open(temp_path, 'rb') as video:
            data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': caption}
            files = {'video': video}
            requests.post(url, data=data, files=files)
            
        # Limpiar: borrar la copia ligera, mantener el original pesado en la PC
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    except Exception as e:
        print(f"❌ Error en la cadena de envío: {e}")

def send_video_alert(filepath, caption="🚨 Evidencia en Video"):
    """
    Inicia un hilo independiente para subir el video sin congelar la cámara.
    """
    if os.path.exists(filepath):
        hilo = threading.Thread(target=_upload_video_thread, args=(filepath, caption))
        hilo.start()
    else:
        print(f"⚠️ Error: No se encontró el archivo de video {filepath}")

def comprimir_para_telegram(input_path, output_path):
    """
    Lee el video original y crea una versión de bajo peso:
    - Reduce resolución a 640x360 (o proporcional)
    - Reduce FPS a 10
    """
    cap = cv2.VideoCapture(input_path)
    # Definir el codec liviano
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Reducir resolución a la mitad (aprox 360p para video HD)
    target_width = 640
    target_height = 360
    
    out = cv2.VideoWriter(output_path, fourcc, 10, (target_width, target_height))
    
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Reducir FPS: solo guardamos 1 de cada 2 frames (de 20fps a 10fps)
        if count % 2 == 0:
            frame_small = cv2.resize(frame, (target_width, target_height))
            out.write(frame_small)
        count += 1
        
    cap.release()
    out.release()
    return output_path