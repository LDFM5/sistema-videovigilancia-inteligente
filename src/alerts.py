"""
alerts.py

Módulo responsable de estructurar las notificaciones ante eventos detectados.
"""
from telegram_bot import send_text_async

def send_alert(cam_name, mensaje_evento="Evento sospechoso detectado"):
    """
    Construye y envía una alerta a Telegram con el nombre de la cámara y el evento.
    """
    message = (
        "🚨 ALERTA DE SEGURIDAD\n\n"
        f"Cámara: {cam_name}\n"
        f"Evento: {mensaje_evento}"
    )
    
    send_text_async(message)