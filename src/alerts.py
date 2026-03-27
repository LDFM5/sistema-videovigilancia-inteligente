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


# =========================
# ENVIAR ALERTA TELEGRAM
# =========================

import requests
import threading
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID


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