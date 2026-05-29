"""
telegram_bot.py

Módulo de infraestructura de red exclusivo para el enlace con la API de Telegram.
Encapsula transacciones HTTP síncronas y asíncronas para el despacho de telemetría y evidencia.
"""

import requests
import threading
import sys
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID


def send_text_async(message, shared_state=None):
    """
    Despacha una cadena de texto formateada mediante un subproceso asíncrono aislado.
    Evita bloqueos en el hilo de inferencia principal ante latencias en la red WAN.
    """
    def _send():
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message
        }
        try:
            response = requests.post(url, data=payload, timeout=5.0)
            if response.status_code != 200:
                if shared_state:
                    shared_state.emitir_evento_dashboard('system_log', {
                        "type": "error", 
                        "message": f"BOT_TELEGRAM_FALLO: RECHAZADO POR LA API -> CÓDIGO: {response.status_code}"
                    })
        except Exception as e:
            if shared_state:
                shared_state.emitir_evento_dashboard('system_log', {
                    "type": "error", 
                    "message": f"BOT_TELEGRAM_FALLO: ERROR DE RED AL ENVIAR TEXTO -> REF: {str(e).upper()}"
                })
            
    thread = threading.Thread(
        target=_send,
        daemon=True,
        name="Telegram-Async-Sender"
    )
    thread.start()


def send_video_sync(filepath, caption, shared_state=None):
    """
    Sube un binario de video a los servidores de Telegram de forma síncrona.
    CRÍTICO: Este método debe ser invocado exclusivamente dentro de un hilo secundario 
    (como el Worker asignado en recorder.py) para no congelar las interfaces críticas.
    """
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendVideo"
    try:
        with open(filepath, 'rb') as video:
            data = {
                'chat_id': TELEGRAM_CHAT_ID, 
                'caption': caption,
                'parse_mode': 'HTML'
            }
            files = {'video': video}
            
            if shared_state:
                shared_state.emitir_evento_dashboard('system_log', {
                    "type": "info", 
                    "message": "BOT_TELEGRAM: SUBIENDO ARCHIVO DE VIDEO COMPRIMIDO..."
                })

            response = requests.post(url, data=data, files=files, timeout=60.0)
            
            if response.status_code == 200:
                if shared_state:
                    shared_state.emitir_evento_dashboard('system_log', {
                        "type": "success", 
                        "message": "BOT_TELEGRAM: TRANSFERENCIA MULTIMEDIA COMPLETADA CON ÉXITO."
                    })
            else:
                if shared_state:
                    shared_state.emitir_evento_dashboard('system_log', {
                        "type": "error", 
                        "message": f"BOT_TELEGRAM_FALLO: VIDEO RECHAZADO POR LA API -> CÓDIGO: {response.status_code}"
                    })
                
    except Exception as e:
        if shared_state:
            shared_state.emitir_evento_dashboard('system_log', {
                "type": "error", 
                "message": f"BOT_TELEGRAM_FALLO: ERROR EN EL PIPELINE BINARIO -> REF: {str(e).upper()}"
            })