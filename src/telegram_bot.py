"""
telegram_bot.py

Módulo de infraestructura de red exclusivo para el enlace con la API de Telegram.
Encapsula transacciones HTTP síncronas y asíncronas para el despacho de telemetría y evidencia
con protección estricta contra credenciales nulas o errores de archivo.
"""

import os
import requests
import threading
import config


def send_text_async(message, shared_state=None):
    """
    Despacha una cadena de texto formateada mediante un subproceso asíncrono aislado.
    Evita bloqueos en el hilo de inferencia principal ante latencias en la red WAN.
    """
    token = getattr(config, 'TELEGRAM_TOKEN', None)
    chat_id = getattr(config, 'TELEGRAM_CHAT_ID', None)

    # 🎯 SALVAGUARDA: Cancelar envío si no hay credenciales configuradas
    if not token or not chat_id:
        if shared_state:
            shared_state.emitir_evento_dashboard('system_log', {
                "type": "info",
                "message": "BOT_TELEGRAM: ENVÍO DE TEXTO OMITIDO (CREDANCIALES NO CONFIGURADAS EN config_local.py)"
            })
        return

    def _send():
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
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
    token = getattr(config, 'TELEGRAM_TOKEN', None)
    chat_id = getattr(config, 'TELEGRAM_CHAT_ID', None)

    # 🎯 SALVAGUARDA: Cancelar envío si no hay credenciales configuradas
    if not token or not chat_id:
        if shared_state:
            shared_state.emitir_evento_dashboard('system_log', {
                "type": "info",
                "message": "BOT_TELEGRAM: ENVÍO DE VIDEO OMITIDO (CREDANCIALES NO CONFIGURADAS)"
            })
        return

    # 🎯 SALVAGUARDA: Verificar si el archivo comprimido existe antes de intentar leerlo
    if not os.path.exists(filepath):
        if shared_state:
            shared_state.emitir_evento_dashboard('system_log', {
                "type": "error",
                "message": f"BOT_TELEGRAM_FALLO: ARCHIVO MULTIMEDIA NO ENCONTRADO EN DISCO -> {filepath}"
            })
        return

    url = f"https://api.telegram.org/bot{token}/sendVideo"
    try:
        with open(filepath, 'rb') as video:
            data = {
                'chat_id': chat_id, 
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