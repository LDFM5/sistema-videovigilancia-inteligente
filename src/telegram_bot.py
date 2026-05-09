"""
telegram_bot.py

Módulo exclusivo para la comunicación con la API de Telegram.
Maneja peticiones HTTP asíncronas y síncronas para enviar textos y archivos.
"""
import requests
import threading
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

def send_text_async(message):
    """Envía un mensaje de texto a Telegram usando un hilo en segundo plano."""
    def _send():
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message
        }
        try:
            requests.post(url, data=payload, timeout=5)
        except Exception as e:
            print(f"❌ Error enviando mensaje a Telegram: {e}")
            
    thread = threading.Thread(target=_send)
    thread.daemon = True
    thread.start()

def send_video_sync(filepath, caption):
    """
    Sube un video a Telegram de forma síncrona.
    NOTA: Debe ser llamado desde un hilo externo para no congelar el programa.
    """
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendVideo"
    try:
        with open(filepath, 'rb') as video:
            data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': caption}
            files = {'video': video}
            # Un timeout más largo (60s) porque subir videos tarda más
            requests.post(url, data=data, files=files, timeout=60)
    except Exception as e:
        print(f"❌ Error subiendo video a Telegram: {e}")