"""
telegram_bot.py

Integración con la API de Telegram para enviar alertas y videos de evidencia.
Valida las credenciales y los archivos antes de realizar las solicitudes HTTP.
Incluye una cola asíncrona dedicada con rate-limiting para evitar errores HTTP 429.
"""

import os
import time
import queue
import requests
import threading
import config

_TELEGRAM_QUEUE = queue.Queue(maxsize=30)
_WORKER_STARTED = False
_WORKER_LOCK = threading.Lock()
_MIN_API_INTERVAL_SECONDS = 1.5


def _telegram_worker():
    last_call_time = 0.0
    while True:
        try:
            item = _TELEGRAM_QUEUE.get()
            if item is None:
                break
            
            task_type, data, shared_state = item
            
            # Respetar el intervalo mínimo entre llamadas para cumplir con el rate-limit de Telegram
            now = time.monotonic()
            elapsed = now - last_call_time
            if elapsed < _MIN_API_INTERVAL_SECONDS:
                time.sleep(_MIN_API_INTERVAL_SECONDS - elapsed)

            token = getattr(config, 'TELEGRAM_TOKEN', None)
            chat_id = getattr(config, 'TELEGRAM_CHAT_ID', None)

            if not token or not chat_id:
                _TELEGRAM_QUEUE.task_done()
                continue

            if task_type == "text":
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                payload = {
                    "chat_id": chat_id,
                    "text": data["message"],
                    "parse_mode": "HTML"
                }
                try:
                    resp = requests.post(url, data=payload, timeout=8.0)
                    last_call_time = time.monotonic()
                    if resp.status_code != 200 and shared_state:
                        shared_state.emitir_evento_dashboard('system_log', {
                            "type": "warn",
                            "message": f"Telegram status {resp.status_code}: {resp.text[:80]}"
                        })
                except Exception as e:
                    if shared_state:
                        shared_state.emitir_evento_dashboard('system_log', {
                            "type": "error",
                            "message": f"Error de red enviando texto a Telegram: {e}"
                        })

            elif task_type == "video":
                filepath = data["filepath"]
                caption = data.get("caption", "")
                remove_after = data.get("remove_after", False)
                if os.path.exists(filepath):
                    url = f"https://api.telegram.org/bot{token}/sendVideo"
                    try:
                        with open(filepath, 'rb') as video:
                            post_data = {
                                'chat_id': chat_id,
                                'caption': caption,
                                'parse_mode': 'HTML'
                            }
                            files = {'video': video}
                            resp = requests.post(url, data=post_data, files=files, timeout=90.0)
                            last_call_time = time.monotonic()
                            if resp.status_code == 200 and shared_state:
                                shared_state.emitir_evento_dashboard('system_log', {
                                    "type": "success",
                                    "message": "Video de evidencia entregado a Telegram."
                                })
                            elif shared_state:
                                shared_state.emitir_evento_dashboard('system_log', {
                                    "type": "warn",
                                    "message": f"Telegram rechazó video: código {resp.status_code}."
                                })
                    except Exception as e:
                        if shared_state:
                            shared_state.emitir_evento_dashboard('system_log', {
                                "type": "error",
                                "message": f"Error subiendo video a Telegram: {e}"
                            })
                    finally:
                        if remove_after and os.path.exists(filepath):
                            try:
                                os.remove(filepath)
                            except Exception:
                                pass

        except Exception as general_err:
            print(f"[ERROR Telegram Worker] {general_err}")
        finally:
            _TELEGRAM_QUEUE.task_done()


def _ensure_worker_started():
    global _WORKER_STARTED
    with _WORKER_LOCK:
        if not _WORKER_STARTED:
            t = threading.Thread(target=_telegram_worker, daemon=True, name="TelegramRateLimitedWorker")
            t.start()
            _WORKER_STARTED = True


def send_text_async(message, shared_state=None):
    """
    Encola un mensaje de texto para envío asíncrono con rate limiting.
    """
    token = getattr(config, 'TELEGRAM_TOKEN', None)
    chat_id = getattr(config, 'TELEGRAM_CHAT_ID', None)

    if not token or not chat_id:
        return

    _ensure_worker_started()
    try:
        if _TELEGRAM_QUEUE.full():
            _TELEGRAM_QUEUE.get_nowait()
        _TELEGRAM_QUEUE.put_nowait(("text", {"message": message}, shared_state))
    except Exception:
        pass


def send_video_sync(filepath, caption, remove_after_send=False, shared_state=None):
    """
    Encola un video de evidencia para envío asíncrono respetando la tasa de Telegram.
    Si remove_after_send es True, elimina el archivo temporal de forma segura tras la subida.
    """
    token = getattr(config, 'TELEGRAM_TOKEN', None)
    chat_id = getattr(config, 'TELEGRAM_CHAT_ID', None)

    if not token or not chat_id:
        return

    if not os.path.exists(filepath):
        return

    _ensure_worker_started()
    try:
        _TELEGRAM_QUEUE.put(
            (
                "video",
                {
                    "filepath": filepath,
                    "caption": caption,
                    "remove_after": bool(remove_after_send),
                },
                shared_state,
            ),
            timeout=2.0,
        )
    except Exception:
        pass
