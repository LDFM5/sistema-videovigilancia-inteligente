"""
alerts.py

Gestión centralizada de alertas para la interfaz web y Telegram.
"""

import time
import threading
from telegram_bot import send_text_async

_COOLDOWN_REGISTRY = {}
_COOLDOWN_LOCK = threading.Lock()

COOLDOWN_SECONDS = {
    "ARMAS": 25.0,
    "FIREARM": 25.0,
    "MELEE_WEAPON": 25.0,
    "COMPORTAMIENTO": 30.0,
    "VIOLENCIA": 30.0,
    "VIOLENCE": 30.0,
    "DEFAULT": 20.0
}


def dispatch_security_alert(cam_name, log_type, message_payload, shared_state):
    """
    Registra un incidente en el estado compartido y envía el aviso a Telegram con rate limiting / cooldown.
    """
    cam_upper = str(cam_name).upper()
    log_type_upper = str(log_type).upper()
    message_text = str(message_payload).strip()
    
    timestamp_str = time.strftime("%d/%m/%Y  ──  %H:%M:%S")

    # =========================================================================
    # 1. ACTUALIZACIÓN DE LA INTERFAZ EN TIEMPO REAL (Siempre se actualiza el Dashboard)
    # =========================================================================
    if shared_state:
        try:
            shared_state.emitir_evento_dashboard('camera_status', {
                "camera": cam_name.lower().strip(), 
                "status": "detecting"
            })

            shared_state.emitir_evento_dashboard('system_log', {
                "type": "warn", 
                "message": f"Detección confirmada en {cam_upper}. Tipo de incidente: {log_type_upper}."
            })

            shared_state.emitir_evento_dashboard('critical_alert', {
                "message": f"{log_type_upper}. Cámara: {cam_upper}."
            })
        except Exception as e:
            print(f"[ERROR] No se pudo actualizar la telemetría de alertas: {e}")

    # =========================================================================
    # 2. VERIFICACIÓN DE COOLDOWN PARA TELEGRAM (Evita spam y HTTP 429)
    # =========================================================================
    ahora = time.monotonic()
    clave_cooldown = f"{cam_upper}_{log_type_upper}"
    tiempo_cooldown = COOLDOWN_SECONDS.get(log_type_upper, COOLDOWN_SECONDS["DEFAULT"])

    with _COOLDOWN_LOCK:
        ultimo_envio = _COOLDOWN_REGISTRY.get(clave_cooldown, 0.0)
        if ahora - ultimo_envio < tiempo_cooldown:
            # En cooldown, omitir mensaje repetido a Telegram
            return
        _COOLDOWN_REGISTRY[clave_cooldown] = ahora

    # =========================================================================
    # 3. CONSTRUCCIÓN Y ENVÍO DEL AVISO POR TELEGRAM
    # =========================================================================
    telegram_report = (
        f"🚨 <b>ALERTA DE SEGURIDAD</b>\n"
        f"───────────────────────\n"
        f"<b>ORIGEN:</b> {cam_upper}\n"
        f"<b>PROTOCOLO:</b> {log_type_upper}\n"
        f"<b>HORA:</b> {timestamp_str}\n"
        f"<b>DETALLE:</b> {message_text}\n"
    )
    
    try:
        send_text_async(telegram_report, shared_state=shared_state)
    except Exception as e:
        if shared_state:
            shared_state.emitir_evento_dashboard('system_log', {
                "type": "error", 
                "message": f"No se pudo enviar la alerta por Telegram: {e}"
            })
