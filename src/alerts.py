"""
alerts.py

Subsistema centralizado de enrutamiento y despacho de telemetría de seguridad.
Estructura reportes ejecutivos automatizados para interfaces web y pasarelas externas.
"""

import time
from telegram_bot import send_text_async


def dispatch_security_alert(cam_name, log_type, message_payload, shared_state):
    """
    Controlador de incidentes. Utiliza la referencia de memoria 'shared_state'
    para inyectar datos directamente en el canal HTTP activo del servidor Flask
    y despachar reportes ejecutivos a Telegram con protección contra fallos.
    """
    # 🎯 Casteo seguro contra objetos no-string
    cam_upper = str(cam_name).upper()
    log_type_upper = str(log_type).upper()
    message_upper = str(message_payload).upper()
    
    timestamp_str = time.strftime("%d/%m/%Y  ──  %H:%M:%S")

    # =========================================================================
    # 1. EMISIÓN DE TELEMETRÍA AL DASHBOARD EN TIEMPO REAL
    # =========================================================================
    if shared_state:
        try:
            # Forzar recuadro de la cámara a estado crítico en la GUI web
            shared_state.emitir_evento_dashboard('camera_status', {
                "camera": cam_name.lower().strip(), 
                "status": "detecting"
            })

            # Imprimir registro en la consola de telemetría web
            shared_state.emitir_evento_dashboard('system_log', {
                "type": "warn", 
                "message": f"MOTOR_INFERENCIA: COINCIDENCIA_POSITIVA EN FUENTE '{cam_upper}' -> PROTOCOLO: {log_type_upper}"
            })

            # Inyectar en el feed de alertas críticas de la interfaz
            shared_state.emitir_evento_dashboard('critical_alert', {
                "message": f"{log_type_upper} - ORIGEN: {cam_upper}"
            })
        except Exception as e:
            print(f"ERROR_PASARELA_ALERTAS: FALLO EN TELEMETRÍA COMPARTIDA -> {str(e).upper()}")

    # =========================================================================
    # 2. CONSTRUCCIÓN Y DESPACHO DEL INFORME CORPORATIVO PARA TELEGRAM
    # =========================================================================
    telegram_report = (
        f"🚨 ALERTA DE INCIDENTE\n"
        f"───────────────────────\n"
        f"SISTEMA: PRINCIPAL\n"
        f"ORIGEN: {cam_upper}\n"
        f"PROTOCOLO: {log_type_upper}\n"
        f"HORA: {timestamp_str}\n"
        f"DETALLE: {message_upper}\n"
    )
    
    # 🎯 Aislar la ejecución del bot para evitar interrupciones en el motor principal
    try:
        send_text_async(telegram_report, shared_state=shared_state)
    except Exception as e:
        if shared_state:
            shared_state.emitir_evento_dashboard('system_log', {
                "type": "error", 
                "message": f"TELEGRAM_ERROR: FALLO AL DESPACHAR MENSAJE DE TEXTO -> {str(e).upper()}"
            })