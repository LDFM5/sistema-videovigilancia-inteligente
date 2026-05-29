"""
alerts.py

Subsistema centralizado de enrutamiento y despacho de telemetría de seguridad.
Estructura reportes ejecutivos automatizados para interfaces web y pasarelas externas.
"""

from telegram_bot import send_text_async


def dispatch_security_alert(cam_name, log_type, message_payload, shared_state):
    """
    Controlador de incidentes. Utiliza la referencia de memoria 'shared_state'
    para inyectar datos directamente en el canal HTTP activo del servidor Flask.
    """
    cam_upper = cam_name.upper()
    log_type_upper = log_type.upper()
    message_upper = message_payload.upper()

    # =========================================================================
    # 1. EMISIÓN DE TELEMETRÍA USANDO EL PUNTERO COMPARTIDO REAL
    # =========================================================================
    if shared_state:
        try:
            # Forzar el recuadro a parpadeo rojo
            shared_state.emitir_evento_dashboard('camera_status', {
                "camera": cam_name.lower(), 
                "status": "detecting"
            })

            # Imprimir el log de consola web en el mismo instante
            shared_state.emitir_evento_dashboard('system_log', {
                "type": "warn", 
                "message": f"MOTOR_INFERENCIA: COINCIDENCIA_POSITIVA EN FUENTE '{cam_upper}' -> PROTOCOLO: {log_type_upper}"
            })

            # Añadir al feed de alertas críticas
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
        f"DETALLE: {message_upper}\n"
    )
    
    send_text_async(telegram_report, shared_state=shared_state)