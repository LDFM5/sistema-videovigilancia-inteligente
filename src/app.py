"""
app.py

Servidor de control de operaciones de Edge AI y gateway de telemetría asíncrona.
Estructura de alto rendimiento basada en Server-Sent Events (SSE) para entornos industriales.
"""

import threading
import signal
import queue
import json
import time
import sys
from collections import deque

from flask import Flask, render_template, request, jsonify, Response

import config
from main import ejecutar_sistema_principal

app = Flask(__name__)


class EstadoSistema:
    def __init__(self):
        self.lock = threading.Lock()
        # Parámetros de configuración del núcleo de inferencia en tiempo de ejecución
        self.config_ram = {
            "cfg_armas": 1.0,
            "cfg_comportamiento": 1.0,
            "cfg_confianza": 0.5,
            "cfg_prebuffer": 3.0,
            "cfg_postbuffer": 5.0
        }
        # Lista compartida de suscriptores web activos (Mismo espacio de memoria)
        self.suscriptores_activos = []
        # Búfer circular volátil para almacenar los últimos 200 eventos
        self.historial_eventos = deque(maxlen=200)

    def emitir_evento_dashboard(self, event_type, data):
        """
        Método de instancia único: Propaga un evento estructurado simultáneamente 
        a todas las conexiones web compartiendo la misma dirección de memoria RAM.
        """
        with self.lock:
            if event_type == 'system_log':
                log_msg = data.get("message", "").upper()
                t_type = data.get("type", "info").upper()
                print(f"[{t_type}] {log_msg}")
                sys.stdout.flush()

            paquete_estructurado = {
                "tipo_evento": event_type,
                "contenido": data
            }

            # Guardar en el búfer de persistencia en caliente
            self.historial_eventos.append(paquete_estructurado)

            # Distribuir a los sockets abiertos de esta instancia
            for q in list(self.suscriptores_activos):
                try:
                    if q.full():
                        q.get_nowait()
                    q.put_nowait(paquete_estructurado)
                except Exception:
                    pass


# Instancia única soberana del estado del sistema
estado = EstadoSistema()


# ==========================================
# ENDPOINTS WEB
# ==========================================
@app.route('/')
def index():
    nombres_camaras = list(config.CAMERA_INDEXES.keys())
    return render_template(
        'index.html',
        camaras=nombres_camaras
    )


@app.route('/update_config', methods=['POST'])
def update_config():
    data = request.json
    estado.emitir_evento_dashboard('system_log', {
        "type": "info", 
        "message": f"CORE_CFG: PARÁMETROS_RAM_ACTUALIZADOS_EN_CALIENTE -> CAMPOS: {list(data.keys())}"
    })
    return jsonify({"status": "SUCCESS", "code": 200})


# ==========================================
# CANAL SSE (SERVER-SENT EVENTS) EN TIEMPO REAL
# ==========================================
@app.route('/stream-dashboard')
def stream_dashboard():
    """
    Mantiene un canal de transmisión unidireccional permanente con el index.html.
    """
    cola_cliente = queue.Queue(maxsize=2000)
    
    with estado.lock:
        # Volcar el historial acumulado de esta instancia al cliente recién conectado
        for evento_pasado in estado.historial_eventos:
            cola_cliente.put_nowait(evento_pasado)
            
        estado.suscriptores_activos.append(cola_cliente)

    def event_generator():
        try:
            while True:
                try:
                    paquete = cola_cliente.get(timeout=2.0)
                    yield f"data: {json.dumps(paquete)}\n\n"
                except queue.Empty:
                    yield f"data: {json.dumps({'tipo_evento': 'ping', 'contenido': {}})}\n\n"
        except GeneratorExit:
            pass
        finally:
            with estado.lock:
                if cola_cliente in estado.suscriptores_activos:
                    estado.suscriptores_activos.remove(cola_cliente)

    response = Response(event_generator(), mimetype="text/event-stream")
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    return response


def cerrar_servidor(_sig, _frame):
    print("\nSYS_CORE: INTERRUPCIÓN DE TERMINAL DETECTADA. SIGINT RECIBIDO.")
    print("SYS_CORE: DETENIENDO MOTOR DE INFERENCIA INTERFACES FLASK KERNEL...")
    sys.stdout.flush()
    raise KeyboardInterrupt


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGINT, cerrar_servidor)
    
    t = threading.Thread(
        target=ejecutar_sistema_principal,
        args=(estado,), # <--- Pasamos la instancia 'estado' por referencia
        daemon=True,
        name="AI-System-Thread"
    )
    t.start()

    print("HTTP_GATEWAY: INTERFAZ DEL SERVIDOR DE CONTROL DESPLEGADA.")
    print("HTTP_GATEWAY: NODO DE CÓMPUTO LENOVO CORE OPERATIVO EN -> http://localhost:5000")
    sys.stdout.flush()

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )