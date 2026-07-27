"""
app.py

Servidor de control de operaciones de Edge AI y gateway de telemetría asíncrona.
Estructura de alto rendimiento basada en Server-Sent Events (SSE) para entornos industriales.
"""

import threading
import signal
import queue
import json
import sys
from collections import deque

from flask import Flask, render_template, request, jsonify, Response

import config
from main import ejecutar_sistema_principal

app = Flask(__name__)


class EstadoSistema:
    def __init__(self):
        self.lock = threading.Lock()
        
        # PARÁMETROS SINCRONIZADOS: Inicialización desde config.py con umbrales desacoplados
        self.config_ram = {
            "cfg_armas": config.ACTIVAR_MODELO_ARMAS,
            "cfg_comportamiento": config.ACTIVAR_MODELO_COMPORTAMIENTO,
            "cfg_confianza_armas": config.CONF_WEAPON,
            "cfg_confianza_comportamiento": config.CONF_BEHAVIOR,
            "cfg_prebuffer": config.PRE_BUFFER_SECONDS,
            "cfg_postbuffer": config.POST_BUFFER_SECONDS,
            "cfg_debug": config.MODO_DEBUG
        }
        
        # Lista compartida de suscriptores web activos
        self.suscriptores_activos = []
        # Búfer circular volátil para almacenar los últimos 200 eventos
        self.historial_eventos = deque(maxlen=200)

    def emitir_evento_dashboard(self, event_type, data):
        """
        Método de instancia único: Propaga un evento estructurado simultáneamente 
        a todas las conexiones web activas.
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

            # Tomar una foto instantánea de la lista de suscriptores para minimizar tiempo de lock
            suscriptores = list(self.suscriptores_activos)

        # Despacho fuera del lock principal
        for q in suscriptores:
            try:
                if q.full():
                    q.get_nowait()
                q.put_nowait(paquete_estructurado)
            except Exception:
                pass


# Instancia única soberana del estado del sistema
estado = EstadoSistema()


# ==========================================
# ENDPOINTS WEB Y CONTROL DE PARÁMETROS
# ==========================================
@app.route('/')
def index():
    nombres_camaras = list(config.CAMERA_INDEXES.keys())
    return render_template(
        'index.html',
        camaras=nombres_camaras,
        config_ram=estado.config_ram
    )


@app.route('/update_config', methods=['POST'])
def update_config():
    """
    Actualiza los parámetros del sistema en caliente dentro de la memoria RAM
    aplicando casteo estricto por tipo de dato.
    """
    data = request.json
    with estado.lock:
        for key in data:
            if key in estado.config_ram:
                # 1. Banderas de control (Booleanas)
                if key in ["cfg_armas", "cfg_comportamiento", "cfg_debug"]:
                    estado.config_ram[key] = bool(data[key])
                # 2. Duración de búferes de grabación (Enteros)
                elif key in ["cfg_prebuffer", "cfg_postbuffer"]:
                    estado.config_ram[key] = int(float(data[key]))
                # 3. Umbrales de confianza para modelos (Flotantes)
                else:
                    estado.config_ram[key] = float(data[key])
                    
    estado.emitir_evento_dashboard('system_log', {
        "type": "info", 
        "message": f"CONTROL_PANEL: VARIABLE_RAM_MODIFICADA -> DATOS: {list(data.keys())}"
    })
    return jsonify({"status": "SUCCESS", "code": 200})


@app.route('/save_config', methods=['POST'])
def save_config():
    """
    Guarda los parámetros actuales de la memoria RAM de forma permanente en el archivo físico JSON.
    """
    try:
        with estado.lock:
            datos_a_guardar = dict(estado.config_ram)
            
        # Escribir en el disco en una operación atómica
        config.guardar_configuracion_disco(datos_a_guardar)
        
        estado.emitir_evento_dashboard('system_log', {
            "type": "success", 
            "message": "CONTROL_PANEL: CONFIGURACIÓN GUARDADA DE FORMA PERMANENTE EN EL DISCO."
        })
        return jsonify({"status": "SUCCESS", "code": 200})
        
    except Exception as e:
        try:
            estado.emitir_evento_dashboard('system_log', {
                "type": "error", 
                "message": f"CONTROL_PANEL_ERROR: NO SE PUDO REESCRIBIR EL ARCHIVO JSON -> {str(e).upper()}"
            })
        except Exception:
            pass
        return jsonify({"status": "ERROR", "code": 500})


@app.route('/restore_defaults', methods=['POST'])
def restore_defaults():
    """
    Restablece el archivo JSON y la memoria RAM a los valores de fábrica.
    """
    try:
        valores_defecto = config.restaurar_valores_fabrica()
        
        with estado.lock:
            estado.config_ram["cfg_armas"] = valores_defecto["cfg_armas"]
            estado.config_ram["cfg_comportamiento"] = valores_defecto["cfg_comportamiento"]
            estado.config_ram["cfg_confianza_armas"] = valores_defecto["cfg_confianza_armas"]
            estado.config_ram["cfg_confianza_comportamiento"] = valores_defecto["cfg_confianza_comportamiento"]
            estado.config_ram["cfg_prebuffer"] = valores_defecto["cfg_prebuffer"]
            estado.config_ram["cfg_postbuffer"] = valores_defecto["cfg_postbuffer"]
            estado.config_ram["cfg_debug"] = valores_defecto["cfg_debug"]
                
        estado.emitir_evento_dashboard('system_log', {
            "type": "warn", 
            "message": "CONTROL_PANEL: VALORES DE FÁBRICA RESTABLECIDOS. COMPRUEBE LA INTERFAZ."
        })
        return jsonify({"status": "SUCCESS", "data": valores_defecto, "code": 200})
        
    except Exception as e:
        try:
            estado.emitir_evento_dashboard('system_log', {
                "type": "error", 
                "message": f"CONTROL_PANEL_ERROR: FALLO AL RESTABLECER VALORES -> {str(e).upper()}"
            })
        except Exception:
            pass
        return jsonify({"status": "ERROR", "code": 500})

        
@app.route('/get_initial_state', methods=['GET'])
def get_initial_state():
    """
    Devuelve el mapa de estados actual de las cámaras sin mantener bloqueado el cerrojo global.
    """
    from main import obtener_mapa_estados_actual
    
    try:
        mapa_estados = obtener_mapa_estados_actual()
        return jsonify({"status": "SUCCESS", "data": mapa_estados, "code": 200})
    except Exception as e:
        return jsonify({"status": "ERROR", "message": str(e), "code": 500})


# ==========================================
# CANAL SSE (SERVER-SENT EVENTS) EN TIEMPO REAL
# ==========================================
@app.route('/stream-dashboard')
def stream_dashboard():
    """
    Mantiene la conexión permanente de eventos para el dashboard.
    """
    cola_cliente = queue.Queue(maxsize=2000)
    
    with estado.lock:
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
    print("SYS_CORE: DETENIENDO MOTOR DE INFERENCIA E INTERFACES FLASK KERNEL...")
    sys.stdout.flush()
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGINT, cerrar_servidor)
    
    t = threading.Thread(
        target=ejecutar_sistema_principal,
        args=(estado,),
        daemon=True,
        name="AI-System-Thread"
    )
    t.start()

    print("HTTP_GATEWAY: INTERFAZ DEL SERVIDOR DE CONTROL DESPLEGADA.")
    print("HTTP_GATEWAY: NODO DE CÓMPUTO OPERATIVO EN -> http://localhost:5000")
    sys.stdout.flush()

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )