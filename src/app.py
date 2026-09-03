"""
app.py

Servidor web de control y distribución de eventos mediante Server-Sent Events (SSE).
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
        # Búfer circular con los últimos 200 eventos.
        self.historial_eventos = deque(maxlen=200)

    def emitir_evento_dashboard(self, event_type, data):
        """
        Propaga un evento estructurado a todas las conexiones web activas.
        """
        with self.lock:
            if event_type == 'system_log':
                log_msg = data.get("message", "")
                t_type = data.get("type", "info").upper()
                print(f"[{t_type}] {log_msg}")
                sys.stdout.flush()

            paquete_estructurado = {
                "tipo_evento": event_type,
                "contenido": data
            }

            # Guardar el evento en el historial reciente.
            self.historial_eventos.append(paquete_estructurado)

            # Copiar la lista de suscriptores antes de liberar el bloqueo.
            suscriptores = list(self.suscriptores_activos)

        # Despacho fuera del lock principal
        for q in suscriptores:
            try:
                if q.full():
                    q.get_nowait()
                q.put_nowait(paquete_estructurado)
            except Exception:
                pass


# Instancia compartida del estado del sistema.
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
    convirtiendo cada valor al tipo de dato esperado.
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
        "message": f"Configuración actualizada: {', '.join(data.keys())}."
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
            "message": "La configuración se guardó correctamente."
        })
        return jsonify({"status": "SUCCESS", "code": 200})
        
    except Exception as e:
        try:
            estado.emitir_evento_dashboard('system_log', {
                "type": "error", 
                "message": f"No se pudo guardar la configuración: {e}"
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
            "message": "Se restablecieron los valores predeterminados."
        })
        return jsonify({"status": "SUCCESS", "data": valores_defecto, "code": 200})
        
    except Exception as e:
        try:
            estado.emitir_evento_dashboard('system_log', {
                "type": "error", 
                "message": f"No se pudieron restablecer los valores predeterminados: {e}"
            })
        except Exception:
            pass
        return jsonify({"status": "ERROR", "code": 500})


# ==========================================
# ENDPOINTS REST: GESTIÓN DINÁMICA DE CÁMARAS
# ==========================================
@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    """Retorna la lista de cámaras configuradas."""
    cams = config.obtener_camaras_configuradas()
    return jsonify({"status": "SUCCESS", "cameras": cams, "code": 200})


@app.route('/api/cameras', methods=['POST'])
def add_or_update_camera():
    """Agrega o actualiza una cámara (USB por índice o IP por URL RTSP)."""
    data = request.json or {}
    cam_name = str(data.get("name", "")).strip().lower()
    source_val = str(data.get("source", "")).strip()

    if not cam_name or not source_val:
        return jsonify({"status": "ERROR", "message": "El nombre y la fuente son obligatorios.", "code": 400})

    try:
        source = int(source_val) if source_val.isdigit() else source_val
    except Exception:
        source = source_val

    cfg = config.cargar_configuracion_inicial()
    if "camaras" not in cfg:
        cfg["camaras"] = {}
    cfg["camaras"][cam_name] = source
    config.guardar_configuracion_disco(cfg)

    estado.emitir_evento_dashboard('system_log', {
        "type": "success",
        "message": f"Cámara '{cam_name}' guardada ({source})."
    })
@app.route('/api/cameras/<cam_name>', methods=['DELETE'])
def delete_camera(cam_name):
    """Elimina una cámara de la configuración."""
    cam_name = str(cam_name).strip().lower()
    cfg = config.cargar_configuracion_inicial()
    if "camaras" in cfg and cam_name in cfg["camaras"]:
        del cfg["camaras"][cam_name]
        config.guardar_configuracion_disco(cfg)
        estado.emitir_evento_dashboard('system_log', {
            "type": "warn",
            "message": f"Cámara '{cam_name}' eliminada de la configuración."
        })
        return jsonify({"status": "SUCCESS", "cameras": cfg["camaras"], "code": 200})
    return jsonify({"status": "ERROR", "message": "Cámara no encontrada.", "code": 404})


@app.route('/api/scan_cameras', methods=['GET'])
def scan_cameras():
    """Escanea cámaras locales USB y genera miniaturas para previsualización."""
    from cameras import escanear_camaras_locales
    try:
        cams = escanear_camaras_locales()
        return jsonify({"status": "SUCCESS", "cameras": cams, "code": 200})
    except Exception as e:
        return jsonify({"status": "ERROR", "message": str(e), "code": 500})


@app.route('/api/test_camera', methods=['POST'])
def test_camera():
    """Prueba si una fuente USB, RTSP o archivo entrega video."""
    from cameras import probar_fuente_camara
    data = request.json or {}
    source = data.get("source", "")
    if source is None or str(source).strip() == "":
        return jsonify({"status": "ERROR", "message": "Debes especificar una fuente.", "code": 400})

    resultado = probar_fuente_camara(source)
    return jsonify(resultado)


@app.route('/api/scan_network_rtsp', methods=['GET'])
def scan_network_rtsp():
    """Escanea la subred en busca de dispositivos con puerto RTSP 554 abierto."""
    from cameras import escanear_camaras_red_rtsp
    try:
        res = escanear_camaras_red_rtsp()
        return jsonify({"status": "SUCCESS", "data": res, "code": 200})
    except Exception as e:
        return jsonify({"status": "ERROR", "message": str(e), "code": 500})


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
    print("\n[INFO] Se recibió una solicitud de cierre.")
    print("[INFO] Deteniendo el motor de inferencia y el servidor web.")
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

    print("[INFO] Servidor de control iniciado.")
    print("[INFO] Interfaz disponible en http://localhost:5000.")
    sys.stdout.flush()

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )
