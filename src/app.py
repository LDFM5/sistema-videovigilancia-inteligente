import threading
from flask import Flask, render_template, request, jsonify
import config
from main import ejecutar_sistema_principal

app = Flask(__name__)

class EstadoSistema:
    def __init__(self):
        # ¡Limpieza! Ya no necesitamos guardar self.frames aquí,
        # porque el video se va directo a MediaMTX.
        self.lock = threading.Lock()
        self.config_ram = {
            "UMBRAL_VELOCIDAD_GOLPE": 15.0,
            "UMBRAL_VELOCIDAD_CAIDA": 20.0
        }

estado = EstadoSistema()

@app.route('/')
def index():
    # Enviamos los nombres de las cámaras al HTML
    nombres_camaras = list(config.CAMERA_INDEXES.keys())
    return render_template('index.html', camaras=nombres_camaras)

# Mantenemos esta ruta porque la usas para cambiar configuraciones en vivo
@app.route('/update_config', methods=['POST'])
def update_config():
    data = request.json
    with estado.lock:
        for key in data:
            if key in estado.config_ram:
                estado.config_ram[key] = float(data[key])
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    # 1. Iniciamos el motor de IA en un hilo en segundo plano
    t = threading.Thread(target=ejecutar_sistema_principal, args=(estado,), daemon=True)
    t.start()
    
    # 2. Iniciamos el servidor de la página web
    print("🚀 Servidor Web iniciado. Entra a http://localhost:5000")
    print("📡 Recuerda tener abierta la consola de mediamtx.exe")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)