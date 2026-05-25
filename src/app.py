import threading
import signal

from flask import Flask, render_template, request, jsonify

import config
from main import ejecutar_sistema_principal

app = Flask(__name__)


class EstadoSistema:

    def __init__(self):

        self.lock = threading.Lock()


estado = EstadoSistema()


@app.route('/')
def index():

    nombres_camaras = list(config.CAMERA_INDEXES.keys())

    return render_template(
        'index.html',
        camaras=nombres_camaras
    )

def update_config():

    data = request.json

    with estado.lock:

        for key in data:

            if key in estado.config_ram:
                estado.config_ram[key] = float(data[key])

    return jsonify({"status": "ok"})


# ==========================================
# CIERRE LIMPIO
# ==========================================
def cerrar_servidor(_sig, _frame):

    print("\n🛑 Cerrando servidor Flask...")
    raise KeyboardInterrupt

signal.signal(signal.SIGINT, cerrar_servidor)


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":

    # THREAD IA
    t = threading.Thread(
        target=ejecutar_sistema_principal,
        args=(estado,),
        daemon=True,
        name="AI-System-Thread"
    )

    t.start()

    print("🚀 Servidor Web iniciado")
    print("🌐 http://localhost:5000")
    print("📡 MediaMTX debe estar activo")

    # FLASK
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )