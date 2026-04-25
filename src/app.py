import cv2
import threading
import time
from flask import Flask, render_template, Response, request, jsonify

# Importamos tu lógica de los otros archivos
import config
from main import ejecutar_sistema_principal 

app = Flask(__name__)

# --- MEMORIA COMPARTIDA SEGURA ---
class EstadoSistema:
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock() # Evita conflictos de lectura/escritura
        self.config_ram = {
            "UMBRAL_VELOCIDAD_GOLPE": config.UMBRAL_VELOCIDAD_GOLPE,
            "UMBRAL_VELOCIDAD_CAIDA": config.UMBRAL_VELOCIDAD_CAIDA
        }

estado = EstadoSistema()

def generate_frames():
    while True:
        frame_actual = None
        
        # 1. Entramos rápido, copiamos y SALIMOS del candado
        with estado.lock:
            if estado.frame is not None:
                frame_actual = estado.frame.copy()
        
        # 2. Si YOLO aún no arranca, esperamos pacientemente
        if frame_actual is None:
            time.sleep(0.1) # ¡Esta es la línea que salva el sistema!
            continue
            
        # 3. Procesamos la imagen de OpenCV a Web FUERA del candado 
        # (Así el motor puede seguir trabajando sin estorbos)
        ret, buffer = cv2.imencode('.jpg', frame_actual)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.03) # Límite de 30 FPS web

@app.route('/')
def index():
    # Enviamos la configuración actual a la página
    return render_template('index.html', conf=estado.config_ram)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_config', methods=['POST'])
def update_config():
    data = request.json
    with estado.lock:
        for key in data:
            if key in estado.config_ram:
                estado.config_ram[key] = float(data[key])
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    # Iniciamos el motor de main.py en un hilo
    # Pasamos el objeto 'estado' para que main.py escriba ahí
    t = threading.Thread(target=ejecutar_sistema_principal, args=(estado,), daemon=True)
    t.start()
    
    # Iniciamos la web
    print("🚀 Dashboard iniciado en http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)