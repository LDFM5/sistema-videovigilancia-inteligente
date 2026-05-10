import cv2
import threading
import time
from flask import Flask, render_template, Response, request, jsonify
import config
from main import ejecutar_sistema_principal

app = Flask(__name__)

class EstadoSistema:
    def __init__(self):
        self.frames = {} 
        self.lock = threading.Lock()
        self.config_ram = {
            "UMBRAL_VELOCIDAD_GOLPE": 15.0,
            "UMBRAL_VELOCIDAD_CAIDA": 20.0
        }

estado = EstadoSistema()

# El generador recibe qué cámara queremos transmitir
def generar_frames(cam_name, alta_calidad=False):
    while True:
        with estado.lock:
            if cam_name not in estado.frames or estado.frames[cam_name] is None:
                frame = None
            else:
                frame = estado.frames[cam_name].copy()

        if frame is None:
            time.sleep(0.1)
            continue

        alto, ancho = frame.shape[:2]

        # LÓGICA DINÁMICA DE CALIDAD
        if alta_calidad:
            nuevo_ancho = 1280 # HD
            calidad_jpeg = 85
        else:
            nuevo_ancho = 640  # SD para la cuadrícula
            calidad_jpeg = 60

        nuevo_alto = int((nuevo_ancho / ancho) * alto)
        frame_web = cv2.resize(frame, (nuevo_ancho, nuevo_alto))

        opciones_jpeg = [int(cv2.IMWRITE_JPEG_QUALITY), calidad_jpeg]
        ret, buffer = cv2.imencode('.jpg', frame_web, opciones_jpeg)
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.03)

# CAMBIO 3: Rutas de Flask actualizadas
@app.route('/')
def index():
    # Enviamos los nombres de las cámaras al HTML
    nombres_camaras = list(config.CAMERA_INDEXES.keys())
    return render_template('index.html', camaras=nombres_camaras)

# Actualizamos la ruta para que lea si pedimos HD
@app.route('/video_feed/<cam_name>')
def video_feed(cam_name):
    # Si la URL trae "?hq=true", activamos la alta calidad
    es_alta_calidad = request.args.get('hq') == 'true'
    return Response(generar_frames(cam_name, es_alta_calidad), mimetype='multipart/x-mixed-replace; boundary=frame')

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