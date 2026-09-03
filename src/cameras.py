"""
cameras.py

Captura de video desde cámaras USB, fuentes de red (RTSP) y archivos de video locales:
- Captura continua mediante hilos independientes
- Conservación del fotograma más reciente y escalado en segundo plano
- Selección de backend (DirectShow para USB y automático para RTSP/archivos)
- Reconexión asíncrona ante interrupciones del flujo
- Soporte para bucle infinito en archivos de video locales (ideal para pruebas)
- Herramientas de descubrimiento y diagnóstico (escáner USB local, escáner de red RTSP y prueba de fuente)
"""

import os
import cv2
import threading
import time
import socket
import base64
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor

from config import CAMERA_INDEXES


def normalizar_fuente(fuente):
    """Limpia comillas y resuelve rutas absolutas para archivos de video locales o URLs."""
    if isinstance(fuente, int):
        return fuente
    if isinstance(fuente, str):
        f_str = fuente.strip().strip('"').strip("'")
        if f_str.isdigit():
            return int(f_str)
        if os.path.isfile(f_str):
            return os.path.abspath(f_str)
        # Buscar relativo a la raíz del proyecto
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidate = os.path.join(project_root, f_str)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
        return f_str
    return fuente


# ==========================================================
# CLASE DE CÁMARA CON RESILIENCIA DE RED Y HARDWARE
# ==========================================================
class CameraStream:

    def __init__(self, cam_name, cam_index):
        self.cam_name = cam_name.upper()
        self.cam_index = normalizar_fuente(cam_index)
        self.cap = None

        # Identificar si es un archivo de video local
        self.es_archivo_local = isinstance(self.cam_index, str) and os.path.isfile(self.cam_index)

        # Dimensiones estandarizadas para la tubería principal (Grid 16:9)
        self.width = 800
        self.height = 450
        self.fps = 30

        # Estructuras de memoria compartida
        self.latest_frame = None
        self.latest_frame_id = 0
        self.latest_frame_time = None
        self.lock = threading.Lock()
        self.running = True
        self.estado_error_enviado = False

        # Puntero para la emisión de eventos al dashboard
        self.shared_state = None

        # Intentar apertura inicial
        self._intentar_conexion_fisica()

        # Hilo asíncrono de captura continua
        self.thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name=f"CameraThread-{self.cam_name}"
        )
        self.thread.start()

    def _intentar_conexion_fisica(self):
        """
        Ejecuta la apertura del descriptor de OpenCV adaptando el backend
        según el tipo de fuente (Índice entero USB vs Cadena RTSP/Archivo).
        """
        try:
            if self.cap is not None:
                self.cap.release()

            # Seleccionar el backend según el tipo de fuente.
            if isinstance(self.cam_index, int):
                # Utilizar DirectShow para cámaras USB en Windows.
                self.cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
            else:
                # Backend predeterminado para rutas RTSP, HTTP o archivos locales
                self.cap = cv2.VideoCapture(self.cam_index)
            
            if not self.cap.isOpened():
                return False

            # Configuraciones de hardware para reducir latencia en cámaras USB
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if isinstance(self.cam_index, int):
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_FPS, 30)

            # Obtener FPS entregados por la fuente
            hardware_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if hardware_fps > 1 and hardware_fps <= 60:
                self.fps = int(hardware_fps)

            return True
        except Exception:
            return False

    # ======================================================
    # BUCLE DE CAPTURA CON TOLERANCIA A FALLOS
    # ======================================================
    def _capture_loop(self):
        consecutivos_fallidos = 0
        
        while self.running:
            start_iter = time.monotonic()

            if self.cap is None or not self.cap.isOpened():
                ret, frame = False, None
            else:
                try:
                    ret, frame = self.cap.read()
                except Exception:
                    ret, frame = False, None

            if not ret or frame is None:
                # Si es un archivo de video local y llegó al final, reiniciar en bucle
                if self.es_archivo_local and self.cap is not None:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    try:
                        ret, frame = self.cap.read()
                    except Exception:
                        ret, frame = False, None

                if not ret or frame is None:
                    consecutivos_fallidos += 1
                    
                    # Umbral de tolerancia: 45 cuadros fallidos continuos
                    if consecutivos_fallidos >= 45:
                        if self.shared_state and not self.estado_error_enviado:
                            self.shared_state.emitir_evento_dashboard('camera_status', {
                                "camera": self.cam_name.lower(),
                                "status": "error"
                            })
                            self.shared_state.emitir_evento_dashboard('system_log', {
                                "type": "error",
                                "message": f"Se interrumpió la conexión con la cámara {self.cam_name}."
                            })
                            self.estado_error_enviado = True

                        # Protocolo de reintento cada 4 segundos
                        time.sleep(4.0)
                        self._intentar_conexion_fisica()
                    else:
                        time.sleep(0.02)
                    continue

            # Recuperación exitosa de la señal
            consecutivos_fallidos = 0
            if self.estado_error_enviado:
                if self.shared_state:
                    self.shared_state.emitir_evento_dashboard('camera_status', {
                        "camera": self.cam_name.lower(),
                        "status": "analyzing"
                    })
                    self.shared_state.emitir_evento_dashboard('system_log', {
                        "type": "success",
                        "message": f"Se restableció la conexión con la cámara {self.cam_name}."
                    })
                self.estado_error_enviado = False

            # Escalar el fotograma en el hilo de captura.
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame_resized = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            else:
                frame_resized = frame

            with self.lock:
                self.latest_frame = frame_resized
                self.latest_frame_id += 1
                self.latest_frame_time = time.monotonic()

            # Si es un archivo de video local, regular la tasa de fotogramas para velocidad real
            if self.es_archivo_local:
                elapsed = time.monotonic() - start_iter
                target_period = 1.0 / max(1, self.fps)
                sleep_time = target_period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    # ======================================================
    # LECTURA RÁPIDA (MEMORIA RAM COMPARTIDA)
    # ======================================================
    def read(self):
        with self.lock:
            if self.latest_frame is None:
                return False, None
            return (
                True,
                self.latest_frame.copy(),
                self.latest_frame_id,
                self.latest_frame_time,
            )

    # ======================================================
    # LIBERACIÓN DE RECURSOS
    # ======================================================
    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
            
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
        print(f"[INFO] Recursos liberados para la cámara {self.cam_name}.")


# ==========================================================
# INICIALIZACIÓN DE CÁMARAS
# ==========================================================
def initialize_cameras():
    cameras = {}
    camera_resolutions = {}
    camera_fps = {}

    for cam_name, cam_index in CAMERA_INDEXES.items():
        cam = CameraStream(cam_name, cam_index)
        cameras[cam_name] = cam
        camera_resolutions[cam_name.upper()] = (cam.width, cam.height)
        camera_fps[cam_name.upper()] = cam.fps

    return cameras, camera_resolutions, camera_fps


# ==========================================================
# HERRAMIENTAS DE DESCUBRIMIENTO Y DIAGNÓSTICO DE CÁMARAS
# ==========================================================

def _obtener_nombres_dispositivos_windows():
    """Consulta los nombres de cámaras conectadas mediante PowerShell en Windows."""
    try:
        cmd = ['powershell', '-NoProfile', '-Command', 'Get-PnpDevice -Class Camera,Image | Select-Object -Property FriendlyName | ConvertTo-Json']
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=2.5)
        if res.returncode == 0 and res.stdout.strip():
            data = json.loads(res.stdout)
            if isinstance(data, dict):
                data = [data]
            return [d.get('FriendlyName') for d in data if d.get('FriendlyName')]
    except Exception:
        pass
    return []


def escanear_camaras_locales(max_index=6):
    """
    Escanea índices de cámaras USB (0 a max_index), capturando resolución,
    FPS y una miniatura base64 para previsualización inmediata en el dashboard.
    """
    nombres_dispositivos = _obtener_nombres_dispositivos_windows()
    camaras_detectadas = []

    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)

                # Generar miniatura liviana en base64 (160x90)
                thumb = cv2.resize(frame, (160, 90))
                _, buf = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 60])
                b64 = base64.b64encode(buf).decode('utf-8')

                nombre_amigable = nombres_dispositivos[idx] if idx < len(nombres_dispositivos) else f"Cámara USB ({idx})"
                camaras_detectadas.append({
                    "index": idx,
                    "name": nombre_amigable,
                    "resolution": f"{w}x{h}",
                    "fps": fps,
                    "thumbnail": f"data:image/jpeg;base64,{b64}"
                })
            cap.release()

    return camaras_detectadas


def probar_fuente_camara(source_val):
    """
    Prueba si una fuente dada (índice entero, URL RTSP/HTTP, o archivo MP4) es accesible.
    Devuelve estado, resolución, FPS y miniatura.
    """
    src = normalizar_fuente(source_val)

    backend = cv2.CAP_DSHOW if isinstance(src, int) else cv2.CAP_ANY
    cap = cv2.VideoCapture(src, backend) if isinstance(src, int) else cv2.VideoCapture(src)

    if not cap.isOpened():
        return {"status": "ERROR", "message": "No se pudo conectar con la fuente especificada."}

    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        return {"status": "ERROR", "message": "La fuente abrió pero no entregó fotogramas válidos."}

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)

    thumb = cv2.resize(frame, (160, 90))
    _, buf = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 65])
    b64 = base64.b64encode(buf).decode('utf-8')
    cap.release()

    return {
        "status": "SUCCESS",
        "resolution": f"{w}x{h}",
        "fps": fps,
        "thumbnail": f"data:image/jpeg;base64,{b64}"
    }


def escanear_camaras_red_rtsp():
    """
    Escanea la subred local buscando direcciones IP con el puerto RTSP (554) abierto.
    """
    # Detectar prefijo de la subred local
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        local_ip = s.getsockname()[0]
        prefix = '.'.join(local_ip.split('.')[:3])
    except Exception:
        prefix = '192.168.1'
    finally:
        s.close()

    def _check_host(ip):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.25)
        try:
            res = sock.connect_ex((ip, 554))
            if res == 0:
                return {
                    "ip": ip,
                    "port": 554,
                    "suggested_url": f"rtsp://admin:password@{ip}:554/stream1"
                }
        except Exception:
            pass
        finally:
            sock.close()
        return None

    ips = [f"{prefix}.{i}" for i in range(1, 255)]
    with ThreadPoolExecutor(max_workers=60) as executor:
        results = [r for r in executor.map(_check_host, ips) if r is not None]

    return {"subnet": prefix, "found": results}
