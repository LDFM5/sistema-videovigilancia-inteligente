"""
cameras.py

Sistema de captura optimizado:
- Captura continua por hilos independientes (Threaded Capture)
- Estrategia Latest-Frame-Only con escalado asíncrono en segundo plano
- Selección inteligente de backend (DirectShow para USB / Auto para RTSP)
- Motor adaptativo de autoreconexión asíncrona ante caídas de flujo
"""

import cv2
import threading
import time

from config import CAMERA_INDEXES


# ==========================================================
# CLASE DE CÁMARA CON RESILIENCIA DE RED Y HARDWARE
# ==========================================================
class CameraStream:

    def __init__(self, cam_name, cam_index):
        self.cam_name = cam_name.upper()
        self.cam_index = cam_index
        self.cap = None

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

            # Selección inteligente de backend
            if isinstance(self.cam_index, int):
                # Backend DirectShow optimizado para Windows
                self.cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
            else:
                # Backend predeterminado para rutas RTSP, HTTP o archivos
                self.cap = cv2.VideoCapture(self.cam_index)
            
            if not self.cap.isOpened():
                return False

            # Configuraciones de hardware para reducir latencia
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
    # BUCLE DE CAPTURA CON TOLERANCIA A FALLOS INDUSTRIAL
    # ======================================================
    def _capture_loop(self):
        consecutivos_fallidos = 0
        
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                ret, frame = False, None
            else:
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
                            "message": f"CAPTURADOR_FLUJO: CONEXIÓN INTERRUMPIDA EN CANAL '{self.cam_name}'."
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
                        "message": f"CAPTURADOR_FLUJO: SEÑAL RECONECTADA CON LA FUENTE: {self.cam_name}."
                    })
                self.estado_error_enviado = False

            # 🚀 OPTIMIZACIÓN: Escalado asíncrono en segundo plano para liberar al hilo principal
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame_resized = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            else:
                frame_resized = frame

            with self.lock:
                self.latest_frame = frame_resized
                self.latest_frame_id += 1
                self.latest_frame_time = time.monotonic()

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
    # LIBERACIÓN SEGURA DE RECURSOS
    # ======================================================
    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
            
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
        print(f"INFO: Recursos de hardware liberados para canal: {self.cam_name}")


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
