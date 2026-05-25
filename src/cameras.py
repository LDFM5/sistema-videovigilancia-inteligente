"""
cameras.py

Sistema de captura optimizado:
- Captura continua por hilos independientes
- Actualización en tiempo real (Latest-frame-only)
- Baja latencia mediante buffer acotado
- Inicializacion en HD nativo y escalado simetrico por software
"""

import cv2
import threading
import time

from config import CAMERA_INDEXES


# ==========================================================
# CLASE DE CÁMARA
# ==========================================================
class CameraStream:

    def __init__(self, cam_name, cam_index):

        self.cam_name = cam_name
        self.cam_index = cam_index

        # Se fuerza el backend DirectShow (CAP_DSHOW) para asegurar soporte HD en Windows
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

        # Configuraciones de hardware criticas para baja latencia
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Se solicita una resolucion panoramica estandar admitida por el hardware
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"CRITICAL: Error de inicializacion en dispositivo de captura: {cam_name}"
            )

        # Estandarizacion de dimensiones para el pipeline de inferencia y streaming
        self.width = 800
        self.height = 450

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 1:
            self.fps = 30

        print(f"INFO: Canal {cam_name} inicializado correctamente a {self.width}x{self.height} via escalado de software")
        print(f"INFO: Tasa de refresco asignada para {cam_name}: {self.fps} FPS")

        # Estructuras de memoria compartida para concurrencia
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True

        # Hilo asincrono secundario de captura continua
        self.thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name=f"CameraThread-{cam_name}"
        )
        self.thread.start()

    # ======================================================
    # LOOP DE CAPTURA
    # ======================================================
    def _capture_loop(self):

        while self.running:
            ret, frame = self.cap.read()

            if not ret:
                time.sleep(0.01)
                continue

            with self.lock:
                self.latest_frame = frame

    # ======================================================
    # OBTENER ÚLTIMO FRAME (PROCESAMIENTO DIGITAL DE SEÑAL)
    # ======================================================
    def read(self):

        with self.lock:
            if self.latest_frame is None:
                return False, None

            frame_copy = self.latest_frame.copy()

        # Reduccion al tamaño del grid manteniendo proporciones de aspecto panoramicas nativas
        if frame_copy.shape[1] != 800 or frame_copy.shape[0] != 450:
            frame_copy = cv2.resize(frame_copy, (800, 450), interpolation=cv2.INTER_LINEAR)

        return True, frame_copy

    # ======================================================
    # LIBERAR HARDWARE
    # ======================================================
    def release(self):
        self.running = False
        time.sleep(0.2)
        self.cap.release()
        print(f"INFO: Recursos de hardware liberados para canal: {self.cam_name}")


# ==========================================================
# INICIALIZAR CÁMARAS
# ==========================================================
def initialize_cameras():

    cameras = {}
    camera_resolutions = {}
    camera_fps = {}

    for cam_name, cam_index in CAMERA_INDEXES.items():
        try:
            cam = CameraStream(
                cam_name,
                cam_index
            )

            cameras[cam_name] = cam
            camera_resolutions[cam_name] = (cam.width, cam.height)
            camera_fps[cam_name] = cam.fps

        except Exception as e:
            print(f"ERROR: No se pudo establecer enlace con {cam_name}. Detalles: {e}")

    return cameras, camera_resolutions, camera_fps