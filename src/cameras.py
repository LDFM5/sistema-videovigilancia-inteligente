"""
cameras.py

Sistema de captura optimizado:
- Captura continua por thread
- Latest-frame-only
- Baja latencia
- Evita buffering acumulado
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

        self.cap = cv2.VideoCapture(cam_index)

        # ==========================================
        # CONFIGURACIÓN OPTIMIZADA
        # ==========================================
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Resolución razonable para tiempo real
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

        # FPS
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():

            raise RuntimeError(
                f"❌ No se pudo abrir cámara: {cam_name}"
            )

        # ==========================================
        # INFO REAL
        # ==========================================
        self.width = int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        self.height = int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        if self.fps <= 1:
            self.fps = 30

        print(f"📷 {cam_name}: {self.width}x{self.height}")
        print(f"🎥 FPS: {self.fps}")

        # ==========================================
        # FRAME COMPARTIDO
        # ==========================================
        self.latest_frame = None

        self.lock = threading.Lock()

        self.running = True

        # ==========================================
        # THREAD INTERNO
        # ==========================================
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
    # OBTENER ÚLTIMO FRAME
    # ======================================================
    def read(self):

        with self.lock:

            if self.latest_frame is None:
                return False, None

            return True, self.latest_frame.copy()

    # ======================================================
    # LIBERAR
    # ======================================================
    def release(self):

        self.running = False

        time.sleep(0.2)

        self.cap.release()


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

            camera_resolutions[cam_name] = (
                cam.width,
                cam.height
            )

            camera_fps[cam_name] = cam.fps

        except Exception as e:

            print(e)

    return cameras, camera_resolutions, camera_fps