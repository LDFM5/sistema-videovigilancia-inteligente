"""
cameras.py

Sistema de captura optimizado:
- Captura continua por hilos independientes
- Actualización en tiempo real (Latest-frame-only)
- Baja latencia mediante buffer acotado
- Inicialización en HD nativo y escalado simétrico por software
- Motor adaptativo de autoreconexión asíncrona ante caídas de flujo
"""

import cv2
import threading
import time

from config import CAMERA_INDEXES


# ==========================================================
# CLASE DE CÁMARA CON RESILIENCIA DE RED
# ==========================================================
class CameraStream:

    def __init__(self, cam_name, cam_index):
        self.cam_name = cam_name.upper()
        self.cam_index = cam_index
        self.cap = None

        # Estandarización obligatoria de dimensiones para el pipeline core
        self.width = 800
        self.height = 450
        self.fps = 30

        # Intentar la apertura inicial en frío del dispositivo de hardware
        self._intentar_conexion_fisica()

        # Estructuras de memoria compartida para concurrencia limpia
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True
        self.estado_error_enviado = False

        # Puntero diferido para el bus de datos compartido (Sincronizado dinámicamente)
        self.shared_state = None

        # Hilo asíncrono secundario de captura continua y monitoreo perimetral
        self.thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name=f"CameraThread-{cam_name}"
        )
        self.thread.start()

    def _intentar_conexion_fisica(self):
        """
        Ejecuta la apertura del descriptor de OpenCV aplicando configuraciones
        de hardware restrictivas para asegurar la baja latencia en el bus.
        """
        try:
            if self.cap is not None:
                self.cap.release()

            # Se fuerza el backend DirectShow (CAP_DSHOW) para soporte nativo estable
            self.cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
            
            # Configuraciones críticas de hardware para mitigar el lag acumulativo
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            # Extraer los cuadros por segundo reales que entrega el hardware
            hardware_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if hardware_fps > 1:
                self.fps = int(hardware_fps)

            return self.cap.isOpened()
        except Exception:
            return False

    # ======================================================
    # LOOP DE CAPTURA CON TOLERANCIA A FALLOS INDUSTRIAL
    # ======================================================
    def _capture_loop(self):
        consecutivos_fallidos = 0
        
        while self.running:
            # Si el descriptor físico falló en el arranque o fue liberado, forzar reconexión
            if self.cap is None or not self.cap.isOpened():
                ret, frame = False, None
            else:
                ret, frame = self.cap.read()

            if not ret:
                consecutivos_fallidos += 1
                
                # Umbral de tolerancia: Si fallan más de 45 cuadros seguidos (~1.5 a 3 segundos de congelamiento)
                if consecutivos_fallidos >= 45:
                    if self.shared_state and not self.estado_error_enviado:
                        self.shared_state.emitir_evento_dashboard('camera_status', {
                            "camera": self.cam_name.lower(),
                            "status": "error"
                        })
                        self.shared_state.emitir_evento_dashboard('system_log', {
                            "type": "error",
                            "message": f"CAPTURADOR_FLUJO: CONEXIÓN INTERRUMPIDA CON EL CANAL '{self.cam_name}'."
                        })
                        self.estado_error_enviado = True

                    # Protocolo asíncrono de re-enganche de señal: Reintentar abrir cada 4 segundos
                    time.sleep(4.0)
                    self._intentar_conexion_fisica()
                else:
                    time.sleep(0.02)
                continue

            # Si el frame es exitoso, resetear el contador de fallos y restablecer el dashboard
            consecutivos_fallidos = 0
            if self.estado_error_enviado:
                if self.shared_state:
                    self.shared_state.emitir_evento_dashboard('camera_status', {
                        "camera": self.cam_name.lower(),
                        "status": "analyzing"
                    })
                    self.shared_state.emitir_evento_dashboard('system_log', {
                        "type": "success",
                        "message": f"CAPTURADOR_FLUJO: SEÑAL RECOVERY RECONECTADA CON LA FUENTE: {self.cam_name}."
                    })
                self.estado_error_enviado = False

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

        # Reducción al tamaño del grid manteniendo proporciones de aspecto panorámicas nativas
        if frame_copy.shape[1] != 800 or frame_copy.shape[0] != 450:
            frame_copy = cv2.resize(frame_copy, (800, 450), interpolation=cv2.INTER_LINEAR)

        return True, frame_copy

    # ======================================================
    # LIBERAR HARDWARE
    # ======================================================
    def release(self):
        self.running = False
        time.sleep(0.2)
        if self.cap is not None:
            self.cap.release()
        print(f"INFO: Recursos de hardware liberados para canal: {self.cam_name}")


# ==========================================================
# INICIALIZAR CÁMARAS (ENFOQUE TOLERANTE EN FRÍO)
# ==========================================================
def initialize_cameras():
    cameras = {}
    camera_resolutions = {}
    camera_fps = {}

    for cam_name, cam_index in CAMERA_INDEXES.items():
        # Inicializar la instancia de forma segura. Si el hardware no está conectado físicamente,
        # la clase no colapsará el backend, sino que el hilo asíncrono lo buscará en caliente.
        cam = CameraStream(cam_name, cam_index)

        cameras[cam_name] = cam
        camera_resolutions[cam_name.upper()] = (cam.width, cam.height)
        camera_fps[cam_name.upper()] = cam.fps

    return cameras, camera_resolutions, camera_fps