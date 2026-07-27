"""
streamer.py

Subsistema de transmisión RTSP industrial optimizado para baja latencia.
Despliegue robusto compatible con MediaMTX y WebRTC.
"""

import subprocess
import cv2
import threading
import queue
import time


class RTSPStreamer:

    def __init__(self, cam_name, width=800, height=450, fps=15, shared_state=None):
        self.cam_name = cam_name.upper()
        self.width = width
        self.height = height
        self.fps = fps
        self.shared_state = shared_state 
        
        self.rtsp_url = f"rtsp://localhost:8554/{cam_name.lower()}"

        print(f"INIT_KERNEL: INITIALIZING RTSP TRANSMISSION -> {self.rtsp_url}")

        # Búfer circular acotado a 1 solo cuadro para garantizar cero acumulación de lag
        self.frame_queue = queue.Queue(maxsize=1)
        self.running = True

        # ======================================================
        # COMANDO FFMPEG ALTAMENTE OPTIMIZADO PARA BAJA LATENCIA
        # ======================================================
        command = [
            'ffmpeg',
            '-y', # Sobrescribir sin preguntar

            # INPUT (Inyección directa BGR24 desde la memoria RAM)
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-i', '-',

            # OUTPUT (Empaquetado H.264 para MediaMTX)
            '-an',                    # Desactivar canal de audio
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-profile:v', 'baseline',
            '-pix_fmt', 'yuv420p',
            
            # Ajustes de latencia extrema
            '-bf', '0',               # Sin B-frames para evitar retrasos en el decodificador
            '-max_delay', '0',        # Salida TCP inmediata
            '-threads', '2',          # Límite estricto de hilos para no saturar la CPU
            
            # Control de tasa de bits
            '-b:v', '800k',           # Flujo de 800 kbps por canal
            '-maxrate', '800k',
            '-bufsize', '1600k',
            '-g', str(self.fps),      # 1 I-Frame por segundo para rápida recuperación de señal
            
            '-f', 'rtsp',
            '-rtsp_transport', 'tcp', # Protocolo TCP para entrega secuencial y ordenada
            self.rtsp_url
        ]

        # ======================================================
        # LANZAMIENTO DE SUBPROCESO FFMPEG
        # ======================================================
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Hilo de lectura continua para evitar desbordamiento en la tubería stderr del SO
        self.error_reader_thread = threading.Thread(
            target=self._consume_stderr,
            daemon=True,
            name=f"FFmpeg-Stderr-{self.cam_name}"
        )
        self.error_reader_thread.start()

        # Hilo secundario para inyección asíncrona de cuadros
        self.thread = threading.Thread(
            target=self._stream_loop,
            daemon=True,
            name=f"Streamer-{self.cam_name}"
        )
        self.thread.start()

    def _consume_stderr(self):
        """Drena el canal de errores de FFmpeg en segundo plano para evitar bloqueos del SO."""
        try:
            while self.running and self.process and self.process.poll() is None:
                line = self.process.stderr.readline()
                if not line:
                    break
        except Exception:
            pass

    # ==========================================================
    # BUCLE PRINCIPAL DE TRANSMISIÓN
    # ==========================================================
    def _stream_loop(self):
        if self.shared_state:
            self.shared_state.emitir_evento_dashboard('system_log', {
                "type": "success", 
                "message": f"TUBO_FFMPEG: INYECCIÓN DE FLUJO ESTABLECIDA EN CANAL: {self.cam_name}"
            })

        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                if self.process and self.process.stdin:
                    self.process.stdin.write(frame.tobytes())
                    self.process.stdin.flush()
            except (BrokenPipeError, OSError, Exception) as e:
                if self.shared_state:
                    self.shared_state.emitir_evento_dashboard('camera_status', {
                        "camera": self.cam_name.lower(), 
                        "status": "error"
                    })
                    self.shared_state.emitir_evento_dashboard('system_log', {
                        "type": "error", 
                        "message": f"NÚCLEO_FLUJO_RTSP: TUBERÍA ROTA EN CANAL '{self.cam_name}' -> REF: {str(e).upper()}"
                    })
                break

    # ==========================================================
    # RECEPCIÓN Y ENCOLAMIENTO DE FOTOGRAMAS
    # ==========================================================
    def enviar_frame(self, frame):
        if not self.running or frame is None:
            return

        # 🚀 OPTIMIZACIÓN: Redimensionar únicamente si el fotograma no coincide con la resolución objetivo
        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            frame_to_send = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        else:
            frame_to_send = frame

        # Descartar fotograma antiguo si la cola está llena
        try:
            if self.frame_queue.full():
                self.frame_queue.get_nowait()
        except queue.Empty:
            pass

        # Colocar fotograma fresco
        try:
            self.frame_queue.put_nowait(frame_to_send)
        except queue.Full:
            pass

    # ==========================================================
    # CIERRE Y LIBERACIÓN LIMPIA DE RECURSOS
    # ==========================================================
    def cerrar(self):
        print(f"SYS_KERNEL: TERMINATING STREAM LINK FOR SOURCE -> {self.cam_name}")
        self.running = False

        if self.process:
            try:
                if self.process.stdin:
                    self.process.stdin.close()
            except Exception:
                pass

            try:
                self.process.terminate()
                self.process.wait(timeout=1.0)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None

        print(f"INFO: Conexión RTSP cerrada correctamente para fuente: {self.cam_name}")