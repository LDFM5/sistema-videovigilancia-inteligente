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

    def __init__(self, cam_name, width=640, height=480, fps=15, shared_state=None):
        self.cam_name = cam_name.upper()
        self.width = width
        self.height = height
        self.fps = fps
        # Guardar la referencia del estado unificado
        self.shared_state = shared_state 
        
        self.rtsp_url = f"rtsp://localhost:8554/{cam_name}"

        # Consola nativa usando nomenclatura estándar de microkernel
        print(f"INIT_KERNEL: INITIALIZING RTSP TRANSMISSION -> {self.rtsp_url}")

        # ======================================================
        # QUEUE DE SOLO 1 FRAME
        # ======================================================
        self.frame_queue = queue.Queue(maxsize=1)

        self.running = True

        # ======================================================
        # COMANDO FFMPEG
        # ======================================================
        command = [
            'ffmpeg',
            '-y', # Sobrescribir sin preguntar

            # ==========================================
            # INPUT (Lectura desde OpenCV)
            # ==========================================
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}',
            '-r', str(fps),
            '-i', '-',

            # ==========================================
            # OUTPUT (Inyección a MediaMTX)
            # ==========================================
            '-an',                    # Sin audio (ahorra ancho de banda)
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-profile:v', 'baseline',
            '-pix_fmt', 'yuv420p',
            
            # --- TWEAKS DE LATENCIA EXTREMA ---
            '-bf', '0',               # Fuerza 0 B-frames (cero predicción temporal = cero lag)
            '-max_delay', '0',        # Prohíbe a FFmpeg retener paquetes en su salida TCP
            '-threads', '2',          # Limita los hilos. Si FFmpeg usa todos tus núcleos, ahoga a YOLO.
            
            # --- CONTROL DE TRÁFICO (VITAL PARA MULTI-CÁMARA) ---
            '-b:v', '800k',           # Target de 800 kbps por cámara
            '-maxrate', '800k',       # Máximo estricto
            '-bufsize', '1600k',      # Buffer de red al doble del maxrate (Estándar CBR)
            '-g', str(fps),           # Un I-Frame por segundo exacto (recuperación rápida si hay pérdida)
            
            '-f', 'rtsp',
            '-rtsp_transport', 'tcp', # TCP asegura que los frames lleguen en orden
            self.rtsp_url
        ]

        # ======================================================
        # LANZAR FFMPEG
        # ======================================================
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE # El pipe de error requiere lectura constante no bloqueante
        )

        # OPTIMIZACIÓN: Hilo secundario para limpiar constantemente el búfer stderr de FFmpeg
        # Esto previene interrupciones críticas (Deadlock de tubería del SO)
        self.error_reader_thread = threading.Thread(
            target=self._consume_stderr,
            daemon=True,
            name=f"FFmpeg-Stderr-{self.cam_name}"
        )
        self.error_reader_thread.start()

        # ======================================================
        # THREAD STREAMING
        # ======================================================
        self.thread = threading.Thread(
            target=self._stream_loop,
            daemon=True,
            name=f"Streamer-{self.cam_name}"
        )

        self.thread.start()

    def _consume_stderr(self):
        """Consume el canal de errores asincrónicamente para que el buffer del SO no se desborde."""
        try:
            while self.running and self.process.poll() is None:
                line = self.process.stderr.readline()
                if not line:
                    break
                # Se puede descomentar en fases avanzadas de diagnóstico de red
                # print(f"FFMPEG_RAW_LOG ({self.cam_name}): {line.decode().strip()}")
        except:
            pass

    # ==========================================================
    # LOOP STREAMING
    # ==========================================================
    def _stream_loop(self):
        # 🚨 SOLUCIÓN COMPLETADA: Ya no hay 'from app import...'
        if self.shared_state:
            self.shared_state.emitir_evento_dashboard('system_log', {
                "type": "success", 
                "message": f"TUBO_FFMPEG: INYECCIÓN DE FLUJO ESTABLECIDA PARA LA FUENTE: {self.cam_name}"
            })

        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self.process.stdin.write(frame.tobytes())
                self.process.stdin.flush() 
            except Exception as e:
                if self.shared_state:
                    self.shared_state.emitir_evento_dashboard('camera_status', {
                        "camera": self.cam_name.lower(), 
                        "status": "error"
                    })
                    self.shared_state.emitir_evento_dashboard('system_log', {
                        "type": "error", 
                        "message": f"NÚCLEO_FLUJO_RTSP: TUBERÍA ROTA EN CANAL '{self.cam_name}' -> CÓDIGO_REF: {str(e).upper()}"
                    })
                break

    # ==========================================================
    # ENVIAR FRAME
    # ==========================================================
    def enviar_frame(self, frame):

        if not self.running:
            return

        # ==========================================
        # RESIZE
        # ==========================================
        frame = cv2.resize(
            frame,
            (self.width, self.height)
        )

        # ==========================================
        # ELIMINAR FRAME VIEJO
        # ==========================================
        try:
            if self.frame_queue.full():
                self.frame_queue.get_nowait()
        except:
            pass

        # ==========================================
        # AGREGAR FRAME NUEVO
        # ==========================================
        try:
            self.frame_queue.put_nowait(frame)
        except:
            pass

    # ==========================================================
    # CERRAR
    # ==========================================================
    def cerrar(self):

        print(f"SYS_KERNEL: TERMINATING STREAM LINK FOR SOURCE -> {self.cam_name}")

        self.running = False
        time.sleep(0.2)

        try:
            if self.process.stdin:
                self.process.stdin.close()
        except:
            pass

        try:
            self.process.kill()
        except:
            pass

        try:
            self.process.wait(timeout=1)
        except:
            pass