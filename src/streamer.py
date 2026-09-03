"""
streamer.py

Transmisión RTSP optimizada para baja latencia.
Compatible con MediaMTX y WebRTC.
"""

import subprocess
import cv2
import threading
import queue
import time

_NVENC_AVAILABLE = None
_NVENC_CHECK_LOCK = threading.Lock()

def _verificar_soporte_nvenc():
    """Detecta si FFmpeg soporta h264_nvenc."""
    global _NVENC_AVAILABLE
    with _NVENC_CHECK_LOCK:
        if _NVENC_AVAILABLE is not None:
            return _NVENC_AVAILABLE
        try:
            res = subprocess.run(
                ['ffmpeg', '-hide_banner', '-encoders'],
                capture_output=True, text=True, timeout=5,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
            )
            _NVENC_AVAILABLE = 'h264_nvenc' in res.stdout
        except Exception:
            _NVENC_AVAILABLE = False
        return _NVENC_AVAILABLE


class RTSPStreamer:

    def __init__(self, cam_name, width=800, height=450, fps=15, shared_state=None):
        self.cam_name = cam_name.upper()
        self.width = width
        self.height = height
        self.fps = fps
        self.shared_state = shared_state 
        
        self.rtsp_url = f"rtsp://localhost:8554/{cam_name.lower()}"
        self.usar_nvenc = _verificar_soporte_nvenc()

        print(f"[INFO] Iniciando la transmisión RTSP en {self.rtsp_url} (Encoder: {'h264_nvenc' if self.usar_nvenc else 'libx264'}).")

        # Mantener únicamente el fotograma más reciente para evitar latencia acumulada.
        self.frame_queue = queue.Queue(maxsize=1)
        self.running = True
        self.process = None
        self.process_lock = threading.Lock()
        self.restart_failures = 0
        self.successful_writes = 0

        self.command = self._construir_comando_ffmpeg(self.usar_nvenc)

        # ======================================================
        # LANZAMIENTO DE SUBPROCESO FFMPEG
        # ======================================================
        try:
            self._start_ffmpeg()
        except Exception as error:
            self._emit_stream_error(error)

        # Hilo secundario para inyección asíncrona de cuadros
        self.thread = threading.Thread(
            target=self._stream_loop,
            daemon=True,
            name=f"Streamer-{self.cam_name}"
        )
        self.thread.start()

    def _construir_comando_ffmpeg(self, usar_nvenc):
        cmd = [
            'ffmpeg',
            '-y',
            # INPUT (Inyección directa BGR24 desde RAM)
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-i', '-',
            '-an',
        ]

        if usar_nvenc:
            cmd.extend([
                '-c:v', 'h264_nvenc',
                '-preset', 'p1',          # Preset más rápido y liviano en hardware
                '-tune', 'ull',           # Ultra-low latency
                '-zerolatency', '1',
                '-pix_fmt', 'yuv420p',
                '-bf', '0',
                '-max_delay', '0',
                '-b:v', '800k',
                '-maxrate', '800k',
                '-bufsize', '1600k',
                '-g', str(self.fps),
            ])
        else:
            cmd.extend([
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-tune', 'zerolatency',
                '-profile:v', 'baseline',
                '-pix_fmt', 'yuv420p',
                '-bf', '0',
                '-max_delay', '0',
                '-threads', '2',
                '-b:v', '800k',
                '-maxrate', '800k',
                '-bufsize', '1600k',
                '-g', str(self.fps),
            ])

        cmd.extend([
            '-f', 'rtsp',
            '-rtsp_transport', 'tcp',
            self.rtsp_url
        ])
        return cmd

    def _start_ffmpeg(self):
        try:
            process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            with self.process_lock:
                self.process = process

            threading.Thread(
                target=self._consume_stderr,
                args=(process,),
                daemon=True,
                name=f"FFmpeg-Stderr-{self.cam_name}",
            ).start()
        except Exception as e:
            if self.usar_nvenc:
                print(f"[WARN] Falló inicialización de NVENC para {self.cam_name}, conmutando a libx264...")
                self.usar_nvenc = False
                self.command = self._construir_comando_ffmpeg(False)
                self._start_ffmpeg()
            else:
                raise e

    def _consume_stderr(self, process):
        """Drena el canal de errores de FFmpeg en segundo plano para evitar bloqueos del SO."""
        try:
            while self.running and process.poll() is None:
                line = process.stderr.readline()
                if not line:
                    break
        except Exception:
            pass

    def _stop_ffmpeg(self):
        with self.process_lock:
            process = self.process
            self.process = None

        if process is None:
            return
        try:
            if process.stdin:
                process.stdin.close()
        except Exception:
            pass
        try:
            process.terminate()
            process.wait(timeout=1.0)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    def _emit_stream_error(self, error):
        if self.shared_state:
            self.shared_state.emitir_evento_dashboard('camera_status', {
                "camera": self.cam_name.lower(),
                "status": "error",
            })
            self.shared_state.emitir_evento_dashboard('system_log', {
                "type": "error",
                "message": f"Falló la transmisión RTSP de {self.cam_name}: {error}",
            })

    def _restart_ffmpeg(self, error):
        self._stop_ffmpeg()
        self.restart_failures += 1
        self.successful_writes = 0
        delay = min(5.0, 0.25 * (2 ** min(self.restart_failures - 1, 5)))
        self._emit_stream_error(error)

        deadline = time.monotonic() + delay
        while self.running and time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            time.sleep(min(0.1, remaining))
        if not self.running:
            return False

        try:
            self._start_ffmpeg()
            if self.shared_state:
                self.shared_state.emitir_evento_dashboard('system_log', {
                    "type": "success",
                    "message": f"Se restableció la transmisión RTSP de {self.cam_name}.",
                })
            return True
        except Exception as restart_error:
            self._emit_stream_error(restart_error)
            return False

    # ==========================================================
    # BUCLE PRINCIPAL DE TRANSMISIÓN
    # ==========================================================
    def _stream_loop(self):
        if self.shared_state:
            self.shared_state.emitir_evento_dashboard('system_log', {
                "type": "success", 
                "message": f"Transmisión de video disponible para {self.cam_name}."
            })

        interval = 1.0 / max(1.0, float(self.fps))
        next_write_time = time.monotonic()
        latest_frame = None

        while self.running:
            now = time.monotonic()
            timeout = max(0.0, min(0.1, next_write_time - now))
            try:
                latest_frame = self.frame_queue.get(timeout=timeout)
                # Si llegaron varios frames entre dos pulsos, conservar solo
                # el más reciente para no acumular latencia.
                while True:
                    latest_frame = self.frame_queue.get_nowait()
            except queue.Empty:
                pass

            if not self.running:
                break

            now = time.monotonic()
            if latest_frame is None or now < next_write_time:
                continue

            process = self.process
            if process is None or process.poll() is not None:
                self._restart_ffmpeg("PROCESO FFMPEG NO DISPONIBLE")
                next_write_time = time.monotonic() + interval
                continue

            try:
                if process.stdin:
                    process.stdin.write(latest_frame.tobytes())
                    process.stdin.flush()
                self.successful_writes += 1
                if self.successful_writes >= self.fps * 2:
                    self.restart_failures = 0
            except (BrokenPipeError, OSError, ValueError) as error:
                self._restart_ffmpeg(error)
                next_write_time = time.monotonic() + interval
                continue

            # Mantener 15 pulsos reales por segundo. Si FFmpeg se bloqueó, no
            # enviar una ráfaga atrasada: retomar desde el reloj actual.
            next_write_time += interval
            if next_write_time < now:
                next_write_time = now + interval

    # ==========================================================
    # RECEPCIÓN Y ENCOLAMIENTO DE FOTOGRAMAS
    # ==========================================================
    def enviar_frame(self, frame):
        if not self.running or frame is None:
            return

        # Redimensionar únicamente cuando el fotograma no coincide con la resolución objetivo.
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

        # Añadir el fotograma más reciente.
        try:
            self.frame_queue.put_nowait(frame_to_send)
        except queue.Full:
            pass

    # ==========================================================
    # CIERRE Y LIBERACIÓN DE RECURSOS
    # ==========================================================
    def cerrar(self):
        print(f"[INFO] Deteniendo la transmisión RTSP de {self.cam_name}.")
        self.running = False
        self._stop_ffmpeg()
        if (
            hasattr(self, "thread")
            and self.thread.is_alive()
            and threading.current_thread() is not self.thread
        ):
            self.thread.join(timeout=1.0)

        print(f"[INFO] Transmisión RTSP cerrada para {self.cam_name}.")
