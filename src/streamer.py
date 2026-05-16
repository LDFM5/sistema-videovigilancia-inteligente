"""
streamer.py

Streamer RTSP robusto y optimizado para baja latencia.
Compatible con MediaMTX + WebRTC.
"""

import subprocess
import cv2
import threading
import queue
import time


class RTSPStreamer:

    def __init__(self, cam_name, width=640, height=480, fps=15):

        self.cam_name = cam_name
        self.width = width
        self.height = height
        self.fps = fps

        self.rtsp_url = f"rtsp://localhost:8554/{cam_name}"

        print(f"📡 Inicializando RTSP -> {self.rtsp_url}")

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

            # ==========================================
            # INPUT
            # ==========================================
            '-f', 'rawvideo',

            '-vcodec', 'rawvideo',

            '-pix_fmt', 'bgr24',

            '-s', f'{width}x{height}',

            '-r', str(fps),

            '-i', '-',

            # ==========================================
            # OUTPUT
            # ==========================================
            '-an',

            '-c:v', 'libx264',

            '-preset', 'ultrafast',

            '-tune', 'zerolatency',

            '-pix_fmt', 'yuv420p',

            '-profile:v', 'baseline',

            '-g', '10',

            '-f', 'rtsp',

            '-rtsp_transport', 'tcp',

            self.rtsp_url
        ]

        # ======================================================
        # LANZAR FFMPEG
        # ======================================================
        self.process = subprocess.Popen(

            command,

            stdin=subprocess.PIPE,

            stderr=subprocess.PIPE
        )

        # ======================================================
        # THREAD STREAMING
        # ======================================================
        self.thread = threading.Thread(
            target=self._stream_loop,
            daemon=True,
            name=f"Streamer-{cam_name}"
        )

        self.thread.start()

    # ==========================================================
    # LOOP STREAMING
    # ==========================================================
    def _stream_loop(self):

        while self.running:

            try:

                frame = self.frame_queue.get(timeout=0.1)

            except queue.Empty:
                continue

            try:

                self.process.stdin.write(
                    frame.tobytes()
                )

            except Exception as e:

                print(f"❌ FFmpeg error ({self.cam_name}): {e}")

                try:

                    err = self.process.stderr.read().decode()

                    print(err)

                except:
                    pass

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

        print(f"🛑 Cerrando stream -> {self.cam_name}")

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