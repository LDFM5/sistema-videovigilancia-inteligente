"""
streamer.py

Módulo de transmisión de video a nivel industrial.
Utiliza FFmpeg para comprimir frames crudos de OpenCV a H.264 
y los inyecta en un servidor RTSP (MediaMTX).
"""
import subprocess
import cv2

class RTSPStreamer:
    def __init__(self, cam_name, width=640, height=480, fps=15):
        self.cam_name = cam_name
        self.width = width
        self.height = height
        
        # URL donde escucha MediaMTX por defecto
        self.rtsp_url = f"rtsp://localhost:8554/{cam_name}"
        
        print(f"📡 Iniciando pipeline RTSP para {cam_name} -> {self.rtsp_url}")
        
        # Comando mágico de FFmpeg: Toma video crudo (raw), lo comprime a H264 sin latencia (zerolatency)
        # Comando mágico de FFmpeg adaptado 100% para WebRTC en navegadores
        # Comando mágico de FFmpeg para WebRTC
        command = [
            'ffmpeg',
            '-y', 
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f"{width}x{height}",
            # '-r', str(fps),  <-- ELIMINAMOS ESTA LÍNEA para que no fuerce los FPS de entrada
            '-i', '-',
            
            # --- CONFIGURACIÓN DE SALIDA ---
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-pix_fmt', 'yuv420p',
            '-profile:v', 'baseline',
            '-rtsp_transport', 'tcp',
            '-g', '30', # Forzamos un fotograma clave cada 30 frames para limpiar tirones
            
            '-f', 'rtsp',
            self.rtsp_url
        ]
        
        # Ocultamos la salida de FFmpeg para no ensuciar la consola
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def enviar_frame(self, frame):
        try:
            # Aseguramos que el frame tenga el tamaño exacto que FFmpeg espera
            frame_resized = cv2.resize(frame, (self.width, self.height))
            self.process.stdin.write(frame_resized.tobytes())
        except Exception as e:
            print(f"❌ Error inyectando frame RTSP ({self.cam_name}): {e}")

    def cerrar(self):
        self.process.stdin.close()
        self.process.wait()