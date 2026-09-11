### Versión en Español

# Sistema de Videovigilancia Inteligente

Sistema de videovigilancia en tiempo real para la detección de armas y comportamientos violentos a partir de múltiples fuentes de video (cámaras web, transmisiones RTSP y archivos de video).

El proyecto procesa los fotogramas en paralelo para identificar armas y evaluar patrones de movimiento violento, aplicando un filtro temporal para reducir falsas alarmas antes de registrar eventos o emitir notificaciones.

---

## Funcionalidades

* **Detección de objetos y armas:** Inferencia con YOLO (acelerada con FP16 en GPU) para detectar armas de fuego y armas blancas. El modelo también reconoce objetos cotidianos (teléfonos, billeteras) para reducir confusiones con objetos sostenidos en la mano.
* **Detección de violencia:** Clasificación de movimiento mediante TSM (Temporal Shift Module) sobre ventanas cortas de video (1.5 segundos).
* **Confirmación temporal:** Filtro lógico que requiere que una detección o agresión se mantenga durante un tiempo determinado antes de disparar una alerta, evitando falsos positivos por movimientos rápidos o transitorios.
* **Grabación de evidencias:** Búfer en memoria que guarda video de los segundos previos y posteriores a una alerta, generando un archivo MP4 con el incidente y metadatos OSD.
* **Transmisión de ultra baja latencia:** Distribución de video en vivo mediante RTSP y WebRTC (WHEP) a través de MediaMTX y FFmpeg con codificación acelerada por hardware (NVENC).
* **Panel web y selector de modelos:** Interfaz en Flask para monitoreo en vivo, ajuste de umbrales en caliente y navegación libre por el sistema de archivos para cargar cualquier modelo (`.pt`, `.engine`, `.onnx`, `.pth`) sin reiniciar el servicio.
* **Notificaciones por Telegram:** Envío automático del video del incidente mediante un bot de Telegram al confirmarse una alerta.

---

## Estructura del Proyecto

```text
├── src/
│   ├── app.py              # Servidor web Flask, API REST y rutas de la interfaz
│   ├── behavior_cnn.py     # Carga e inferencia del modelo temporal de violencia (TSM)
│   ├── cameras.py          # Captura y lectura de video desde USB, RTSP o archivos locales
│   ├── config.py           # Configuración de rutas, resolución de modelos y parámetros del sistema
│   ├── detection.py        # Inferencia de YOLO para detección de armas y objetos
│   ├── main.py             # Bucle principal que coordina cámaras, modelos y grabaciones
│   ├── recorder.py         # Búfer circular para almacenar video previo y posterior a la alerta
│   ├── streamer.py         # Transmisor RTSP de baja latencia hacia MediaMTX mediante FFmpeg
│   ├── telegram_bot.py     # Manejador del bot de Telegram para envío de clips
│   ├── temporal_logic.py   # Lógica temporal para confirmación y descarte de alertas
│   ├── visualization.py    # Dibujo de recuadros, máscaras y OSD adaptable en cuadrícula y evidencia
│   └── templates/
│       └── index.html      # Plantilla web del panel de monitoreo y explorador de archivos
├── train_tsm_behavior.py   # Script de entrenamiento y evaluación del clasificador TSM de comportamiento
├── train_yolo_weapons.py   # Script de entrenamiento y fine-tuning del detector YOLO
└── requirements.txt        # Dependencias de Python del proyecto
```

---

## Requisitos del Sistema

* **Python 3.10 o superior**.
* **Tarjeta gráfica NVIDIA con soporte CUDA** (recomendado para inferencia en tiempo real y codificación de video por hardware NVENC).
* **MediaMTX** (v1.0.0 o superior): Servidor multimedia de transmisión en tiempo real que convierte las transmisiones RTSP a WebRTC/WHEP para reproducirlas en el navegador sin retraso.
  * Descarga: [MediaMTX Releases en GitHub](https://github.com/bluenviron/mediamtx/releases).
* **FFmpeg**: Herramienta de codificación de video añadida al `PATH` del sistema (con soporte para `h264_nvenc` o `libx264`).
  * Descarga: [FFmpeg Oficial](https://ffmpeg.org/download.html).
* **Dependencias de Python**: PyTorch, Ultralytics (YOLO), OpenCV, Flask, Torchvision (ver `requirements.txt`).

---

## Uso

1. **Clonar el repositorio:**
```bash
git clone https://github.com/luisd/sistema-videovigilancia-inteligente.git
cd sistema-videovigilancia-inteligente
```

2. **Instalar las dependencias de Python:**
```bash
pip install -r requirements.txt
```

3. **Iniciar el servidor MediaMTX:**
En una terminal independiente, ejecutar el binario de MediaMTX:
```bash
# En Windows:
mediamtx.exe

# En Linux / macOS:
./mediamtx
```
MediaMTX quedará escuchando en los puertos `8554` (RTSP) y `8889` (WebRTC/HTTP).

4. **Verificar FFmpeg en el sistema:**
Asegúrate de que `ffmpeg` esté en las variables de entorno ejecutando:
```bash
ffmpeg -version
```

5. **Colocar los modelos entrenados:**
Puedes colocar los modelos en la carpeta `models/` o seleccionarlos directamente desde el explorador de archivos en la barra de Ajustes de la interfaz web:
* Modelo YOLO de objetos/armas (`.pt`, `.engine` o `.onnx`).
* Modelo de comportamiento violento (`.pth`).

6. **Ejecutar la aplicación:**
```bash
python src/app.py
```

7. **Monitoreo:**
Abrir el navegador web en `http://localhost:5000`.

---

### English version

# Intelligent Video Surveillance System

A real-time video surveillance system for detecting weapons and violent behavior across multiple video feeds (webcams, RTSP streams, and local video files).

The application processes frames in parallel to identify weapons and classify physical aggression patterns, using a temporal confirmation filter to reduce false alarms before recording incident clips or dispatching notifications.

---

## Features

* **Weapon and Object Detection:** YOLO-based inference (accelerated with FP16 GPU inference) targeting firearms and bladed weapons, alongside common everyday items (phones, wallets) to reduce false positives from held objects.
* **Violence Recognition:** Action classification using a Temporal Shift Module (TSM) network over short temporal sliding windows (1.5 seconds).
* **Temporal Confirmation:** A rule-based temporal filter requiring detections to persist before triggering an alert, suppressing transient glitches and brief movements.
* **Automated Clip Recording:** Rolling memory buffer capturing footage before and after an incident, exporting an MP4 evidence clip with customized OSD overlays.
* **Ultra-Low Latency Streaming:** Real-time video delivery via RTSP and WebRTC (WHEP) powered by MediaMTX and FFmpeg with NVENC hardware encoding.
* **Web Dashboard & Model Browser:** Flask-powered dashboard with live monitoring, runtime threshold adjustments, and an integrated system file explorer to load custom models (`.pt`, `.engine`, `.onnx`, `.pth`) on the fly without stopping streams.
* **Telegram Notifications:** Automatically dispatches recorded incident clips through a Telegram bot upon confirmed alerts.

---

## Project Structure

```text
├── src/
│   ├── app.py              # Flask web server, REST API, and dashboard routes
│   ├── behavior_cnn.py     # Model loading and inference for behavior classification (TSM)
│   ├── cameras.py          # Video capture from USB devices, RTSP streams, or files
│   ├── config.py           # Path definitions, dynamic model resolution, and defaults
│   ├── detection.py        # YOLO inference handler for weapons and objects
│   ├── main.py             # Main coordinator loop for cameras, models, and recording
│   ├── recorder.py         # Circular memory buffer for pre- and post-incident recording
│   ├── streamer.py         # Low-latency RTSP video streamer to MediaMTX using FFmpeg
│   ├── telegram_bot.py     # Telegram bot handler for dispatching video clips
│   ├── temporal_logic.py   # State logic for temporal confirmation of alerts
│   ├── visualization.py    # Bounding boxes, alert banners, and proportional OSD rendering
│   └── templates/
│       └── index.html      # Web dashboard template with integrated file explorer modal
├── train_tsm_behavior.py   # Training and evaluation script for the TSM behavior classifier
├── train_yolo_weapons.py   # Training and fine-tuning script for the YOLO detector
└── requirements.txt        # Python library dependencies
```

---

## System Requirements

* **Python 3.10 or higher**.
* **NVIDIA GPU with CUDA support** (recommended for real-time inference and hardware NVENC encoding).
* **MediaMTX** (v1.0.0 or higher): Real-time multimedia server that bridges incoming RTSP streams to WebRTC/WHEP for sub-second browser playback.
  * Download: [MediaMTX Releases on GitHub](https://github.com/bluenviron/mediamtx/releases).
* **FFmpeg**: Video encoding suite accessible via system `PATH` (with `h264_nvenc` or `libx264` support).
  * Download: [FFmpeg Official Website](https://ffmpeg.org/download.html).
* **Python Libraries**: PyTorch, Ultralytics (YOLO), OpenCV, Flask, Torchvision (see `requirements.txt`).

---

## Getting Started

1. **Clone the repository:**
```bash
git clone https://github.com/luisd/sistema-videovigilancia-inteligente.git
cd sistema-videovigilancia-inteligente
```

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

3. **Start the MediaMTX server:**
In a separate terminal, launch the MediaMTX executable:
```bash
# On Windows:
mediamtx.exe

# On Linux / macOS:
./mediamtx
```
MediaMTX will bind to port `8554` (RTSP ingestion) and `8889` (WebRTC/HTTP streaming).

4. **Verify FFmpeg:**
Ensure `ffmpeg` is available in your system environment variables:
```bash
ffmpeg -version
```

5. **Provide Model Weights:**
You can place model weights inside `models/` or pick any file anywhere on your filesystem using the Settings panel file browser:
* YOLO objects/weapons model (`.pt`, `.engine`, or `.onnx`).
* Violent behavior model (`.pth`).

6. **Run the Application:**
```bash
python src/app.py
```

7. **Access the Dashboard:**
Open your browser and navigate to `http://localhost:5000`.
