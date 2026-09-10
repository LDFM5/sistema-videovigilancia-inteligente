### Versión en Español

# Sistema de Videovigilancia Inteligente

Sistema de videovigilancia en tiempo real para la detección de armas y comportamientos violentos a partir de múltiples fuentes de video (cámaras web, transmisiones RTSP y archivos de video).

El proyecto procesa los fotogramas en paralelo para identificar armas y evaluar patrones de movimiento violento, aplicando un filtro temporal para reducir falsas alarmas antes de registrar eventos o emitir notificaciones.

---

## Funcionalidades

* **Detección de objetos y armas:** Inferencia con YOLO (acelerada con FP16 en GPU) para detectar armas de fuego y armas blancas. El modelo también reconoce objetos cotidianos (teléfonos, billeteras) para reducir confusiones con objetos sostenidos en la mano.
* **Detección de violencia:** Clasificación de movimiento mediante TSM (Temporal Shift Module) sobre ventanas cortas de video (1.5 segundos).
* **Confirmación temporal:** Filtro lógico que requiere que una detección o agresión se mantenga durante un tiempo determinado antes de disparar una alerta, evitando falsos positivos por movimientos rápidos o transitorios.
* **Grabación de evidencias:** Búfer en memoria que guarda video de los segundos previos y posteriores a una alerta, generando un archivo MP4 con el incidente.
* **Panel web local:** Interfaz en Flask para visualizar las cámaras en tiempo real, ajustar umbrales de confianza y activar modos de depuración visual.
* **Notificaciones por Telegram:** Envío automático del video del incidente mediante un bot de Telegram al confirmarse una alerta.

---

## Estructura del Código (`src/`)

```text

src/
├── app.py              # Servidor web Flask y rutas de la interfaz
├── behavior_cnn.py     # Carga e inferencia del modelo temporal de violencia (TSM)
├── cameras.py          # Captura y lectura de video desde USB, RTSP o archivos locales
├── config.py           # Configuración de rutas, umbrales y parámetros por defecto
├── detection.py        # Inferencia de YOLO para detección de armas y objetos
├── main.py             # Bucle principal que coordina cámaras, modelos y grabaciones
├── recorder.py         # Búfer circular para almacenar video previo y posterior a la alerta
├── streamer.py         # Transmisión de video en vivo vía MJPEG para el navegador
├── telegram_bot.py     # Manejador del bot de Telegram para envío de clips
├── temporal_logic.py   # Lógica temporal para confirmación y descarte de alertas
├── visualization.py    # Dibujo de recuadros y textos sobre los fotogramas
└── templates/
    └── index.html      # Plantilla web del panel de monitoreo
```

---

## Requisitos

* Python 3.10 o superior.
* Tarjeta gráfica NVIDIA con soporte CUDA (recomendado para ejecución en tiempo real).
* Dependencias principales: PyTorch, Ultralytics (YOLO), OpenCV, Flask, Torchvision.

---

## Uso

1. Clonar el repositorio:
```bash
git clone https://github.com/luisd/sistema-videovigilancia-inteligente.git
cd sistema-videovigilancia-inteligente
```

2. Instalar las dependencias necesarias:
```bash
pip install -r requirements.txt
```

3. Colocar los modelos entrenados en el directorio configurado en `src/config.py`:
* Modelo YOLO de armas (`.pt`).
* Modelo de comportamiento violento (`.pth`).

4. Ejecutar la aplicación:
```bash
python src/app.py
```

5. Abrir el navegador en `http://localhost:5000`.

---

### English version

# Intelligent Video Surveillance System

A real-time video surveillance system for detecting weapons and violent behavior across multiple video feeds (webcams, RTSP streams, and local video files).

The application processes frames in parallel to identify weapons and classify physical aggression patterns, using a temporal confirmation filter to reduce false alarms before recording incident clips or dispatching notifications.

---

## Features

* **Weapon and Object Detection:** YOLO-based inference (with FP16 GPU acceleration) targeting firearms and bladed weapons. The model also recognizes common handheld objects (such as phones and wallets) to reduce false positives.
* **Violence Recognition:** Action classification using a Temporal Shift Module (TSM) network over short temporal sliding windows (1.5 seconds).
* **Temporal Confirmation:** A rule-based temporal filter that requires detections to persist over time before triggering an alert, filtering out brief transient movements.
* **Automated Clip Recording:** A rolling memory buffer records seconds before and after an incident, exporting an MP4 clip as evidence.
* **Local Web Dashboard:** A Flask interface to view camera feeds in real time, adjust confidence thresholds, and toggle diagnostic overlays.
* **Telegram Notifications:** Automatically dispatches recorded incident clips through a Telegram bot upon confirmed alerts.

---

## Code Structure (`src/`)

```text
src/
├── app.py              # Flask web server and dashboard routes
├── behavior_cnn.py     # Model loading and inference for behavior classification (TSM)
├── cameras.py          # Video capture from USB devices, RTSP streams, or files
├── config.py           # Path definitions, thresholds, and default settings
├── detection.py        # YOLO inference handler for weapons and objects
├── main.py             # Main coordinator loop for cameras, models, and recording
├── recorder.py         # Circular memory buffer for pre- and post-incident recording
├── streamer.py         # MJPEG live video streamer for browser display
├── telegram_bot.py     # Telegram bot handler for dispatching video clips
├── temporal_logic.py   # State logic for temporal confirmation of alerts
├── visualization.py    # Drawing bounding boxes and diagnostic text on frames
└── templates/
    └── index.html      # Web dashboard template
```

---

## Requirements

* Python 3.10 or higher.
* NVIDIA GPU with CUDA support (recommended for real-time performance).
* Core libraries: PyTorch, Ultralytics (YOLO), OpenCV, Flask, Torchvision.

---

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/luisd/sistema-videovigilancia-inteligente.git
cd sistema-videovigilancia-inteligente
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place the trained models in the paths defined in `src/config.py`:
* YOLO weapon model (`.pt`).
* Violent behavior model (`.pth`).

4. Start the application:
```bash
python src/app.py
```

5. Open your browser and go to `http://localhost:5000`.
