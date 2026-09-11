"""
train_yolo_weapons.py

Script de entrenamiento y ajuste fino (fine-tuning) para modelos de detección de
objetos y armas basados en la arquitectura YOLO (Ultralytics).

Permite:
- Comprobación y descarga automatizada del dataset de anotaciones (Roboflow / local).
- Entrenamiento supervisado con hiperparámetros optimizados para escenarios de videovigilancia.
- Aumentación de datos en tiempo real (condiciones de iluminación, escala y contraste).
- Limpieza automática de artefactos pesados temporales para optimizar el almacenamiento.
- Evaluación automática de métricas (mAP50, mAP50-95, precisión y recall) en el conjunto de prueba.
- Exportación directa del mejor punto de control (best.pt) para su integración en producción.
"""

import sys
import os
import glob
import shutil
import argparse
import subprocess
from typing import Optional, Union

import torch

# Asegurar codificación UTF-8 en terminales Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def obtener_dataset(
    api_key: Optional[str] = None,
    workspace: str = "public-security",
    project_name: str = "weapon-detection-test-zmk0t",
    version_num: int = 3
) -> str:
    """
    Verifica la presencia local del dataset o lo descarga mediante la API de Roboflow.

    Args:
        api_key: Clave de autenticación en Roboflow (o variable ROBOFLOW_API_KEY).
        workspace: Nombre del espacio de trabajo en Roboflow.
        project_name: Identificador del proyecto de anotaciones.
        version_num: Número de versión del conjunto de datos.

    Returns:
        Ruta absoluta al archivo data.yaml del dataset.
    """
    # 1. Comprobar si el dataset ya existe en el directorio de trabajo
    posibles_rutas = [
        f"Weapon-Detection-Test-{version_num}/data.yaml",
        f"Weapon-Detection-Test-3/data.yaml",
        f"{project_name}-{version_num}/data.yaml",
        f"weapon-detection-test-zmk0t-{version_num}/data.yaml"
    ]
    for ruta in posibles_rutas:
        if os.path.exists(ruta):
            print(f"[INFO] Dataset local localizado: {os.path.abspath(ruta)}")
            return os.path.abspath(ruta)

    # 2. Obtener clave de API desde argumento o variable de entorno
    clave_api = api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not clave_api:
        print("[ERROR] Dataset local no encontrado y no se proporcionó una clave de API de Roboflow.")
        print("        Especifica --api-key <TU_CLAVE> o define la variable de entorno ROBOFLOW_API_KEY.")
        sys.exit(1)

    print(f"[INFO] Descargando dataset: {project_name} (versión {version_num})...")

    # 3. Descarga directa mediante el enlace de exportación de Roboflow
    try:
        import urllib.request
        import json
        import zipfile

        api_url = f"https://api.roboflow.com/{workspace}/{project_name}/{version_num}/yolov11?api_key={clave_api}"
        req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            meta = json.loads(resp.read().decode())

        download_url = meta.get("export", {}).get("link")
        if download_url:
            dest_dir = f"Weapon-Detection-Test-{version_num}"
            os.makedirs(dest_dir, exist_ok=True)
            zip_path = "dataset_temp.zip"

            print("[INFO] Descargando archivo comprimido del dataset...")
            curl_res = subprocess.run(["curl", "-L", download_url, "-o", zip_path], check=False)
            if curl_res.returncode == 0 and os.path.exists(zip_path):
                print("[INFO] Descomprimiendo archivos...")
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(dest_dir)
                if os.path.exists(zip_path):
                    os.remove(zip_path)

                data_yaml = os.path.join(dest_dir, "data.yaml")
                if os.path.exists(data_yaml):
                    print(f"[INFO] Dataset preparado correctamente: {os.path.abspath(data_yaml)}")
                    return os.path.abspath(data_yaml)
    except Exception as e:
        print(f"[WARN] No se pudo completar la descarga directa ({e}). Intentando vía cliente Roboflow...")

    # 4. Método secundario mediante la librería oficial de Roboflow
    from roboflow import Roboflow
    rf = Roboflow(api_key=clave_api)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_num)
    dataset = version.download("yolov11")
    return os.path.abspath(os.path.join(dataset.location, "data.yaml"))


def entrenar(
    data_yaml: Optional[str] = None,
    model_size: str = "yolo11s.pt",
    epochs: int = 100,
    imgsz: int = 800,
    batch: int = 16,
    device: Union[int, str] = 0,
    workers: int = 8,
    project: Optional[str] = None,
    name: str = "exp",
    patience: int = 25,
    resume: bool = False,
    last_pt: Optional[str] = None,
    api_key: Optional[str] = None,
    workspace: str = "public-security",
    project_name: str = "weapon-detection-test-zmk0t",
    version_num: int = 3
):
    """
    Ejecuta el ciclo de entrenamiento supervisado con la arquitectura YOLO seleccionada.

    Args:
        data_yaml: Ruta al archivo descriptor del conjunto de datos.
        model_size: Pesos base de la arquitectura (ej. yolo11n.pt, yolo11s.pt, yolo11m.pt).
        epochs: Número máximo de épocas de entrenamiento.
        imgsz: Resolución espacial de las imágenes de entrada (ancho y alto).
        batch: Tamaño del lote por iteración.
        device: Identificador de la GPU CUDA o 'cpu'.
        workers: Número de hilos paralelos para carga de datos en DataLoader.
        project: Directorio donde se guardarán los resultados del experimento.
        name: Subdirectorio del experimento (se incrementa automáticamente si existe).
        patience: Épocas sin mejora antes de detener tempranamente el entrenamiento.
        resume: Indica si se reanuda un entrenamiento previo interrumpido.
        last_pt: Ruta al punto de control last.pt para reanudación.
        api_key: Clave de autenticación en Roboflow (opcional).
        workspace: Espacio de trabajo de Roboflow.
        project_name: Identificador del proyecto en Roboflow.
        version_num: Versión del dataset.
    """
    from ultralytics import YOLO

    if project is None:
        project = "/workspace/runs/detect" if os.path.exists("/workspace") else "runs/detect"

    print("=" * 70)
    print("CONFIGURACIÓN DEL ENTRENAMIENTO")
    print("=" * 70)
    print(f"Arquitectura base:      {model_size}")
    print(f"Resolución de entrada:  {imgsz}x{imgsz}")
    print(f"Épocas:                 {epochs}")
    print(f"Tamaño de lote (batch): {batch}")
    print(f"Paciencia (Early Stop): {patience}")
    print(f"Directorio de salida:   {project}/{name}")

    # Verificación de disponibilidad de GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"Acelerador CUDA:        {gpu_name} ({vram_gb:.2f} GB VRAM)")
        device_arg = device
    else:
        print("[AVISO] GPU CUDA no disponible. Ejecutando en CPU.")
        device_arg = "cpu"
        workers = 2
        batch = min(batch, 4)
    print("=" * 70)

    # Reanudación de entrenamiento
    if resume and last_pt:
        print(f"[INFO] Reanudando entrenamiento desde: {last_pt}")
        model = YOLO(last_pt)
        return model.train(resume=True)

    # Obtención del archivo data.yaml
    if not data_yaml or not os.path.exists(data_yaml):
        data_yaml = obtener_dataset(
            api_key=api_key,
            workspace=workspace,
            project_name=project_name,
            version_num=version_num
        )

    data_yaml_abs = os.path.abspath(data_yaml)
    if not os.path.exists(data_yaml_abs):
        print(f"[ERROR] Archivo no encontrado: {data_yaml_abs}")
        return None

    # Normalizar ruta raíz dentro del archivo data.yaml
    try:
        import yaml
        with open(data_yaml_abs, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
        yaml_data["path"] = os.path.dirname(data_yaml_abs).replace("\\", "/")
        with open(data_yaml_abs, "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f, sort_keys=False)
    except Exception as e:
        print(f"[WARN] Error al verificar campo 'path' en data.yaml: {e}")

    print(f"\n[INFO] Inicializando pesos preentrenados: {model_size}...")
    model = YOLO(model_size)

    print("[INFO] Iniciando ciclo de optimización con AdamW...")
    results = model.train(
        data=data_yaml_abs,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device_arg,
        workers=workers,
        project=project,
        name=name,
        exist_ok=False,
        patience=patience,
        save=True,
        save_period=-1,         # Evita guardar puntos de control en cada época para economizar espacio
        cache=False,
        amp=True,               # Precisión mixta automática para aceleración en GPU
        plots=True,             # Generación de curvas PR, F1 y matriz de confusión
        
        # Parámetros del optimizador
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,            # Decaimiento del ritmo de aprendizaje con curva coseno
        warmup_epochs=3.0,
        weight_decay=0.0005,
        
        # Ponderación de componentes de pérdida
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # Aumentación de datos adaptada a cámaras de vigilancia
        hsv_h=0.015,
        hsv_s=0.7,              # Variación de saturación para emular visión nocturna o infrarroja
        hsv_v=0.4,              # Variación de brillo para sombras y cambios de luz ambiental
        degrees=10.0,           # Rotación leve
        translate=0.1,          # Desplazamiento horizontal/vertical
        scale=0.4,              # Escalamiento para armas en primer plano y a distancia
        shear=2.0,              # Deformación tangencial leve
        perspective=0.0001,
        fliplr=0.5,             # Simetría horizontal
        flipud=0.0,             # Sin inversión vertical
        mosaic=1.0,             # Mosaico activo para entrenamiento con objetos pequeños
        mixup=0.0,
        close_mosaic=10,        # Desactiva mosaico en las últimas 10 épocas para estabilizar bordes
        erasing=0.0             # Desactiva borrado aleatorio para conservar armas de sección delgada
    )

    exp_dir = str(results.save_dir) if hasattr(results, "save_dir") else os.path.join(project, name)

    # Limpieza de imágenes temporales de entrenamiento para reducir volumen en disco
    print("\n[INFO] Limpiando archivos temporales de visualización de batches...")
    for patron in ["train_batch*.jpg", "val_batch*_labels.jpg"]:
        for arch in glob.glob(os.path.join(exp_dir, patron)):
            try:
                os.remove(arch)
            except OSError:
                pass

    # Eliminar checkpoints intermedios si existieran
    weights_dir = os.path.join(exp_dir, "weights")
    if os.path.exists(weights_dir):
        for arch in glob.glob(os.path.join(weights_dir, "epoch*.pt")):
            try:
                os.remove(arch)
            except OSError:
                pass

    print(f"\n[INFO] Entrenamiento finalizado. Resultados almacenados en: {exp_dir}")

    # Evaluación en el conjunto de prueba (Test Split)
    best_weights = os.path.join(weights_dir, "best.pt")
    if os.path.exists(best_weights):
        print("\n[INFO] Evaluando mejor modelo (best.pt) en el conjunto de prueba...")
        best_model = YOLO(best_weights)
        try:
            metrics = best_model.val(data=data_yaml_abs, split="test", imgsz=imgsz, device=device_arg, plots=True)
            print("\nMÉTRICAS EN CONJUNTO DE PRUEBA:")
            print(f"  - mAP@50:      {metrics.box.map50:.4f}")
            print(f"  - mAP@50-95:   {metrics.box.map:.4f}")
            print(f"  - Precisión:   {metrics.box.mp:.4f}")
            print(f"  - Recall:      {metrics.box.mr:.4f}")
        except Exception as e:
            print(f"[WARN] Error durante la evaluación del conjunto de prueba: {e}")

        # Copiar pesos a la raíz con el nombre estándar de producción
        archivo_prod = "Modelo_objetos_sospechosos.pt"
        shutil.copy2(best_weights, archivo_prod)
        print(f"\n[INFO] Pesos listos para producción exportados a: {os.path.abspath(archivo_prod)}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento y fine-tuning de detectores de objetos y armas (YOLO)."
    )
    parser.add_argument("--data", type=str, default=None, help="Ruta al archivo data.yaml (opcional si existe localmente)")
    parser.add_argument("--model", type=str, default="yolo11s.pt", choices=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"], help="Arquitectura o pesos base")
    parser.add_argument("--epochs", type=int, default=100, help="Número de épocas de entrenamiento")
    parser.add_argument("--imgsz", type=int, default=800, help="Resolución espacial de entrada")
    parser.add_argument("--batch", type=int, default=16, help="Tamaño del lote")
    parser.add_argument("--device", type=str, default="0", help="Identificador de GPU (ej. 0) o 'cpu'")
    parser.add_argument("--workers", type=int, default=8, help="Número de hilos de lectura para DataLoader")
    parser.add_argument("--patience", type=int, default=25, help="Épocas de paciencia para Early Stopping")
    parser.add_argument("--project", type=str, default=None, help="Directorio principal para guardar experimentos")
    parser.add_argument("--name", type=str, default="exp", help="Nombre del subdirectorio del experimento")
    parser.add_argument("--resume", action="store_true", help="Reanuda un entrenamiento interrumpido")
    parser.add_argument("--last", type=str, default=None, help="Ruta a last.pt cuando se utiliza --resume")
    parser.add_argument("--api-key", type=str, default=None, help="Clave de API de Roboflow (o definir variable ROBOFLOW_API_KEY)")
    parser.add_argument("--workspace", type=str, default="public-security", help="Nombre del espacio de trabajo en Roboflow")
    parser.add_argument("--project-name", type=str, default="weapon-detection-test-zmk0t", help="Identificador del proyecto en Roboflow")
    parser.add_argument("--version", type=int, default=3, help="Versión del dataset en Roboflow")
    args = parser.parse_args()

    dev = int(args.device) if args.device.isdigit() else args.device

    entrenar(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=dev,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        resume=args.resume,
        last_pt=args.last,
        api_key=args.api_key,
        workspace=args.workspace,
        project_name=args.project_name,
        version_num=args.version
    )


if __name__ == "__main__":
    main()
