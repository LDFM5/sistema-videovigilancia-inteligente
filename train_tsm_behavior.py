"""
train_tsm_behavior.py

Script de entrenamiento y fine-tuning del modelo de clasificación de comportamiento
y detección de violencia basado en la arquitectura ResNet-18 con módulos de
desplazamiento temporal (Temporal Shift Module - TSM).

Características principales:
- Arquitectura convolucional temporal 2D eficiente para secuencias de 16 fotogramas.
- División estratificada por identificador de video (GroupKFold / GroupSplit) para evitar contaminación entre entrenamiento y validación.
- Función de pérdida Balanced Focal Loss con suavizado de etiquetas (Label Smoothing).
- Optimización con tasas de aprendizaje diferenciales para el extractor convolucional y la cabeza clasificadora.
- Aumentaciones espaciotemporales sincronizadas (recortes aleatorios, rotación leve, variación de iluminación y Video Cutout).
- Generación automática de métricas, curvas de convergencia (Loss/Accuracy), matriz de confusión y reporte de clasificación.
- Exportación del mejor punto de control como 'best_model.pth' y 'comportamiento.pth'.
"""

import sys
import os
import glob
import random
from collections import defaultdict
from typing import List, Tuple, Set

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import argparse

# Asegurar codificación UTF-8 en terminales Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ==========================================
# 1. PARÁMETROS BASE Y DISPOSITIVO
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if os.path.exists(os.path.join(SCRIPT_DIR, "dataset_curado")):
    DEFAULT_DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset_curado")
elif os.path.exists(os.path.join(SCRIPT_DIR, "behavior", "dataset_curado")):
    DEFAULT_DATASET_DIR = os.path.join(SCRIPT_DIR, "behavior", "dataset_curado")
else:
    DEFAULT_DATASET_DIR = os.path.join(SCRIPT_DIR, "behavior", "dataset_curado")

DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, "runs", "behavior")

NUM_SEGMENTS = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 2. GESTIÓN SECUENCIAL DE EXPERIMENTOS
# ==========================================
def obtener_directorio_experimento(base_dir: str) -> str:
    """
    Crea de forma secuencial el directorio para el nuevo experimento (exp1, exp2, ...).
    """
    os.makedirs(base_dir, exist_ok=True)
    directorios_previos = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("exp")
    ]

    numeros = []
    for d in directorios_previos:
        try:
            num = int(d.replace("exp", ""))
            numeros.append(num)
        except ValueError:
            pass

    siguiente = max(numeros) + 1 if numeros else 1
    exp_dir = os.path.join(base_dir, f"exp{siguiente}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


# ==========================================
# 3. DIVISIÓN ESTRATIFICADA POR VIDEO
# ==========================================
def extraer_identificador_video(clip_path: str) -> str:
    """
    Extrae el nombre base del video de origen para evitar fuga de información
    entre clips continuos en los conjuntos de entrenamiento y validación.
    """
    folder_name = os.path.basename(clip_path)
    for etiqueta in ["_clip", "_hardneg", "_normal"]:
        if etiqueta in folder_name:
            return folder_name.split(etiqueta)[0]
    return folder_name


def agrupar_clips_por_video(clip_paths: List[str]) -> defaultdict:
    """Agrupa las rutas de clips según el identificador de su video de origen."""
    grupos = defaultdict(list)
    for ruta in clip_paths:
        vid_id = extraer_identificador_video(ruta)
        grupos[vid_id].append(ruta)
    return grupos


def division_estratificada_grupos(
    violent_clips: List[str],
    normal_clips: List[str],
    train_ratio: float = 0.8
) -> Tuple[List[str], List[int], List[str], List[int], Set[str], Set[str]]:
    """
    Realiza una partición de los clips garantizando que los clips provenientes
    de un mismo video no coexistan en entrenamiento y validación.
    """
    violent_groups = agrupar_clips_por_video(violent_clips)
    normal_groups = agrupar_clips_por_video(normal_clips)

    violent_vids = set(violent_groups.keys())
    pure_normal_vids = set(normal_groups.keys()) - violent_vids

    v_vids_list = list(violent_vids)
    random.shuffle(v_vids_list)
    v_split = int(len(v_vids_list) * train_ratio)
    train_v_vids = set(v_vids_list[:v_split])
    val_v_vids = set(v_vids_list[v_split:])

    pn_vids_list = list(pure_normal_vids)
    random.shuffle(pn_vids_list)
    pn_split = int(len(pn_vids_list) * train_ratio)
    train_pn_vids = set(pn_vids_list[:pn_split])
    val_pn_vids = set(pn_vids_list[pn_split:])

    train_vids = train_v_vids.union(train_pn_vids)
    val_vids = val_v_vids.union(val_pn_vids)

    train_clips, train_labels = [], []
    val_clips, val_labels = [], []

    for vid, clips in violent_groups.items():
        if vid in train_vids:
            train_clips.extend(clips)
            train_labels.extend([1] * len(clips))
        else:
            val_clips.extend(clips)
            val_labels.extend([1] * len(clips))

    for vid, clips in normal_groups.items():
        if vid in train_vids:
            train_clips.extend(clips)
            train_labels.extend([0] * len(clips))
        else:
            val_clips.extend(clips)
            val_labels.extend([0] * len(clips))

    return train_clips, train_labels, val_clips, val_labels, train_vids, val_vids


# ==========================================
# 4. FUNCIÓN DE PÉRDIDA Y ARQUITECTURA TSM
# ==========================================
class BalancedFocalLoss(nn.Module):
    """
    Función de pérdida Focal Loss con ponderación de clase y suavizado de etiquetas.
    Enfoca la optimización en muestras ambiguas o de difícil clasificación.
    """
    def __init__(self, gamma: float = 1.5, pos_weight: float = 1.35, label_smoothing: float = 0.10, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = inputs.size(1)
        log_probs = torch.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        smooth_targets = torch.full_like(inputs, fill_value=self.label_smoothing / (num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        weights = torch.ones_like(targets, dtype=torch.float32)
        weights[targets == 1] = self.pos_weight

        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = torch.pow(1.0 - pt, self.gamma) * weights

        loss = -torch.sum(smooth_targets * log_probs, dim=1) * focal_weight
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


class TemporalShift(nn.Module):
    """
    Módulo de desplazamiento temporal (TSM). Desplaza una fracción de los canales
    hacia adelante y hacia atrás en la dimensión temporal sin costo computacional adicional.
    """
    def __init__(self, net: nn.Module, n_segment: int = 16, fold_div: int = 8):
        super().__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shift(x, self.n_segment, fold_div=self.fold_div)
        return self.net(x)

    @staticmethod
    def shift(x: torch.Tensor, n_segment: int, fold_div: int = 8) -> torch.Tensor:
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]
        out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]
        out[:, :, 2*fold:] = x[:, :, 2*fold:]

        return out.view(nt, c, h, w)


def construir_modelo_tsm(
    num_classes: int = 2,
    n_segment: int = 16,
    dropout_prob: float = 0.6,
    unfreeze_layers: Tuple[str, ...] = ('layer3', 'layer4')
) -> nn.Module:
    """
    Construye la red ResNet-18 con módulos TSM insertados en cada bloque residual.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Congelar capas convolucionales tempranas y descongelar las especificadas
    for name, param in model.named_parameters():
        if any(layer in name for layer in unfreeze_layers):
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Insertar TSM en los 4 bloques convolucionales
    model.layer1 = TemporalShift(model.layer1, n_segment=n_segment)
    model.layer2 = TemporalShift(model.layer2, n_segment=n_segment)
    model.layer3 = TemporalShift(model.layer3, n_segment=n_segment)
    model.layer4 = TemporalShift(model.layer4, n_segment=n_segment)

    # Cabezal de clasificación con regularización Dropout
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_prob),
        nn.Linear(in_features, num_classes)
    )
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


# ==========================================
# 5. DATASET Y TRANSFORMACIONES TEMPORALES
# ==========================================
class VideoClipDataset(Dataset):
    """
    Conjunto de datos para secuencias de video. Aplica transformaciones espaciotemporales
    consistentes a lo largo de todos los fotogramas del clip.
    """
    def __init__(self, clip_paths: List[str], labels: List[int], is_train: bool = True):
        self.clip_paths = clip_paths
        self.labels = labels
        self.is_train = is_train

        if self.is_train:
            self.color_jitter = transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.10)

    def __len__(self) -> int:
        return len(self.clip_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        clip_dir = self.clip_paths[idx]
        label = self.labels[idx]

        frame_files = sorted(glob.glob(os.path.join(clip_dir, "*.jpg")))[:NUM_SEGMENTS]
        images = [Image.open(f).convert("RGB") for f in frame_files]

        do_flip = self.is_train and (random.random() > 0.5)
        do_grayscale = self.is_train and (random.random() < 0.15)
        do_cutout = self.is_train and (random.random() < 0.30)

        if self.is_train:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                images[0], scale=(0.65, 1.0), ratio=(0.85, 1.15)
            )
            angle = random.uniform(-6.0, 6.0) if random.random() > 0.5 else 0.0

            if do_cutout:
                cut_h = random.randint(30, 60)
                cut_w = random.randint(30, 60)
                cut_y = random.randint(0, 224 - cut_h)
                cut_x = random.randint(0, 224 - cut_w)

        processed_frames = []
        for img in images:
            if self.is_train:
                img = F.crop(img, i, j, h, w)
                if angle != 0.0:
                    img = F.rotate(img, angle)
                img = self.color_jitter(img)
                if do_grayscale:
                    img = F.to_grayscale(img, num_output_channels=3)
                if do_flip:
                    img = F.hflip(img)

            img = F.resize(img, (224, 224))
            img_tensor = F.to_tensor(img)

            if self.is_train and do_cutout:
                img_tensor[:, cut_y:cut_y+cut_h, cut_x:cut_x+cut_w] = 0.0

            img_tensor = F.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            processed_frames.append(img_tensor)

        clip_tensor = torch.stack(processed_frames)
        return clip_tensor, label


# ==========================================
# 6. PIPELINE DE ENTRENAMIENTO
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento supervisado de TSM ResNet-18 para detección de violencia y conducta hostil."
    )
    parser.add_argument("--dataset_dir", type=str, default=DEFAULT_DATASET_DIR, help="Ruta al directorio de clips procesados")
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR, help="Ruta base para almacenamiento de experimentos")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamaño de lote por dispositivo")
    parser.add_argument("--grad_accum", type=int, default=2, help="Pasos de acumulación de gradiente (lote efectivo = batch_size * grad_accum)")
    parser.add_argument("--epochs", type=int, default=25, help="Número de épocas de entrenamiento")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Épocas de calentamiento lineal de learning rate")
    parser.add_argument("--patience", type=int, default=18, help="Épocas de paciencia para Early Stopping (0 para desactivar)")
    parser.add_argument("--lr_backbone", type=float, default=3.5e-6, help="Tasa de aprendizaje para el extractor convolucional")
    parser.add_argument("--lr_fc", type=float, default=1.8e-4, help="Tasa de aprendizaje para el clasificador lineal")
    parser.add_argument("--unfreeze", type=str, default="layer3,layer4", help="Capas residuales a descongelar separadas por coma")
    parser.add_argument("--pos_weight", type=float, default=1.35, help="Ponderación para la clase violenta en Focal Loss")
    parser.add_argument("--focal_gamma", type=float, default=1.5, help="Parámetro gamma de Focal Loss")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Coeficiente de regularización L2")
    parser.add_argument("--dropout", type=float, default=0.60, help="Probabilidad de dropout en cabezal clasificador")
    parser.add_argument("--label_smoothing", type=float, default=0.10, help="Factor de suavizado de etiquetas")
    parser.add_argument("--num_workers", type=int, default=8, help="Hilos paralelos para DataLoader")
    args = parser.parse_args()

    print(f"[INFO] Dispositivo seleccionado: {DEVICE}")
    random.seed(42)
    torch.manual_seed(42)

    exp_dir = obtener_directorio_experimento(args.results_dir)
    print(f"[INFO] Directorio de salida: {exp_dir}")
    print(f"[INFO] Configuración: Batch={args.batch_size} (Efectivo={args.batch_size * args.grad_accum}), "
          f"Épocas={args.epochs}, LR_Backbone={args.lr_backbone:.2e}, LR_FC={args.lr_fc:.2e}")

    violent_dir = os.path.join(args.dataset_dir, "violent")
    normal_dir = os.path.join(args.dataset_dir, "normal")

    violent_clips = [entry.path for entry in os.scandir(violent_dir) if entry.is_dir()] if os.path.exists(violent_dir) else []
    all_normal_clips = [entry.path for entry in os.scandir(normal_dir) if entry.is_dir()] if os.path.exists(normal_dir) else []

    if not violent_clips or not all_normal_clips:
        print(f"[ERROR] No se encontraron clips en '{args.dataset_dir}'.")
        print("        Asegúrate de que existan las subcarpetas 'violent' y 'normal' con sus respectivos clips.")
        sys.exit(1)

    normal_groups = agrupar_clips_por_video(all_normal_clips)
    normal_vids = list(normal_groups.keys())
    random.shuffle(normal_vids)

    target_normal_count = int(len(violent_clips) * 1.0)
    selected_normal_clips = []
    for vid in normal_vids:
        if len(selected_normal_clips) >= target_normal_count:
            break
        selected_normal_clips.extend(normal_groups[vid])

    # División estratificada por video
    train_clips, train_labels, val_clips, val_labels, train_vids, val_vids = division_estratificada_grupos(
        violent_clips, selected_normal_clips, train_ratio=0.8
    )

    # Balanceo del conjunto de validación
    val_violent_indices = [i for i, l in enumerate(val_labels) if l == 1]
    val_normal_indices = [i for i, l in enumerate(val_labels) if l == 0]
    random.shuffle(val_normal_indices)
    val_normal_indices = val_normal_indices[:len(val_violent_indices)]

    selected_val_indices = val_violent_indices + val_normal_indices
    val_clips = [val_clips[i] for i in selected_val_indices]
    val_labels = [val_labels[i] for i in selected_val_indices]

    print(f"\n[INFO] Distribución de muestras:")
    print(f"  - Entrenamiento: {len(train_clips)} clips ({sum(train_labels)} violentos, {len(train_labels)-sum(train_labels)} normales) de {len(train_vids)} videos")
    print(f"  - Validación:    {len(val_clips)} clips ({sum(val_labels)} violentos, {len(val_labels)-sum(val_labels)} normales) de {len(val_vids)} videos")

    # Muestreo balanceado
    train_combined = list(zip(train_clips, train_labels))
    random.shuffle(train_combined)
    train_clips, train_labels = zip(*train_combined)

    val_combined = list(zip(val_clips, val_labels))
    random.shuffle(val_combined)
    val_clips, val_labels = zip(*val_combined)

    class_counts = [train_labels.count(0), train_labels.count(1)]
    class_weights = [1.0 / count for count in class_counts]
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_dataset = VideoClipDataset(train_clips, train_labels, is_train=True)
    val_dataset = VideoClipDataset(val_clips, val_labels, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )

    unfreeze_list = tuple(l.strip() for l in args.unfreeze.split(",") if l.strip())
    model = construir_modelo_tsm(
        num_classes=2,
        n_segment=NUM_SEGMENTS,
        dropout_prob=args.dropout,
        unfreeze_layers=unfreeze_list
    ).to(DEVICE)

    criterion = BalancedFocalLoss(
        gamma=args.focal_gamma,
        pos_weight=args.pos_weight,
        label_smoothing=args.label_smoothing
    ).to(DEVICE)

    params_backbone = [p for n, p in model.named_parameters() if "fc" not in n and p.requires_grad]
    params_fc = [p for n, p in model.named_parameters() if "fc" in n and p.requires_grad]

    optimizer = optim.AdamW([
        {"params": params_backbone, "lr": args.lr_backbone, "weight_decay": args.weight_decay},
        {"params": params_fc, "lr": args.lr_fc, "weight_decay": args.weight_decay * 0.5}
    ])

    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_val_preds = []
    best_val_targets = []
    patience_counter = 0

    print("\n[INFO] Iniciando ciclo de entrenamiento...")

    for epoch in range(args.epochs):
        # Ajuste lineal de learning rate durante warmup
        if epoch < args.warmup_epochs:
            warmup_factor = (epoch + 1) / args.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group.get('initial_lr', param_group['lr']) * warmup_factor

        # Fase de entrenamiento
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        optimizer.zero_grad(set_to_none=True)

        for step, (inputs, targets) in enumerate(train_loader):
            b, t, c, h, w = inputs.size()
            inputs = inputs.view(b * t, c, h, w).to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets) / args.grad_accum
                scaler.scale(loss).backward()

                if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets) / args.grad_accum
                loss.backward()

                if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * args.grad_accum * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        # Fase de validación
        model.eval()
        val_loss_running = 0.0
        val_correct = 0
        val_total = 0
        epoch_val_preds = []
        epoch_val_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                b, t, c, h, w = inputs.size()
                inputs = inputs.view(b * t, c, h, w).to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)

                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                val_loss_running += loss.item() * targets.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

                epoch_val_preds.extend(predicted.cpu().numpy().tolist())
                epoch_val_targets.extend(targets.cpu().numpy().tolist())

        val_loss = val_loss_running / val_total
        val_acc = 100.0 * val_correct / val_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Época [{epoch+1:02d}/{args.epochs:02d}] | "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Comprobar mejora y registrar punto de control
        es_mejor = False
        if val_acc > best_val_acc:
            es_mejor = True
        elif abs(val_acc - best_val_acc) < 0.25 and val_loss < best_val_loss:
            es_mejor = True

        if es_mejor:
            best_val_loss = min(best_val_loss, val_loss)
            best_val_acc = max(best_val_acc, val_acc)
            best_val_preds = epoch_val_preds
            best_val_targets = epoch_val_targets
            patience_counter = 0

            torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pth"))
            torch.save(model.state_dict(), os.path.join(exp_dir, "comportamiento.pth"))
            print(f"  --> [MEJOR MODELO] Guardado en {exp_dir} (Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                print(f"\n[INFO] Early Stopping activado tras {args.patience} épocas sin mejora.")
                break

    # Generación de gráficos analíticos
    print(f"\n[INFO] Generando curvas de aprendizaje y matriz de confusión en: {exp_dir}")
    epochs_range = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs_range, history['train_loss'], label='Train Loss', color='tab:red', linewidth=2)
    ax1.plot(epochs_range, history['val_loss'], label='Val Loss', color='tab:orange', linewidth=2, linestyle='--')
    ax1.set_title('Pérdida por Época')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, history['train_acc'], label='Train Acc', color='tab:blue', linewidth=2)
    ax2.plot(epochs_range, history['val_acc'], label='Val Acc', color='tab:green', linewidth=2, linestyle='--')
    ax2.set_title('Precisión (%) por Época')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "training_curves.png"), dpi=300)
    plt.close()

    if best_val_targets and best_val_preds:
        cm = confusion_matrix(best_val_targets, best_val_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Violento"])

        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
        plt.title(f'Matriz de Confusión ({os.path.basename(exp_dir)} - Val Acc: {best_val_acc:.2f}%)')
        plt.savefig(os.path.join(exp_dir, "confusion_matrix.png"), dpi=300)
        plt.close()

        report = classification_report(best_val_targets, best_val_preds, target_names=["Normal", "Violento"])
        print("\nREPORTE FINAL DE CLASIFICACIÓN:")
        print(report)

        with open(os.path.join(exp_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
            f.write(f"Resultados del Experimento: {os.path.basename(exp_dir)}\n")
            f.write(f"Mejor Val Loss: {best_val_loss:.4f}\n")
            f.write(f"Mejor Val Acc: {best_val_acc:.2f}%\n\n")
            f.write(report)

    print(f"\n[INFO] Entrenamiento completado. Artefactos exportados a: {exp_dir}")


if __name__ == "__main__":
    main()
