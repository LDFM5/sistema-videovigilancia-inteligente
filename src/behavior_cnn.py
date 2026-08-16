"""
behavior_cnn.py

Pipeline de inferencia perimetral para el modelo de comportamiento ResNet-18 + TSM (DAOCS).
Optimizado con preprocesamiento matricial vectorizado en GPU y carga limpia de pesos.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

# =========================================================================
# ARQUITECTURA TSM (RESNET-18 CON TEMPORAL SHIFT MODULE)
# =========================================================================

class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=16, fold_div=8):
        super().__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div)
        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div=8):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]
        out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]
        out[:, :, 2*fold:] = x[:, :, 2*fold:]

        return out.view(nt, c, h, w)


class ViolenceNetTSM(nn.Module):
    def __init__(self, num_classes=2, n_segment=16, dropout_prob=0.4):
        super().__init__()
        self.n_segment = n_segment

        # Red convolucional base alineada con el entrenamiento
        self.model = models.resnet18(weights=None)
        self.model.layer1 = TemporalShift(self.model.layer1, n_segment=n_segment)
        self.model.layer2 = TemporalShift(self.model.layer2, n_segment=n_segment)
        self.model.layer3 = TemporalShift(self.model.layer3, n_segment=n_segment)
        self.model.layer4 = TemporalShift(self.model.layer4, n_segment=n_segment)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(in_features, num_classes)
        )

        # Constantes reutilizables: evita crear y transferir dos tensores a la
        # GPU en cada evaluación. No forman parte del checkpoint entrenado.
        self.register_buffer(
            "input_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "input_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, x):
        # x shape: (Batch, Time, Channels, Height, Width)
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)

        out = self.model(x)  # Shape: (B * T, num_classes)
        out = out.view(b, t, -1).mean(dim=1)  # Shape: (B, num_classes)
        return out


# =========================================================================
# PIPELINE DE INFERENCIA EN TIEMPO REAL Y PREPROCESAMIENTO VECTORIZADO
# =========================================================================

def cargar_modelo_violencia(ruta_modelo, dispositivo):
    """
    Instancia ViolenceNetTSM y realiza una limpieza de llaves para evitar discrepancias de prefijos.
    """
    if not os.path.exists(ruta_modelo):
        print(f"\n[ERROR DE RUTA] NO SE ENCONTRÓ EL ARCHIVO EN -> {ruta_modelo}\n")
        return None

    try:
        modelo = ViolenceNetTSM(num_classes=2, n_segment=16, dropout_prob=0.4)
        
        # Carga de checkpoints
        try:
            state_dict = torch.load(ruta_modelo, map_location=dispositivo, weights_only=False)
        except TypeError:
            state_dict = torch.load(ruta_modelo, map_location=dispositivo)

        if isinstance(state_dict, dict):
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]

        # Limpieza automatizada de prefijos (backbone. o model.)
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("backbone.", "").replace("model.", "")
            cleaned_state_dict[new_key] = value

        modelo.model.load_state_dict(cleaned_state_dict)
        modelo.to(dispositivo)
        modelo.eval()
        print(f"\n[SISTEMA IA] Modelo ResNet-18 + TSM cargado exitosamente desde: {os.path.basename(ruta_modelo)}\n")
        return modelo
    except Exception as e:
        print(f"\n[ERROR EN PYTORCH] FALLO AL CARGAR PESOS .PTH -> {e}\n")
        return None


def evaluar_secuencia_violencia(modelo, dispositivo, buffer_fotogramas, umbral_confianza=0.50):
    """
    Extrae 16 fotogramas y ejecuta la inferencia con normalización ImageNet en GPU.
    """
    if modelo is None or len(buffer_fotogramas) < 2:
        return "NORMAL", 0.0

    buffer_list = list(buffer_fotogramas)
    timed_buffer = (
        isinstance(buffer_list[0], tuple)
        and len(buffer_list[0]) == 2
        and isinstance(buffer_list[0][0], (int, float))
    )

    # 1. Muestreo de 16 instantes uniformes. Con timestamps, una cámara lenta
    # puede repetir frames sin alterar el periodo temporal representado.
    if timed_buffer:
        timestamps = np.asarray([item[0] for item in buffer_list], dtype=np.float64)
        if timestamps[-1] - timestamps[0] < 0.5:
            return "NORMAL", 0.0

        target_times = np.linspace(timestamps[0], timestamps[-1], num=16)
        indices = np.searchsorted(timestamps, target_times, side="right") - 1
        indices = np.clip(indices, 0, len(buffer_list) - 1)
        source_frames = [item[1] for item in buffer_list]
    else:
        if len(buffer_list) < 16:
            return "NORMAL", 0.0
        indices = np.linspace(0, len(buffer_list) - 1, num=16, dtype=int)
        source_frames = buffer_list

    # 2. Resizing y apilado
    cuadros_muestreados = []
    for idx in indices:
        f = source_frames[int(idx)]
        if f.shape[0] != 224 or f.shape[1] != 224:
            f = cv2.resize(f, (224, 224), interpolation=cv2.INTER_LINEAR)
        cuadros_muestreados.append(f)

    batch_numpy = np.stack(cuadros_muestreados)
    batch_rgb = batch_numpy[..., ::-1].copy()
    batch_chw = np.transpose(batch_rgb, (0, 3, 1, 2))

    tensor_uint8 = torch.from_numpy(batch_chw).unsqueeze(0).to(dispositivo)

    # 3. Normalización ImageNet en GPU
    tensor_float = tensor_uint8.float() / 255.0
    tensor_entrada = (tensor_float - modelo.input_mean) / modelo.input_std

    # 4. Inferencia
    with torch.inference_mode():
        with torch.amp.autocast('cuda', enabled=(dispositivo.type == 'cuda')):
            salida_logits = modelo(tensor_entrada)
            probabilidades = torch.softmax(salida_logits, dim=1).squeeze(0)

        probabilidad_violencia = probabilidades[1].item()

    if probabilidad_violencia >= umbral_confianza:
        return "VIOLENCE", probabilidad_violencia

    return "NORMAL", probabilidad_violencia
