"""
behavior_cnn.py

Inferencia del modelo de comportamiento ResNet-18 + TSM de DAOCS.
Utiliza preprocesamiento matricial en GPU y normaliza las claves del archivo de pesos.
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
        print(f"\n[ERROR] No se encontró el modelo en {ruta_modelo}.\n")
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
        print(f"\n[INFO] Modelo ResNet-18 + TSM cargado desde {os.path.basename(ruta_modelo)}.\n")
        return modelo
    except Exception as e:
        print(f"\n[ERROR] No se pudo cargar el modelo de comportamiento: {e}\n")
        return None


def preprocesar_buffer_a_tensor(buffer_fotogramas):
    """
    Extrae 16 fotogramas uniformes en el tiempo y devuelve un array numpy (16, 3, 224, 224)
    en formato RGB listo para empaquetar por lotes (Batch) en GPU.
    """
    buffer_list = list(buffer_fotogramas)
    if len(buffer_list) < 2:
        return None

    timed_buffer = (
        isinstance(buffer_list[0], tuple)
        and len(buffer_list[0]) == 2
        and isinstance(buffer_list[0][0], (int, float))
    )

    if timed_buffer:
        timestamps = np.asarray([item[0] for item in buffer_list], dtype=np.float64)
        if timestamps[-1] - timestamps[0] < 0.4:
            return None

        target_times = np.linspace(timestamps[0], timestamps[-1], num=16)
        indices = np.searchsorted(timestamps, target_times, side="right") - 1
        indices = np.clip(indices, 0, len(buffer_list) - 1)
        source_frames = [item[1] for item in buffer_list]
    else:
        if len(buffer_list) < 16:
            return None
        indices = np.linspace(0, len(buffer_list) - 1, num=16, dtype=int)
        source_frames = buffer_list

    cuadros = []
    for idx in indices:
        f = source_frames[int(idx)]
        h, w = f.shape[:2]
        if h != 224 or w != 224:
            f_resized = cv2.resize(f, (224, 224), interpolation=cv2.INTER_LINEAR)
        else:
            f_resized = f
        cuadros.append(f_resized)

    # Convertir BGR a RGB y transponer a (16, 3, 224, 224)
    tensor_np = np.transpose(np.stack(cuadros)[..., ::-1].copy(), (0, 3, 1, 2))
    return tensor_np


def evaluar_secuencias_violencia_batch(modelo, dispositivo, lista_items):
    """
    Procesa de forma simultánea múltiples secuencias de cámaras en un solo lote (Batch) en GPU.
    lista_items: Lista de tuplas (cam_key, buffer_fotogramas, umbral_confianza)
    Retorna: Lista de tuplas (cam_key, clase, probabilidad_violencia)
    """
    if modelo is None or not lista_items:
        return []

    valid_cams = []
    valid_tensors = []
    umbrales = []

    for cam_key, buffer_fotogramas, umbral in lista_items:
        tensor_cam = preprocesar_buffer_a_tensor(buffer_fotogramas)
        if tensor_cam is not None:
            valid_cams.append(cam_key)
            valid_tensors.append(tensor_cam)  # (16, 3, 224, 224)
            umbrales.append(umbral)
        else:
            valid_cams.append(cam_key)
            valid_tensors.append(None)
            umbrales.append(umbral)

    indices_validos = [i for i, t in enumerate(valid_tensors) if t is not None]
    resultados = [(valid_cams[i], "NORMAL", 0.0) for i in range(len(valid_cams))]

    if not indices_validos:
        return resultados

    # Apilar todas las secuencias válidas en un único tensor de lote: (N_validos, 16, 3, 224, 224)
    batch_tensors = np.stack([valid_tensors[i] for i in indices_validos], axis=0)

    tensor_uint8 = torch.from_numpy(batch_tensors).to(dispositivo)
    tensor_float = tensor_uint8.float() / 255.0
    tensor_entrada = (tensor_float - modelo.input_mean) / modelo.input_std

    with torch.inference_mode():
        with torch.amp.autocast('cuda', enabled=(dispositivo.type == 'cuda')):
            salida_logits = modelo(tensor_entrada)  # (N, 2)
            probabilidades = torch.softmax(salida_logits, dim=1)

        probs_violencia = probabilidades[:, 1].tolist()

    for idx_batch, orig_idx in enumerate(indices_validos):
        p_violencia = probs_violencia[idx_batch]
        umbral = umbrales[orig_idx]
        clase = "VIOLENCE" if p_violencia >= umbral else "NORMAL"
        resultados[orig_idx] = (valid_cams[orig_idx], clase, p_violencia)

    return resultados


def evaluar_secuencia_violencia(modelo, dispositivo, buffer_fotogramas, umbral_confianza=0.50):
    """
    Wrapper de compatibilidad para evaluación individual de una sola cámara o clip.
    """
    res = evaluar_secuencias_violencia_batch(
        modelo, dispositivo, [("CAM", buffer_fotogramas, umbral_confianza)]
    )
    if res:
        _, clase, score = res[0]
        return clase, score
    return "NORMAL", 0.0
