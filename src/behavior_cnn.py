"""
behavior_cnn.py

Pipeline de inferencia perimetral para el modelo de comportamiento CNN + GRU (ViolenceNet).
Optimizado con preprocesamiento matricial vectorizado en GPU y carga segura de pesos.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)

# =========================================================================
# ESTRUCTURA DE LA RED NEURONAL (REQUERIDA POR PYTORCH PARA CARGAR EL .PTH)
# =========================================================================

class MobileNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Se cargan los pesos base de ImageNet acordes al entrenamiento
        backbone = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 576

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x


class ViolenceNet(nn.Module):
    def __init__(self, hidden_size=64, num_classes=2):
        super().__init__()
        self.cnn = MobileNetFeatureExtractor()
        self.gru = nn.GRU(
            input_size=self.cnn.feature_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Colapsar batch y tiempo para procesar los fotogramas en lote por la CNN
        x = x.view(
            batch_size * seq_len,
            x.shape[2],
            x.shape[3],
            x.shape[4]
        )

        features = self.cnn(x)

        # Restaurar dimensión de secuencia para el análisis temporal de la GRU
        features = features.view(
            batch_size,
            seq_len,
            -1
        )

        gru_out, _ = self.gru(features)
        
        # Extraer únicamente el estado oculto del último fotograma de la secuencia
        last_out = gru_out[:, -1, :]
        
        out = self.fc(last_out)
        return out


# =========================================================================
# PIPELINE DE INFERENCIA EN TIEMPO REAL Y PREPROCESAMIENTO VECTORIZADO
# =========================================================================

def cargar_modelo_violencia(ruta_modelo, dispositivo):
    """
    Instancia la arquitectura ViolenceNet y carga los pesos binarios de forma segura.
    """
    if not os.path.exists(ruta_modelo):
        print(f"ERROR_MODELO_COMPORTAMIENTO: NO SE ENCONTRÓ EL ARCHIVO EN -> {ruta_modelo}")
        return None

    try:
        modelo = ViolenceNet()
        state_dict = torch.load(ruta_modelo, map_location=dispositivo, weights_only=True)
        
        # Flexibilidad ante diferentes formatos de guardado de checkpoints
        if isinstance(state_dict, dict):
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]

        modelo.load_state_dict(state_dict)
        modelo.to(dispositivo)
        modelo.eval()
        return modelo
    except Exception as e:
        print(f"ERROR_MODELO_COMPORTAMIENTO: FALLO AL CARGAR PESOS .PTH -> {str(e).upper()}")
        return None


def evaluar_secuencia_violencia(modelo, dispositivo, buffer_fotogramas, umbral_confianza=0.50):
    """
    Extrae exactamente 16 fotogramas del búfer temporal y ejecuta la inferencia
    utilizando preprocesamiento matricial vectorizado de alta velocidad.
    """
    if modelo is None or len(buffer_fotogramas) < 16:
        return "NORMAL", 0.0

    # 1. Muestreo equidistante uniforme de 16 índices
    indices = np.linspace(0, len(buffer_fotogramas) - 1, num=16, dtype=int)
    
    # 2. Extracción y apilado en bloque: Forma (16, H, W, C)
    cuadros_muestreados = []
    for idx in indices:
        f = buffer_fotogramas[idx]
        if f.shape[0] != 224 or f.shape[1] != 224:
            f = cv2.resize(f, (224, 224), interpolation=cv2.INTER_LINEAR)
        cuadros_muestreados.append(f)

    batch_numpy = np.stack(cuadros_muestreados)

    # 3. Conversión BGR -> RGB creando una copia continua en memoria (.copy())
    # 🎯 CORRECCIÓN: .copy() elimina las zancadas (strides) negativas para PyTorch
    batch_rgb = batch_numpy[..., ::-1].copy()

    # 4. Reordenar dimensiones: (T, H, W, C) -> (T, C, H, W)
    batch_chw = np.transpose(batch_rgb, (0, 3, 1, 2))

    # 5. Transferir uint8 a la GPU y añadir dimensión de Batch -> (1, 16, 3, 224, 224)
    tensor_uint8 = torch.from_numpy(batch_chw).unsqueeze(0).to(dispositivo)

    # 6. Normalización flotante realizada masivamente en paralelo dentro de la GPU/CUDA
    tensor_entrada = tensor_uint8.float() / 255.0

    # 7. Inferencia deshabilitando el cálculo de gradientes
    with torch.no_grad():
        salida_logits = modelo(tensor_entrada)
        probabilidades = torch.softmax(salida_logits, dim=1).squeeze(0)
        
        # Mapeo: Índice 0 = Normal, Índice 1 = Violence
        probabilidad_violencia = probabilidades[1].item()

    if probabilidad_violencia >= umbral_confianza:
        return "VIOLENCE", probabilidad_violencia
        
    return "NORMAL", probabilidad_violencia