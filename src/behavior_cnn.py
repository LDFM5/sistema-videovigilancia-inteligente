"""
behavior_cnn.py

Pipeline de inferencia perimetral para el modelo de comportamiento CNN + GRU.
Contiene la definición estructural del modelo para la carga de pesos binarios,
el muestreo equidistante, el formateo de tensores y la predicción de violencia.
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
        # Se cargan los mismos pesos base de ImageNet usados en el entrenamiento
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

        # Unir batch y tiempo para procesar los cuadros por la CNN por lote
        x = x.view(
            batch_size * seq_len,
            x.shape[2],
            x.shape[3],
            x.shape[4]
        )

        features = self.cnn(x)

        # Restaurar la dimensión de secuencia para el análisis temporal del GRU
        features = features.view(
            batch_size,
            seq_len,
            -1
        )

        gru_out, _ = self.gru(features)
        
        # Tomar únicamente el estado del último instante temporal (Footprint del clip)
        last_out = gru_out[:, -1, :]
        
        out = self.fc(last_out)
        return out


# =========================================================================
# PIPELINE DE INFERENCIA EN TIEMPO REAL Y CONTROL DE ENTORNO
# =========================================================================

def cargar_modelo_violencia(ruta_modelo, dispositivo):
    """
    Inicializa la red embebida e inyecta los pesos del entrenamiento en el hardware.
    """
    try:
        # Ahora la clase vive en este mismo archivo, por lo que se instancia directamente
        modelo = ViolenceNet()
        modelo.load_state_dict(torch.load(ruta_modelo, map_location=dispositivo))
        modelo.to(dispositivo)
        modelo.eval()
        return modelo
    except Exception as e:
        print(f"ERROR_MODELO_COMPORTAMIENTO: FALLO AL CARGAR PESOS .PTH -> {str(e).upper()}")
        return None


def preprocesar_cuadro_cnn(frame):
    """
    Adapta un cuadro individual de la cámara a las dimensiones nativas de la CNN.
    """
    # Escalar a la dimensión exacta de 224x224 píxeles
    frame_redimensionado = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
    
    # Conversión obligatoria de espacio de color OpenCV (BGR) a PyTorch/PIL (RGB)
    frame_rgb = cv2.cvtColor(frame_redimensionado, cv2.COLOR_BGR2RGB)
    
    # Llevar la matriz de enteros uint8 a flotantes escalados entre 0.0 y 1.0
    frame_normalizado = frame_rgb.astype(np.float32) / 255.0
    return frame_normalizado


def evaluar_secuencia_violencia(modelo, dispositivo, buffer_fotogramas, umbral_confianza=0.50):
    """
    Extrae exactamente 16 fotogramas equidistantes del búfer temporal y ejecuta la inferencia.
    """
    if modelo is None or len(buffer_fotogramas) < 16:
        return "NORMAL", 0.0

    # MUESTREO EQUIDISTANTE: Extraer 16 índices distribuidos de forma uniforme a lo largo del búfer
    indices = np.linspace(0, len(buffer_fotogramas) - 1, num=16, dtype=int)
    
    secuencia_procesada = []
    for idx in indices:
        cuadro_procesado = preprocesar_cuadro_cnn(buffer_fotogramas[idx])
        secuencia_procesada.append(cuadro_procesado)
    
    # Convertir a matriz de NumPy: Forma resultante (16, 224, 224, 3)
    matriz_secuencia = np.array(secuencia_procesada, dtype=np.float32)
    
    # Transponer dimensiones para cumplir con el estándar PyTorch (T, H, W, C) -> (T, C, H, W)
    matriz_transpuesta = np.transpose(matriz_secuencia, (0, 3, 1, 2))
    
    # Añadir dimensión de lote (Batch Dim) -> Forma final esperada por ViolenceNet: (1, 16, 3, 224, 224)
    tensor_entrada = torch.tensor(matriz_transpuesta, dtype=torch.float32).unsqueeze(0)
    tensor_entrada = tensor_entrada.to(dispositivo)
    
    with torch.no_grad():
        # Ejecutar la inferencia a nivel de hardware
        salida_logits = modelo(tensor_entrada)
        
        # Aplicar Softmax para mapear la salida a probabilidades de clase
        probabilidades = torch.softmax(salida_logits, dim=1).squeeze(0)
        
        # Mapeo: Índice 0 = Normal, Índice 1 = Violence
        probabilidad_violencia = probabilidades[1].item()
        
    if probabilidad_violencia >= umbral_confianza:
        return "VIOLENCE", probabilidad_violencia
        
    return "NORMAL", probabilidad_violencia