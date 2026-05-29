"""
behavior.py

Subsistema de clasificación de comportamiento secuencial y análisis temporal.
Utiliza redes neuronales recurrentes (GRU) para la evaluación de trayectorias de esqueleto.
"""

import torch
import torch.nn as nn
import numpy as np


# =========================================================================
# 1. ARQUITECTURA DE LA RED NEURONAL RECURRENTE (GRU)
# =========================================================================
class ActionGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ActionGRU, self).__init__()
        self.hidden_size = hidden_size
        
        # Capa GRU recurrente estándar
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x shape esperado: (batch, sequence_length, input_size)
        out, _ = self.gru(x)
        # Aislamiento del último estado temporal de la secuencia estructurada
        out = self.fc(out[:, -1, :]) 
        return out


# =========================================================================
# 2. CARGADOR ASÍNCRONO DEL MODELO
# =========================================================================
def load_behavior_model(pth_path, input_size=34, hidden_size=64, num_classes=2):
    """
    Carga los pesos binarios del modelo GRU y lo aprovisiona en el hardware seleccionado.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ActionGRU(input_size, hidden_size, num_classes)
    
    # map_location garantiza el mapeo cruzado seguro entre entornos CUDA y CPU
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.to(device)
    model.eval() # Modo de inferencia estricto (Desactiva Dropout y cálculo de gradientes)
    
    print(f"BEHAVIOR_CLASSIFIER: SEQUENTIAL_MODEL_DEPLOYED -> INSTANCE_DEVICE: {str(device).upper()}")
    return model, device


# =========================================================================
# 3. LÓGICA DE INFERENCIA DE COMPORTAMIENTOS ANÓMALOS
# =========================================================================
def predict_behavior(gru_model, device, buffer_esqueletos, clases_nombres):
    """
    Transforma la traza histórica indexada del búfer circular a tensores nativos de PyTorch.
    Efectúa la predicción matricial de comportamiento en tiempo de ejecución.
    """
    # Transformación del buffer circular (deque) a matriz geométrica
    # Shape resultante: (sequence_length, input_size)
    seq_array = np.array(buffer_esqueletos)
    
    # Inyección de dimensión de lote (Batch sizing): (1, seq_len, input_size)
    seq_tensor = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad(): # Optimización de hardware para máxima velocidad
        outputs = gru_model(seq_tensor)
        _, predicted = torch.max(outputs.data, 1)
        clase_idx = predicted.item()
        
    return clases_nombres[clase_idx]